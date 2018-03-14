/*
 * mbsolve: Framework for solving the Maxwell-Bloch/-Lioville equations
 *
 * Copyright (c) 2016, Computational Photonics Group, Technical University of
 * Munich.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
 */

#define EIGEN_DONT_PARALLELIZE
#define EIGEN_NO_MALLOC

#include <unsupported/Eigen/MatrixFunctions>
#include <common_openmp.hpp>
#include <solver_openmp_2lvl_os_red.hpp>

namespace mbsolve{

static solver_factory<solver_openmp_2lvl_os_red> factory("openmp-2lvl-os-old");

/* redundant calculation overlap */
#ifdef XEON_PHI_OFFLOAD
__mb_on_device const unsigned int OL = 32;
#else
const unsigned int OL = 16;
#endif

solver_openmp_2lvl_os_red::solver_openmp_2lvl_os_red
(std::shared_ptr<const device> dev, std::shared_ptr<scenario> scen) :
    solver_int(dev, scen)
{
    /* TODO: scenario, device sanity check */

    /* TODO: solver params
     * courant number
     * overlap
     */

    Eigen::initParallel();
    Eigen::setNbThreads(1);

    if (dev->get_regions().size() == 0) {
        throw std::invalid_argument("No regions in device!");
    }

    /* determine simulation settings */
    init_fdtd_simulation(dev, scen, 0.5);

    /* set up simulaton constants */
    std::map<std::string, unsigned int> id_to_idx;
    std::vector<sim_constants_2lvl> materials;
    materials = init_sim_constants(dev, scen, id_to_idx);

    m_sim_consts.reserve(materials.size());

    /* prepare operator splitting */
    for (int i = 0; i < materials.size(); i++) {
        Eigen::Matrix3d L_0;

        L_0(0, 0) = - materials[i].gamma12;
        L_0(1, 1) = - materials[i].gamma12;
        L_0(2, 2) = - materials[i].tau1;
        L_0(0, 1) = + materials[i].w12;
        L_0(1, 0) = - materials[i].w12;
        m_sim_consts[i].prop_U02 = (L_0 * materials[i].d_t/2).exp();

        Eigen::Vector3d equi_rho(0, 0, materials[i].equi_inv);
        m_sim_consts[i].equi_corr = (Eigen::Matrix3d::Identity() -
                                     m_sim_consts[i].prop_U02) * equi_rho;

        Eigen::Matrix3d U_1;
        U_1(1, 2) = + 1.0;
        U_1(2, 1) = - 4.0;
        m_sim_consts[i].L_1E = U_1 * materials[i].d_t * materials[i].d12;

        /* copy other constants */
        m_sim_consts[i].M_CE = materials[i].M_CE;
        m_sim_consts[i].M_CH = materials[i].M_CH;
        m_sim_consts[i].M_CP = materials[i].M_CP;
        m_sim_consts[i].sigma = materials[i].sigma;
        m_sim_consts[i].w12 = materials[i].w12;
        m_sim_consts[i].d12 = materials[i].d12;
        m_sim_consts[i].tau1 = materials[i].tau1;
        m_sim_consts[i].gamma12 = materials[i].gamma12;
        m_sim_consts[i].equi_inv = materials[i].equi_inv;
        m_sim_consts[i].d_x_inv = materials[i].d_x_inv;
        m_sim_consts[i].d_t = materials[i].d_t;
        m_sim_consts[i].inversion_init = materials[i].inversion_init;
    }

    /* set up indices array and initialize data arrays */
    unsigned int P = omp_get_max_threads();

    std::cout << "Number of threads: " << P << std::endl;
    m_d = new Eigen::Vector3d*[P];
    m_e = new real*[P];
    m_h = new real*[P];
    m_mat_indices = new unsigned int*[P];

    unsigned int *l_mat_indices = new unsigned int[scen->get_num_gridpoints(0)];

    for (unsigned int i = 0; i < scen->get_num_gridpoints(0); i++) {
        unsigned int mat_idx = 0;
        real x = i * scen->get_gridpoint_size(0);

        for (const auto& reg : dev->get_regions()) {
            if ((x >= reg->get_start()) && (x <= reg->get_end())) {
                mat_idx = id_to_idx[reg->get_material()->get_id()];
                break;
            }
        }
        l_mat_indices[i] = mat_idx;
    }

    /* set up results and transfer data structures */
    unsigned int scratch_size = 0;
    for (const auto& rec : scen->get_records()) {
        /* create copy list entry */
        copy_list_entry entry(rec, scen, scratch_size);

        /* add result to solver */
        m_results.push_back(entry.get_result());

        /* calculate scratch size */
        scratch_size += entry.get_size();

        /* take imaginary part into account */
        if (rec->is_complex()) {
            scratch_size += entry.get_size();
        }

        /* TODO check if result is available */
        /*
           throw std::invalid_argument("Requested result is not available!");
        */

        m_copy_list.push_back(entry);
    }

    /* allocate scratchpad result memory */
    m_result_scratch = (real *) mb_aligned_alloc(sizeof(real) * scratch_size);
    m_scratch_size = scratch_size;

    /* create source data */
    m_source_data = new real[scen->get_num_timesteps() *
                             scen->get_sources().size()];
    unsigned int base_idx = 0;
    for (const auto& src : scen->get_sources()) {
        sim_source s;
        s.type = src->get_type();
        s.x_idx[0] = src->get_position(0)/scen->get_gridpoint_size(0);
        s.data_base_idx = base_idx;
        m_sim_sources.push_back(s);

        /* calculate source values */
        for (unsigned int j = 0; j < scen->get_num_timesteps(); j++) {
            m_source_data[base_idx + j] =
                src->get_value(j * scen->get_timestep_size());
        }

        base_idx += scen->get_num_timesteps();
    }

    unsigned int num_gridpoints = m_scenario->get_num_gridpoints(0);
    unsigned int chunk_base = m_scenario->get_num_gridpoints(0)/P;
    unsigned int chunk_rem = m_scenario->get_num_gridpoints(0) % P;
    unsigned int num_timesteps = m_scenario->get_num_timesteps();

#ifndef XEON_PHI_OFFLOAD
    l_copy_list = m_copy_list.data();
    l_sim_consts = m_sim_consts.data();
    l_sim_sources = m_sim_sources.data();
#else
    /* prepare to offload sources */
    unsigned int num_sources = m_sim_sources.size();
    l_sim_sources = new sim_source[num_sources];
    for (int i = 0; i < num_sources; i++) {
        l_sim_sources[i] = m_sim_sources[i];
    }

    /* prepare to offload simulation constants */
    l_sim_consts = new sim_constants_2lvl_os[m_sim_consts.size()];
    for (unsigned int i = 0; i < m_sim_consts.size(); i++) {
        l_sim_consts[i] = m_sim_consts[i];
    }

    /* prepare to offload copy list entries */
    unsigned int num_copy = m_copy_list.size();
    l_copy_list = new copy_list_entry_dev[num_copy];
    for (int i = 0; i < m_copy_list.size(); i++) {
        l_copy_list[i] = m_copy_list[i].get_dev();
    }

#pragma offload target(mic:0) in(P)                                     \
    in(num_sources, num_copy, chunk_base, chunk_rem)                    \
    in(l_mat_indices:length(num_gridpoints))                            \
    in(l_copy_list:length(num_copy) __mb_phi_create)                    \
    in(m_source_data:length(num_timesteps * num_sources) __mb_phi_create) \
    in(l_sim_sources:length(num_sources) __mb_phi_create)               \
    in(l_sim_consts:length(m_sim_consts.size()) __mb_phi_create)        \
    inout(m_e,m_h,m_d:length(P) __mb_phi_create)                        \
    inout(m_mat_indices:length(P) __mb_phi_create)
    {
#endif
        for (unsigned int tid = 0; tid < P; tid++) {
            unsigned int chunk = chunk_base;

            if (tid == P - 1) {
                chunk += chunk_rem;
            }

            /* allocation */
            unsigned int size = chunk + 2 * OL;


            m_d[tid] = (Eigen::Vector3d *)
                mb_aligned_alloc(size * sizeof(Eigen::Vector3d));
            m_h[tid] = (real *) mb_aligned_alloc(size * sizeof(real));
            m_e[tid] = (real *) mb_aligned_alloc(size * sizeof(real));
            m_mat_indices[tid] = (unsigned int *)
                mb_aligned_alloc(size * sizeof(unsigned int));
        }

#pragma omp parallel
        {
            /* TODO serial alloc necessary?
             *
             */

            unsigned int tid = omp_get_thread_num();
            unsigned int chunk = chunk_base;

            if (tid == P - 1) {
                chunk += chunk_rem;
            }

            /* allocation */
            unsigned int size = chunk + 2 * OL;

            Eigen::Vector3d *t_d;
            real *t_h, *t_e;
            unsigned int *t_mat_indices;

            t_d = m_d[tid];
            t_h = m_h[tid];
            t_e = m_e[tid];
            t_mat_indices = m_mat_indices[tid];

            __mb_assume_aligned(t_d);
            __mb_assume_aligned(t_e);
            __mb_assume_aligned(t_h);
            __mb_assume_aligned(t_mat_indices);

            for (int i = 0; i < size; i++) {
                unsigned int global_idx = tid * chunk_base + (i - OL);
                if ((global_idx >= 0) && (global_idx < num_gridpoints)) {
                    unsigned int mat_idx = l_mat_indices[global_idx];
                    t_mat_indices[i] = mat_idx;
                    t_d[i][2] = l_sim_consts[mat_idx].inversion_init;
                } else {
                    t_mat_indices[i] = 0;
                    t_d[i][2] = 0.0;
                }
                t_d[i][0] = 0.0;
                t_d[i][1] = 0.0;
                t_e[i] = 0.0;
                t_h[i] = 0.0;
            }
#pragma omp barrier
        }
#ifdef XEON_PHI_OFFLOAD
    }
#endif

    delete[] l_mat_indices;
}

solver_openmp_2lvl_os_red::~solver_openmp_2lvl_os_red()
{
    unsigned int P = omp_get_max_threads();
    unsigned int num_sources = m_sim_sources.size();
    unsigned int num_copy = m_copy_list.size();
    unsigned int num_gridpoints = m_scenario->get_num_gridpoints(0);
    unsigned int num_timesteps = m_scenario->get_num_timesteps();

#ifdef XEON_PHI_OFFLOAD
#pragma offload target(mic:0) in(P)                                     \
    in(num_sources, num_copy)                                           \
    in(l_copy_list:length(num_copy) __mb_phi_delete)                    \
    in(m_source_data:length(num_timesteps * num_sources) __mb_phi_delete) \
    in(l_sim_sources:length(num_sources) __mb_phi_delete)               \
    in(l_sim_consts:length(m_sim_consts.size()) __mb_phi_delete)        \
    in(m_e,m_h,m_d,m_mat_indices:length(P) __mb_phi_delete)
    {
#endif
#pragma omp parallel
        {
            unsigned int tid = omp_get_thread_num();

            mb_aligned_free(m_h[tid]);
            mb_aligned_free(m_e[tid]);
            mb_aligned_free(m_d[tid]);
            mb_aligned_free(m_mat_indices[tid]);
        }
#ifdef XEON_PHI_OFFLOAD
    }

    delete[] l_copy_list;
    delete[] l_sim_consts;
    delete[] l_sim_sources;
#endif

    mb_aligned_free(m_result_scratch);
    delete[] m_source_data;

    delete[] m_h;
    delete[] m_e;
    delete[] m_d;
    delete[] m_mat_indices;
}

const std::string&
solver_openmp_2lvl_os_red::get_name() const
{
    return factory.get_name();
}

void
solver_openmp_2lvl_os_red::run() const
{
    unsigned int P = omp_get_max_threads();
    unsigned int num_gridpoints = m_scenario->get_num_gridpoints(0);
    unsigned int chunk_base = m_scenario->get_num_gridpoints(0)/P;
    unsigned int chunk_rem = m_scenario->get_num_gridpoints(0) % P;
    unsigned int num_timesteps = m_scenario->get_num_timesteps();
    unsigned int num_sources = m_sim_sources.size();
    unsigned int num_copy = m_copy_list.size();

#ifdef XEON_PHI_OFFLOAD
#pragma offload target(mic:0) in(P)                                     \
    in(chunk_base, chunk_rem, num_gridpoints, num_timesteps)            \
    in(num_sources, num_copy)                                           \
    in(l_copy_list:length(num_copy) __mb_phi_use)                       \
    in(m_source_data:length(num_timesteps * num_sources) __mb_phi_use)  \
    in(l_sim_sources:length(num_sources) __mb_phi_use)                  \
    in(l_sim_consts:length(m_sim_consts.size()) __mb_phi_use)           \
    in(m_e,m_h,m_d,m_mat_indices:length(P) __mb_phi_use)                \
    inout(m_result_scratch:length(m_scratch_size))
    {
#endif
#pragma omp parallel
        {
            unsigned int tid = omp_get_thread_num();
            unsigned int chunk = chunk_base;
            if (tid == P - 1) {
                chunk += chunk_rem;
            }
            unsigned int size = chunk + 2 * OL;

            /* gather pointers */
            Eigen::Vector3d *t_d;
            real *t_h, *t_e;
            unsigned int *t_mat_indices;

            t_d = m_d[tid];
            t_h = m_h[tid];
            t_e = m_e[tid];
            t_mat_indices = m_mat_indices[tid];

            __mb_assume_aligned(t_d);
            __mb_assume_aligned(t_e);
            __mb_assume_aligned(t_h);
            __mb_assume_aligned(t_mat_indices);

            __mb_assume_aligned(m_result_scratch);

            /* gather prev and next pointers from other threads */
            Eigen::Vector3d *n_d, *p_d;
            real *n_h, *n_e;
            real *p_h, *p_e;

            __mb_assume_aligned(p_d);
            __mb_assume_aligned(p_e);
            __mb_assume_aligned(p_h);

            __mb_assume_aligned(n_d);
            __mb_assume_aligned(n_e);
            __mb_assume_aligned(n_h);

            if (tid > 0) {
                p_d = m_d[tid - 1];
                p_h = m_h[tid - 1];
                p_e = m_e[tid - 1];
            }

            if (tid < P - 1) {
                n_d = m_d[tid + 1];
                n_h = m_h[tid + 1];
                n_e = m_e[tid + 1];
            }

            /* main loop */
            for (unsigned int n = 0; n < num_timesteps/OL; n++) {
                /* exchange data */
                if (tid > 0) {
#pragma ivdep
                    for (unsigned int i = 0; i < OL; i++) {
                        t_d[i] = p_d[chunk_base + i];
                        t_e[i] = p_e[chunk_base + i];
                        t_h[i] = p_h[chunk_base + i];
                    }
                }

                if (tid < P - 1) {
#pragma ivdep
                    for (unsigned int i = 0; i < OL; i++) {
                        t_d[OL + chunk_base + i] = n_d[OL + i];
                        t_e[OL + chunk_base + i] = n_e[OL + i];
                        t_h[OL + chunk_base + i] = n_h[OL + i];
                    }
                }

                /* sync after communication */
#pragma omp barrier

                /* sub-loop */
                for (unsigned int m = 0; m < OL; m++) {
                    /* update e */
#pragma omp simd aligned(t_d, t_e, t_mat_indices : ALIGN)
                    for (int i = m; i < size - m - 1; i++) {
                        // for (int i = 0; i < chunk + 2 * OL - 1; i++) {
                        int mat_idx = t_mat_indices[i];

                        real j = l_sim_consts[mat_idx].sigma * t_e[i];

                        real p_t = l_sim_consts[mat_idx].M_CP
                            * l_sim_consts[mat_idx].d12 *
                            (l_sim_consts[mat_idx].w12 * t_d[i][1] -
                             l_sim_consts[mat_idx].gamma12 * t_d[i][0]);

                        t_e[i] += l_sim_consts[mat_idx].M_CE *
                            (-j - p_t + (t_h[i + 1] - t_h[i]) *
                             l_sim_consts[mat_idx].d_x_inv);
                    }

                    /* apply sources */
                    for (unsigned int k = 0; k < num_sources; k++) {
                        int at = l_sim_sources[k].x_idx[0] - tid * chunk_base
                            + OL;
                        if ((at > 0) && (at < chunk + 2 * OL)) {
                            if (l_sim_sources[k].type ==
                                source::type::hard_source) {
                                t_e[at] = m_source_data
                                    [l_sim_sources[k].data_base_idx
                                     + (n * OL + m)];
                            } else if (l_sim_sources[k].type ==
                                       source::type::soft_source) {
                                t_e[at] += m_source_data
                                    [l_sim_sources[k].data_base_idx
                                     + (n * OL + m)];
                            } else {
                            }
                        }
                    }

                    /* update d */
                    //#pragma omp simd aligned(t_e, t_d, t_mat_indices : ALIGN)
                    for (int i = m; i < size - m - 1; i++) {
                        int mat_idx = t_mat_indices[i];

                        Eigen::Matrix3d prop_U1 = m_sim_consts[mat_idx].L_1E;
                        prop_U1 = (prop_U1 * t_e[i]).exp();

                        Eigen::Vector3d temp = t_d[i];

                        /* first time-independent half-step */
                        temp = m_sim_consts[mat_idx].prop_U02 * temp
                            + m_sim_consts[mat_idx].equi_corr;

                        /* time-dependent step */
                        temp = prop_U1 * temp;

                        /* second time-independent half-step */
                        temp = m_sim_consts[mat_idx].prop_U02 * temp
                            + m_sim_consts[mat_idx].equi_corr;

                        t_d[i] = temp;
                    }

                    /* update h */
                    //
#pragma omp simd aligned(t_e, t_mat_indices : ALIGN)
                    for (int i = m + 1; i < size - m - 1; i++) {
                        //for (int i = 1; i < chunk + 2 * OL - 1; i++) {
                        int mat_idx = t_mat_indices[i];

                        t_h[i] += l_sim_consts[mat_idx].M_CH *
                            (t_e[i] - t_e[i - 1]);
                    }

                    /* apply boundary condition */
                    if (tid == 0) {
                        t_h[OL] = 0;
                    }
                    if (tid == P - 1) {
                        t_h[OL + chunk] = 0;
                    }

                    /* save results to scratchpad in parallel */
                    for (int k = 0; k < num_copy; k++) {
                        if (l_copy_list[k].hasto_record(n * OL + m)) {
                            unsigned int pos = l_copy_list[k].get_position();
                            unsigned int cols = l_copy_list[k].get_cols();
                            int base_idx = tid * chunk_base - OL;
                            record::type t = l_copy_list[k].get_type();
                            int off_r = l_copy_list[k].get_offset_scratch_real
                                (n * OL + m, base_idx - pos);

#pragma omp simd
                            for (int i = OL; i < chunk + OL; i++) {
                                int idx = base_idx + i;
                                if ((idx >= pos) && (idx < pos + cols)) {
                                    if (t == record::type::electric) {
                                        m_result_scratch[off_r + i] = t_e[i];
                                    } else if (t == record::type::inversion) {
                                        m_result_scratch[off_r + i] =
                                            t_d[i][2];
                                    } else {
                                        /* TODO handle trouble */
                                    }
                                }
                            }

                            /*
                             *(m_result_scratch +
                             l_copy_list[k].get_scratch_real
                             (n * OL + m, idx - pos)) =
                             *l_copy_list[k].get_real(i, tid);
                             /*        if (cle.is_complex()) {
                             *cle.get_scratch_imag(n * OL + m,
                             idx - pos) =
                             *cle.get_imag(i, tid);
                             }*/

                        }
                    }
                } /* end sub loop */

                /* sync after computation */
#pragma omp barrier
            } /* end main foor loop */

        } /* end openmp region */
#ifdef XEON_PHI_OFFLOAD
    } /* end offload region */
#endif

    /* bulk copy results into result classes */
    for (const auto& cle : m_copy_list) {
        real *dr = m_result_scratch + cle.get_offset_scratch_real(0, 0);
        std::copy(dr, dr + cle.get_size(), cle.get_result_real(0, 0));
        if (cle.is_complex()) {
            real *di = m_result_scratch + cle.get_offset_scratch_imag(0, 0);
            std::copy(di, di + cle.get_size(), cle.get_result_imag(0, 0));
        }
    }
}

}
