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

#include <numeric>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <unsupported/Eigen/MatrixFunctions>
#include <common_openmp.hpp>
#include <solver_openmp_clvl_os_red.hpp>

namespace mbsolve {

static solver_factory<solver_openmp_clvl_os_red<2,1> > f2("openmp-2lvl-os-red");
static solver_factory<solver_openmp_clvl_os_red<2,2> > f2_2("openmp-2lvl-os-red-2d");
static solver_factory<solver_openmp_clvl_os_red<2,3> > f2_3("openmp-2lvl-os-red-3d");
static solver_factory<solver_openmp_clvl_os_red<3,1> > f3("openmp-3lvl-os-red");
static solver_factory<solver_openmp_clvl_os_red<3,2> > f3_2("openmp-3lvl-os-red-2d");
static solver_factory<solver_openmp_clvl_os_red<3,3> > f3_3("openmp-3lvl-os-red-3d");
    
/* redundant calculation overlap */
#ifdef XEON_PHI_OFFLOAD
__mb_on_device const unsigned int OL = 32;
#else
const unsigned int OL = 32;
#endif

const unsigned int VEC = 4;

template<unsigned int num_lvl, unsigned int num_adj, unsigned int dim>
void
fill_rodr_coeff(const Eigen::Matrix<complex, num_adj, num_adj>& eigenvec,
                const Eigen::Matrix<complex, num_adj, 1>& eigenval,
                sim_constants_clvl_os<num_lvl,dim>& sc, unsigned int dim_num)
{
    /* creating sorting order (descending eigenvalues) */
    std::vector<size_t> perm_idx(num_adj);
    std::iota(perm_idx.begin(), perm_idx.end(), 0);
    std::sort(perm_idx.begin(), perm_idx.end(),
              [&eigenval](size_t i1, size_t i2) {
                  return (std::abs(eigenval(i1)) > std::abs(eigenval(i2)));
              });

    /* sort eigenvectors */
    Eigen::Matrix<real, num_adj, num_adj> Q =
        Eigen::Matrix<real, num_adj, num_adj>::Zero();
    for (int i = 0; i < num_adj/2; i++) {
        unsigned int i1 = perm_idx[2 * i];
        unsigned int i2 = perm_idx[2 * i + 1];

        Q.col(2 * i) = 1.0/sqrt(2) *
            (eigenvec.col(i1) + eigenvec.col(i2)).real();
        Q.col(2 * i + 1) = 1.0/sqrt(2) *
            (-eigenvec.col(i1) + eigenvec.col(i2)).imag();
    }
    if (num_adj % 2 != 0) {
        Q(num_adj - 1, num_adj - 1) = 1.0;
    }

    /* TODO optimize
     * ignore eigenvalues = 0
     * group eigenvalues with multiplicity >= 2
     */
    for (int i = 0; i < num_adj/2; i++) {
        unsigned int i1 = perm_idx[2 * i];
        unsigned int i2 = perm_idx[2 * i + 1];
        Eigen::Matrix<real, num_adj, num_adj> b =
            Eigen::Matrix<real, num_adj, num_adj>::Zero();

        /* give warning if eigenvalues do not match */
        if (std::abs(eigenval(i1)) + std::abs(eigenval(i2)) > 1e-5) {
            std::cout << "Warning: Eigenvalues not pairwise: " <<
                eigenval(i1) << " and " << eigenval(i2) << std::endl;
        }
        sc.theta[dim_num][i] = std::abs(eigenval(i1));

        b(2 * i, 2 * i + 1) = -1.0;
        b(2 * i + 1, 2 * i) = +1.0;

        sc.coeff_1[dim_num][i] = Q * b * Q.transpose();
        sc.coeff_2[dim_num][i] = Q * b * b * Q.transpose();

        std::cout << "theta: "<< std::endl << sc.theta[dim_num][i] << std::endl;
        std::cout << "b = " << std::endl << b << std::endl;
    }
}


template<unsigned int num_lvl, unsigned int dim>
solver_openmp_clvl_os_red<num_lvl, dim>::solver_openmp_clvl_os_red
(std::shared_ptr<const device> dev, std::shared_ptr<scenario> scen) :
    solver_int(dev, scen),
    m_name("openmp-" + std::to_string(num_lvl) + "lvl-os-red")
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
    
    setup_generators();

    /* set up simulation-grid */
    grid.num = new unsigned int [3];
    for (unsigned int dim_num=0; dim_num<3; dim_num++){
        grid.num[dim_num] = scen->get_num_gridpoints(dim_num);
    }
    grid.ind = new unsigned int **[grid.num[0]];
    for(unsigned int x = 0; x < grid.num[0]; x++) {
        grid.ind[x] = new unsigned int *[grid.num[1]];
        for(unsigned int y = 0; y < grid.num[1]; y++) {
            grid.ind[x][y] = new unsigned int[grid.num[2]];
            for(unsigned int z = 0; z < grid.num[2]; z++) {
                grid.ind[x][y][z] = x*(grid.num[1]*grid.num[2])+y*grid.num[2]+z;
//                std::cout << grid.ind[x][y][z] << " " ;
            }
        }
//        std::cout <<  std::endl;
    }
    /* set up simulaton constants */
    std::map<std::string, unsigned int> id_to_idx;
    unsigned int j = 0;

    for (const auto& mat_id : dev->get_used_materials()) {
        sim_constants_clvl_os<num_lvl,dim> sc;

        auto mat = material::get_from_library(mat_id);

        /* factor for electric field update */
        sc.M_CE = scen->get_timestep_size()/
            (EPS0 * mat->get_rel_permittivity());

        /* factor for magnetic field update */
        sc.M_CH = scen->get_timestep_size()/
            (MU0 * mat->get_rel_permeability());

        /* convert loss term to conductivity */
        sc.sigma = sqrt(EPS0 * mat->get_rel_permittivity()/
                        (MU0 * mat->get_rel_permeability()))
            * mat->get_losses() * 2.0;

        /* 3-lvl quantum mechanical system */
        /* active region in 3-lvl description? */
        /* TODO: remove ugly dynamic cast to qm_desc_3lvl, allow other
         * representations? */
        std::shared_ptr<qm_desc_clvl<num_lvl> > qm =
            std::dynamic_pointer_cast<qm_desc_clvl<num_lvl> >(mat->get_qm());
        if (qm) {
            /* factor for macroscopic polarization */
            sc.M_CP = 0.5 * mat->get_overlap_factor() *
                qm->get_carrier_density();
            
            /* determine dipole operator as vector */
            for (unsigned int dim_num=0; dim_num<dim; dim_num++){
                sc.v[dim_num] = get_adj_op(qm->get_dipole_op(dim_num));
                std::cout << "v" << dim_num << ": " << std::endl << sc.v[dim_num] << std::endl;
            }
            
            /* time-independent hamiltionian in adjoint representation */
            Eigen::Matrix<real, num_adj, num_adj> M_0;
            M_0 = get_adj_liouvillian(qm->get_hamiltonian());

            /* determine lindblad term in adjoint representation */
            Eigen::Matrix<real, num_adj, num_adj> G;
            G = get_adj_sop(qm->get_lindblad_op());

            /* time-independent part */
            Eigen::Matrix<real, num_adj, num_adj> M = M_0 + G;
            std::cout << "M: " << std::endl << M << std::endl;

            /* determine equilibrium term */
            Eigen::Matrix<real, num_adj, 1> d_eq;
            d_eq = get_adj_deq(qm->get_lindblad_op());
            sc.d_eq = d_eq;

            /* determine inhomogeneous term */
            real eps = std::numeric_limits<real>::epsilon();
            if (d_eq.isZero(eps)) {
                sc.d_in = Eigen::Matrix<real, num_adj, 1>::Zero();
            } else {
                /* solve equation system M * d_in = d_eq */
                sc.d_in = M.fullPivLu().solve(d_eq);

                real err = (M * sc.d_in - d_eq).norm() / d_eq.norm();
                std::cout << "d_in solver error: " << err << std::endl;
                /* TODO: throw exception or only report warning? */
                if (err > 1e-3) {
                    throw std::invalid_argument("Time-indepent matrix not "
                                                "invertible!");
                }
            }
            std::cout << "d_in: " << std::endl << sc.d_in << std::endl;

            /* determine constant propagator */
            Eigen::Matrix<real, num_adj, num_adj> A_0;
            A_0 = (M * scen->get_timestep_size()/2).exp();

            /* determine dipole operator in adjoint representation */
            for (unsigned int dim_num=0; dim_num<dim; dim_num++){
                Eigen::Matrix<real, num_adj, num_adj> U;
                U = get_adj_liouvillian(-qm->get_dipole_op(dim_num));
                std::cout << "U" << dim_num << ": " << std::endl << U << std::endl;

                /* diagonalize dipole operator */
                /* note: returned eigenvectors are normalized */
                Eigen::EigenSolver<Eigen::Matrix<real, num_adj, num_adj> > es(U);

                /* store propagators B1 and B2 */
                //sc.B_1 = A_0 * es.eigenvectors();
                //sc.B_2 = es.eigenvectors().adjoint() * A_0;

                sc.B[dim_num] = es.eigenvectors();
                sc.U[dim_num] = U;
                
                /* for Rodrigues formula */
                sc.U2[dim_num] = U * U;
                sc.theta_1[dim_num] = sqrt(pow(U(0, 1), 2) + pow(U(0, 2),2)
                                     + pow(U(1, 2), 2));
                
                /* for general analytic approach */
                fill_rodr_coeff<num_lvl, num_adj>(es.eigenvectors(),
                                                  es.eigenvalues(), sc, dim_num);
                
                /* store diagonal matrix containing the eigenvalues */
                sc.L[dim_num] = es.eigenvalues() * scen->get_timestep_size();
            }
                
            sc.A_0 = A_0;
            sc.M = M;


            /* TODO refine check? */
            sc.has_qm = true;
            sc.has_dipole = true;

            sc.d_init = get_adj_op(qm->get_d_init());

            std::cout << "init: " << sc.d_init << std::endl;

            /* TODO remove?
            if (scen->get_dm_init_type() == scenario::lower_full) {

            } else if (scen->get_dm_init_type() == scenario::upper_full) {
                sc.d_init = Eigen::Matrix<real, num_adj, 1>::Zero();
                sc.d_init(num_adj - 1) = 1;
            } else {
            }
            */
        } else {
            /* set all qm-related factors to zero */
            sc.M_CP = 0.0;

            sc.has_qm = false;
            sc.has_dipole = false;

            for (unsigned int dim_num=0; dim_num<dim; dim_num++){
                sc.v[dim_num] = Eigen::Matrix<real, num_adj, 1>::Zero();
                sc.B[dim_num] = Eigen::Matrix<complex, num_adj, num_adj>::Zero();
                sc.U[dim_num] = Eigen::Matrix<real, num_adj, num_adj>::Zero();
                sc.L[dim_num] = Eigen::Matrix<complex, num_adj, 1>::Zero();
            }
                
            //sc.B_1 = Eigen::Matrix<complex, num_adj, num_adj>::Zero();
            //sc.B_2 = Eigen::Matrix<complex, num_adj, num_adj>::Zero();
            sc.A_0 = Eigen::Matrix<real, num_adj, num_adj>::Zero();
            sc.M = Eigen::Matrix<real, num_adj, num_adj>::Zero();

            sc.d_in = Eigen::Matrix<real, num_adj, 1>::Zero();
            sc.d_eq = Eigen::Matrix<real, num_adj, 1>::Zero();
            sc.d_init = Eigen::Matrix<real, num_adj, 1>::Zero();
        }

        /* simulation settings */
        for (unsigned int dim_num=0; dim_num<dim; dim_num++){
            sc.d_r_inv[dim_num] = 1.0/scen->get_gridpoint_size(dim_num);
        }
        sc.d_t = scen->get_timestep_size();

        m_sim_consts.push_back(sc);
        id_to_idx[mat->get_id()] = j;
        j++;
    }

    /* set up indices array and initialize data arrays */
    unsigned int P = omp_get_max_threads();

    std::cout << "Number of threads: " << P << std::endl;
    m_d = new Eigen::Matrix<real, num_adj, 1>*[P];
    m_e = new Eigen::Matrix<real, dim, 1>*[P];
    m_h = new Eigen::Matrix<real, dim, 1>*[P];
    m_p = new Eigen::Matrix<real, dim, 1>*[P];
    m_mat_indices = new unsigned int*[P];

    unsigned int *l_mat_indices = new unsigned int[grid.num[0]];

    for (unsigned int i = 0; i < grid.num[0]; i++) {
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

        std::cout << "Rows: " << entry.get_rows() << " Cols: " << entry.get_cols() << std::endl;

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
        s.x_idx = src->get_position()/scen->get_gridpoint_size(0);
        s.data_base_idx = base_idx;
        m_sim_sources.push_back(s);

        /* calculate source values */
        for (unsigned int j = 0; j < scen->get_num_timesteps(); j++) {
            m_source_data[base_idx + j] =
                src->get_value(j * scen->get_timestep_size());
        }

        base_idx += scen->get_num_timesteps();
    }

//    unsigned int num_gridpoints = m_scenario->get_num_gridpoints(0);
    unsigned int chunk_base = grid.num[0]/P;
    unsigned int chunk_rem = grid.num[0] % P;
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
    l_sim_consts = new sim_constants_3lvl_os[m_sim_consts.size()];
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
    in(l_mat_indices:length(grid.num[0]))                            \
    in(l_copy_list:length(num_copy) __mb_phi_create)                    \
    in(m_source_data:length(num_timesteps * num_sources) __mb_phi_create) \
    in(l_sim_sources:length(num_sources) __mb_phi_create)               \
    in(l_sim_consts:length(m_sim_consts.size()) __mb_phi_create)        \
    inout(m_e,m_p,m_h,m_d:length(P) __mb_phi_create)                    \
    inout(m_mat_indices:length(P) __mb_phi_create)
    {
#endif
        for (unsigned int tid = 0; tid < P; tid++) {
            unsigned int chunk = chunk_base;

            if (tid == P - 1) {
                chunk += chunk_rem;
            }

            /* allocation */
            unsigned int size = (chunk + 2 * OL) * grid.num[1] * grid.num[2];

            m_d[tid] = (Eigen::Matrix<real, num_adj, 1> *)
                mb_aligned_alloc(size *
                                sizeof(Eigen::Matrix<real, num_adj, 1>));
            m_h[tid] = (Eigen::Matrix<real, dim, 1> *) mb_aligned_alloc(size *
                                sizeof(Eigen::Matrix<real, dim, 1>));
            m_e[tid] = (Eigen::Matrix<real, dim, 1> *) mb_aligned_alloc(size *
                                sizeof(Eigen::Matrix<real, dim, 1>));
            m_p[tid] = (Eigen::Matrix<real, dim, 1> *) mb_aligned_alloc(size *
                                sizeof(Eigen::Matrix<real, dim, 1>));
            m_mat_indices[tid] = (unsigned int *)
                mb_aligned_alloc((chunk + 2 * OL) * sizeof(unsigned int));
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

            Eigen::Matrix<real, num_adj, 1> *t_d;
            Eigen::Matrix<real, dim, 1> *t_h, *t_e, *t_p;
            unsigned int *t_mat_indices;

            t_d = m_d[tid];
            t_h = m_h[tid];
            t_e = m_e[tid];
            t_p = m_p[tid];
            t_mat_indices = m_mat_indices[tid];

            __mb_assume_aligned(t_d);
            __mb_assume_aligned(t_e);
            __mb_assume_aligned(t_p);
            __mb_assume_aligned(t_h);
            __mb_assume_aligned(t_mat_indices);

            for (int i = 0; i < size; i++) {
                unsigned int global_idx = tid * chunk_base + (i - OL);
                if ((global_idx >= 0) && (global_idx < grid.num[0])) {
                    unsigned int mat_idx = l_mat_indices[global_idx];
                    t_mat_indices[i] = mat_idx;
                    for (unsigned int y=0; y<grid.num[1]; y++) {
                        for (unsigned int z=0; z<grid.num[2]; z++) {
                            t_d[grid.ind[i][y][z]] = l_sim_consts[mat_idx].d_init;
                        }
                    }
                } else {
                    t_mat_indices[i] = 0;
                    for (unsigned int y=0; y<grid.num[1]; y++) {
                        for (unsigned int z=0; z<grid.num[2]; z++) {
                            t_d[grid.ind[i][y][z]] = Eigen::Matrix<real, num_adj, 1>::Zero();
                        }
                    }
                }
                for (unsigned int y=0; y<grid.num[1]; y++) {
                    for (unsigned int z=0; z<grid.num[2]; z++) {
                        t_e[grid.ind[i][y][z]] = Eigen::Matrix<real, dim, 1>::Zero();
                        t_p[grid.ind[i][y][z]] = Eigen::Matrix<real, dim, 1>::Zero();
                        t_h[grid.ind[i][y][z]] = Eigen::Matrix<real, dim, 1>::Zero();
                    }
                }
            }
#pragma omp barrier
        }
#ifdef XEON_PHI_OFFLOAD
    }
#endif

    delete[] l_mat_indices;
}

template<unsigned int num_lvl, unsigned int dim>
solver_openmp_clvl_os_red<num_lvl, dim>::~solver_openmp_clvl_os_red()
{
    unsigned int P = omp_get_max_threads();
    unsigned int num_sources = m_sim_sources.size();
    unsigned int num_copy = m_copy_list.size();
//    unsigned int num_gridpoints = m_scenario->get_num_gridpoints(0);
    unsigned int num_timesteps = m_scenario->get_num_timesteps();

#ifdef XEON_PHI_OFFLOAD
#pragma offload target(mic:0) in(P)                                     \
    in(num_sources, num_copy)                                           \
    in(l_copy_list:length(num_copy) __mb_phi_delete)                    \
    in(m_source_data:length(num_timesteps * num_sources) __mb_phi_delete) \
    in(l_sim_sources:length(num_sources) __mb_phi_delete)               \
    in(l_sim_consts:length(m_sim_consts.size()) __mb_phi_delete)        \
    in(m_e,m_h,m_p,m_d,m_mat_indices:length(P) __mb_phi_delete)
    {
#endif
#pragma omp parallel
        {
            unsigned int tid = omp_get_thread_num();

            mb_aligned_free(m_h[tid]);
            mb_aligned_free(m_e[tid]);
            mb_aligned_free(m_p[tid]);
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
    delete[] m_p;
    delete[] m_d;
    delete[] m_mat_indices;
    for(unsigned int x = 0; x < grid.num[0]; x++) {
        for(unsigned int y = 0; y < grid.num[1]; y++) {
            delete[] grid.ind[x][y];
        }
        delete[] grid.ind[x];
    }
    delete [] grid.ind;
}

template<unsigned int num_lvl, unsigned int dim>
const std::string&
solver_openmp_clvl_os_red<num_lvl, dim>::get_name() const
{
    return m_name;
}

template<unsigned int num_lvl, unsigned int num_adj, unsigned int dim>
void
update_fdtd(unsigned int size, unsigned int border,
            Eigen::Matrix<real, dim, 1> *t_e, Eigen::Matrix<real, dim, 1> *t_p,
            Eigen::Matrix<real, dim, 1> *t_h, Eigen::Matrix<real, num_adj, 1> *t_d,
            unsigned int *t_mat_indices,
            sim_constants_clvl_os<num_lvl,dim> *l_sim_consts, sim_grid grid)
{
#pragma omp simd aligned(t_d, t_e, t_p, t_h, t_mat_indices : ALIGN)
    for (int i = border; i < size - border - 1; i++) {
        int mat_idx = t_mat_indices[i];
        for (unsigned int y=1; y<grid.num[1]-1; y++) {
            for (unsigned int z=1; z<grid.num[2]-1; z++) {
                Eigen::Matrix<real, dim, 1> j = l_sim_consts[mat_idx].sigma
                                                * t_e[grid.ind[i][y][z]];
                switch (dim) {
                    case 3:
                        t_e[grid.ind[i][y][z]][0] += l_sim_consts[mat_idx].M_CE
                            * (-j[0] - t_p[grid.ind[i][y][z]][0]
                            + (t_h[grid.ind[i][y+1][z]][2] - t_h[grid.ind[i][y][z]][2])
                            * l_sim_consts[mat_idx].d_r_inv[1]      //dyHz
                            - (t_h[grid.ind[i][y][z+1]][1] - t_h[grid.ind[i][y][z]][1])
                            * l_sim_consts[mat_idx].d_r_inv[2]);    //dzHy
                        t_e[grid.ind[i][y][z]][1] += l_sim_consts[mat_idx].M_CE
                            * (-j[1] - t_p[grid.ind[i][y][z]][1]
                            + (t_h[grid.ind[i][y][z+1]][0] - t_h[grid.ind[i][y][z]][0])
                            * l_sim_consts[mat_idx].d_r_inv[2]      //dzHx
                            - (t_h[grid.ind[i+1][y][z]][2] - t_h[grid.ind[i][y][z]][2])
                            * l_sim_consts[mat_idx].d_r_inv[0]);    //dxHz
                        t_e[grid.ind[i][y][z]][2] += l_sim_consts[mat_idx].M_CE
                            * (-j[2] - t_p[grid.ind[i][y][z]][2]
                            + (t_h[grid.ind[i+1][y][z]][1] - t_h[grid.ind[i][y][z]][1])
                            * l_sim_consts[mat_idx].d_r_inv[0]      //dxHy
                            - (t_h[grid.ind[i][y+1][z]][0] - t_h[grid.ind[i][y][z]][0])
                            * l_sim_consts[mat_idx].d_r_inv[1]);    //dyHx
                        break;
                        
                    case 2:
                        /* One has to decide between coupled and none coupled fields */
                        /*t_e[grid.ind[i][y][z]][0] += l_sim_consts[mat_idx].M_CE
                            * (-j[0] - t_p[grid.ind[i][y][z]][0]
                            + (t_h[grid.ind[i+1][y][z]][1] - t_h[grid.ind[i][y][z]][1])
                            * l_sim_consts[mat_idx].d_r_inv[0]);    //dyHz
                        t_e[grid.ind[i][y][z]][1] += l_sim_consts[mat_idx].M_CE
                            * (-j[1] - t_p[grid.ind[i][y][z]][1]
                            + (t_h[grid.ind[i][y+1][z]][0] - t_h[grid.ind[i][y][z]][0])
                            * l_sim_consts[mat_idx].d_r_inv[1]);    //dxHy  */
                        t_e[grid.ind[i][y][z]][0] += 0;
                        t_e[grid.ind[i][y][z]][1] += l_sim_consts[mat_idx].M_CE
                            * (-j[1] - t_p[grid.ind[i][y][z]][1]
                            + (t_h[grid.ind[i+1][y][z]][1] - t_h[grid.ind[i][y][z]][1])
                            * l_sim_consts[mat_idx].d_r_inv[0]      //dxHy
                            - (t_h[grid.ind[i][y+1][z]][0] - t_h[grid.ind[i][y][z]][0])
                            * l_sim_consts[mat_idx].d_r_inv[1]);    //dyHx
                        break;
                        
                    default:
                        t_e[grid.ind[i][y][z]][0] += l_sim_consts[mat_idx].M_CE
                            * (-j[0] - t_p[grid.ind[i][y][z]][0]
                            + (t_h[grid.ind[i+1][y][z]][0] - t_h[grid.ind[i][y][z]][0])
                            * l_sim_consts[mat_idx].d_r_inv[0]);    //dxHy
                        break;
                }
            }
        }
        /*
        if (i >= border + 1) {
            t_h[i] += l_sim_consts[mat_idx].M_CH * (t_e[i] - t_e[i - 1]);
        }
        */
    }
}

template<unsigned int num_lvl, unsigned int num_adj, unsigned int dim>
void
update_h(unsigned int size, unsigned int border, Eigen::Matrix<real, dim, 1> *t_e,
         Eigen::Matrix<real, dim, 1> *t_p,
         Eigen::Matrix<real, dim, 1> *t_h, Eigen::Matrix<real, num_adj, 1> *t_d,
         unsigned int *t_mat_indices,
         sim_constants_clvl_os<num_lvl,dim> *l_sim_consts, sim_grid grid)
{
#pragma omp simd aligned(t_d, t_e, t_p, t_h, t_mat_indices : ALIGN)
    for (int i = border; i < size - border - 1; i++) {
        int mat_idx = t_mat_indices[i];

        if (i >= border + 1) {
            for (unsigned int y=1; y<grid.num[1]-1; y++) {
                for (unsigned int z=1; z<grid.num[2]-1; z++) {
                    switch (dim) {
                        case 3:
                            t_h[grid.ind[i][y][z]][0] -= l_sim_consts[mat_idx].M_CH
                                * ((t_e[grid.ind[i][y][z]][2] - t_e[grid.ind[i][y-1][z]][2])
                                * l_sim_consts[mat_idx].d_r_inv[1]      //dyEz
                                - (t_e[grid.ind[i][y][z]][1] - t_e[grid.ind[i][y][z-1]][1])
                                * l_sim_consts[mat_idx].d_r_inv[2]);    //dzEy
                            t_h[grid.ind[i][y][z]][1] -= l_sim_consts[mat_idx].M_CH
                                * ((t_e[grid.ind[i][y][z]][0] - t_e[grid.ind[i][y][z-1]][0])
                                * l_sim_consts[mat_idx].d_r_inv[2]      //dzEx
                                - (t_e[grid.ind[i][y][z]][2] - t_e[grid.ind[i-1][y][z]][2])
                                * l_sim_consts[mat_idx].d_r_inv[0]);    //dxEz
                            t_h[grid.ind[i][y][z]][1] -= l_sim_consts[mat_idx].M_CH
                                * ((t_e[grid.ind[i][y][z]][1] - t_e[grid.ind[i-1][y][z]][1])
                                * l_sim_consts[mat_idx].d_r_inv[0]      //dxEy
                                - (t_e[grid.ind[i][y][z]][0] - t_e[grid.ind[i][y-1][z]][0])
                                * l_sim_consts[mat_idx].d_r_inv[1]);    //dyEx
                            break;
                        case 2:
                            /* One has to decide between coupled and none coupled fields */
                            /*t_h[grid.ind[i][y][z]][0] -= l_sim_consts[mat_idx].M_CH
                                * ((t_e[grid.ind[i][y][z]][1] - t_e[grid.ind[i][y-1][z]][1])
                                * l_sim_consts[mat_idx].d_r_inv[1]);    //dyEz
                            t_h[grid.ind[i][y][z]][1] += l_sim_consts[mat_idx].M_CH
                                * ((t_e[grid.ind[i][y][z]][0] - t_e[grid.ind[i-1][y][z]][0])
                                * l_sim_consts[mat_idx].d_r_inv[0]);    //dxEy */
                            t_h[grid.ind[i][y][z]][0] -= l_sim_consts[mat_idx].M_CH
                                * ((t_e[grid.ind[i][y][z]][1] - t_e[grid.ind[i][y-1][z]][1])
                                * l_sim_consts[mat_idx].d_r_inv[1]);    //dyEz
                            t_h[grid.ind[i][y][z]][1] += l_sim_consts[mat_idx].M_CH
                                * ((t_e[grid.ind[i][y][z]][1] - t_e[grid.ind[i-1][y][z]][1])
                                * l_sim_consts[mat_idx].d_r_inv[0]);    //dxEz
                            break;
                            
                        default:
                            t_h[grid.ind[i][y][z]][0] += l_sim_consts[mat_idx].M_CH
                                * ((t_e[grid.ind[i][y][z]][0] - t_e[grid.ind[i-1][y][z]][0])
                                * l_sim_consts[mat_idx].d_r_inv[0]);    //dxEz
                            break;
                    }
                }
            }
        }
    }
}

template<unsigned int dim>
void
apply_sources(Eigen::Matrix<real, dim, 1> *t_e, real *source_data,
              unsigned int num_sources, sim_source *l_sim_sources, unsigned int time,
              unsigned int base_pos, unsigned int chunk, sim_grid grid) //ToDo template dim
{
    for (unsigned int k = 0; k < num_sources; k++) {
        int at = l_sim_sources[k].x_idx - base_pos + OL;
        if ((at > 0) && (at < chunk + 2 * OL)) {
            real src = source_data[l_sim_sources[k].data_base_idx + time];
            if (l_sim_sources[k].type == source::type::hard_source) {
                for (unsigned int y=0; y<grid.num[1]; y++) {
                    for (unsigned int z=0; z<grid.num[2]; z++) {
                        t_e[grid.ind[at][y][z]][dim-1] = src;
                    }
                }
            } else if (l_sim_sources[k].type == source::type::soft_source) {
                /* TODO: fix source */
                for (unsigned int y=0; y<grid.num[1]; y++) {
                    for (unsigned int z=0; z<grid.num[2]; z++) {
                        t_e[grid.ind[at][y][z]][dim-1] += src;
                    }
                }
            } else {
            }
        }
    }
}

complex mexp(const complex& arg)
{
    return std::exp(arg);
}

template<unsigned int num_lvl, unsigned int num_adj, unsigned int dim>
inline Eigen::Matrix<real, num_adj, num_adj>
mat_exp(const sim_constants_clvl_os<num_lvl,dim>& s, Eigen::Matrix<real, dim, 1> e)
{
    Eigen::Matrix<real, num_adj, num_adj> ret;
//    ret = Eigen::Matrix<real, num_adj, num_adj>::Identity();

#if EXP_METHOD==1
    /* by diagonalization */
    Eigen::Matrix<complex, num_adj, 1> diag_exp = s.L[dim-1] * e[dim-1];
    diag_exp = diag_exp.unaryExpr(&mexp);

    ret = (s.B[dim-1] * diag_exp.asDiagonal() * s.B[dim-1].adjoint()).real();
#elif EXP_METHOD==2
    /* analytic solution */
    if (num_lvl == 2) {
        /* Rodrigues formula */
        ret = sin(s.theta_1[dim-1] * e[dim-1] * s.d_t)/s.theta_1[dim-1] * s.U[dim-1]
            + (1 - cos(s.theta_1[dim-1] * e[dim-1] * s.d_t))/(s.theta_1[dim-1] * s.theta_1[dim-1])
            * s.U2[dim-1] + Eigen::Matrix<real, num_adj, num_adj>::Identity();
    } else {
        ret = Eigen::Matrix<real, num_adj, num_adj>::Identity();
        for (int i = 0; i < num_adj/2; i++) {
            /* TODO nolias()? */
            ret += sin(s.theta[dim-1][i] * e[dim-1] * s.d_t) * s.coeff_1[dim-1][i]
                + (1 - cos(s.theta[dim-1][i] * e[dim-1] * s.d_t)) * s.coeff_2[dim-1][i];
        }
    }
#else
    /* Eigen matrix exponential */
    ret = (s.U[dim-1] * e[dim-1] * s.d_t).exp();
#endif

    return ret;
}

template<unsigned int num_lvl, unsigned int num_adj, unsigned int dim>
void
update_d(unsigned int size, unsigned int border, Eigen::Matrix<real, dim, 1> *t_e,
         Eigen::Matrix<real, dim, 1> *t_p, Eigen::Matrix<real, num_adj, 1> *t_d,
         unsigned int *t_mat_indices,
         sim_constants_clvl_os<num_lvl,dim> *l_sim_consts, sim_grid grid)
{
    //#pragma omp simd aligned(t_d, t_e, t_mat_indices : ALIGN)
    for (int i = border; i < size - border - 1; i++) {
        int mat_idx = t_mat_indices[i];

        for (unsigned int y=1; y<grid.num[1]-1; y++) {
            for (unsigned int z=1; z<grid.num[2]-1; z++) {
                if (l_sim_consts[mat_idx].has_qm) {
                    /* update density matrix */
                    Eigen::Matrix<real, num_adj, 1> d1, d2;

                    /* time-indepedent half step */
                    d1 = l_sim_consts[mat_idx].A_0
                        * (t_d[grid.ind[i][y][z]] + l_sim_consts[mat_idx].d_in)
                        - l_sim_consts[mat_idx].d_in;

                    /* time-dependent full step */
                    if (l_sim_consts[mat_idx].has_dipole) {
                        /* determine time-dependent propagator */
                        Eigen::Matrix<real, num_adj, num_adj> A_I =
                            mat_exp<num_lvl, num_adj, dim>(l_sim_consts[mat_idx], t_e[grid.ind[i][y][z]]);
                        d2 = A_I * d1;
                    } else {
                        d2 = d1;
                    }

                    /* time-indepedent half step */
                    t_d[grid.ind[i][y][z]] = l_sim_consts[mat_idx].A_0
                        * (d2 + l_sim_consts[mat_idx].d_in)
                        - l_sim_consts[mat_idx].d_in;

                    /* update polarization */
                    for (unsigned int dim_num=0; dim_num<dim; dim_num++){
                        t_p[grid.ind[i][y][z]][dim_num] = l_sim_consts[mat_idx].M_CP
                            * l_sim_consts[mat_idx].v[dim_num].transpose()
                            * (l_sim_consts[mat_idx].M * t_d[grid.ind[i][y][z]]
                            + l_sim_consts[mat_idx].d_eq);
                    }
                } else {
                    t_p[grid.ind[i][y][z]] = Eigen::Matrix<real, dim, 1>::Zero();
                }
            }
        }
    }
}

template<unsigned int num_lvl, unsigned int dim>
void
solver_openmp_clvl_os_red<num_lvl, dim>::run() const
{
    unsigned int P = omp_get_max_threads();
//    unsigned int num_gridpoints = m_scenario->get_num_gridpoints(0);
    unsigned int chunk_base = grid.num[0]/P;
    unsigned int chunk_rem = grid.num[0] % P;
    unsigned int num_timesteps = m_scenario->get_num_timesteps();
    unsigned int num_sources = m_sim_sources.size();
    unsigned int num_copy = m_copy_list.size();
    real dt = m_scenario->get_timestep_size();

#ifdef XEON_PHI_OFFLOAD
#pragma offload target(mic:0) in(P)                                     \
    in(chunk_base, chunk_rem, grid.num[0], num_timesteps)            \
    in(num_sources, num_copy)                                           \
    in(l_copy_list:length(num_copy) __mb_phi_use)                       \
    in(m_source_data:length(num_timesteps * num_sources) __mb_phi_use)  \
    in(l_sim_sources:length(num_sources) __mb_phi_use)                  \
    in(l_sim_consts:length(m_sim_consts.size()) __mb_phi_use)           \
    in(m_e,m_p,m_h,m_d,m_mat_indices:length(P) __mb_phi_use)            \
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
            Eigen::Matrix<real, num_adj, 1> *t_d;
            Eigen::Matrix<real, dim, 1> *t_h, *t_e, *t_p;
            unsigned int *t_mat_indices;

            t_d = m_d[tid];
            t_h = m_h[tid];
            t_e = m_e[tid];
            t_p = m_p[tid];
            t_mat_indices = m_mat_indices[tid];

            __mb_assume_aligned(t_d);
            __mb_assume_aligned(t_e);
            __mb_assume_aligned(t_p);
            __mb_assume_aligned(t_h);
            __mb_assume_aligned(t_mat_indices);

            __mb_assume_aligned(m_result_scratch);

            /* gather prev and next pointers from other threads */
            Eigen::Matrix<real, num_adj, 1> *n_d, *p_d;
            Eigen::Matrix<real, dim, 1> *n_h, *n_e;
            Eigen::Matrix<real, dim, 1> *p_h, *p_e;

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
            for (unsigned int n = 0; n <= num_timesteps/OL; n++) {
                /* display progress - only4development */
                if (tid == 0){
                    float perc=n*10000/(num_timesteps/OL);
                    printf("%04.1f%%:\t%9.3es\r", perc/100, n*OL*dt);
                    fflush(stdout);
                }
                
                /* handle loop remainder */
                unsigned int subloop_ct = (n == num_timesteps/OL) ?
                    num_timesteps % OL : OL;

                /* exchange data */
                if (tid > 0) {
#pragma ivdep
                    for (unsigned int i = 0; i < OL; i++) {
                        for (unsigned int y=0; y<grid.num[1]; y++) {
                            for (unsigned int z=0; z<grid.num[2]; z++) {
                                t_d[grid.ind[i][y][z]] = p_d[grid.ind[chunk_base + i][y][z]];
                                t_e[grid.ind[i][y][z]] = p_e[grid.ind[chunk_base + i][y][z]];
                                t_h[grid.ind[i][y][z]] = p_h[grid.ind[chunk_base + i][y][z]];
                            }
                        }
                    }
                }

                if (tid < P - 1) {
#pragma ivdep
                    for (unsigned int i = 0; i < OL; i++) {
                        for (unsigned int y=0; y<grid.num[1]; y++) {
                            for (unsigned int z=0; z<grid.num[2]; z++) {
                                t_d[grid.ind[OL + chunk_base + i][y][z]] = n_d[grid.ind[OL + i][y][z]];
                                t_e[grid.ind[OL + chunk_base + i][y][z]] = n_e[grid.ind[OL + i][y][z]];
                                t_h[grid.ind[OL + chunk_base + i][y][z]] = n_h[grid.ind[OL + i][y][z]];
                            }
                        }
                    }
                }

                /* sync after communication */
#pragma omp barrier

                /* sub-loop */
                for (unsigned int m = 0; m < subloop_ct; m++) {
                    /* align border to vector length */
                    unsigned int border = m - (m % VEC);

                    /* update d */
                    update_d<num_lvl, num_adj, dim>(size, border, t_e, t_p, t_d,
                                               t_mat_indices, l_sim_consts, grid);

                     /* update e + h with fdtd */
                    update_fdtd<num_lvl, num_adj, dim>(size, border, t_e, t_p, t_h,
                                                  t_d, t_mat_indices,
                                                  l_sim_consts, grid);

                    /* apply sources */
                    apply_sources<dim>(t_e, m_source_data, num_sources,
                                  l_sim_sources, n * OL + m, tid * chunk_base,
                                  chunk, grid);

                    update_h<num_lvl, num_adj, dim>(size, border, t_e, t_p, t_h,
                                              t_d, t_mat_indices,
                                              l_sim_consts, grid);


                    /* apply field boundary condition */
                    if (tid == 0) {
                        for (unsigned int y=0; y<grid.num[1]; y++) {
                            for (unsigned int z=0; z<grid.num[2]; z++) {
                                t_h[grid.ind[OL][y][z]] = Eigen::Matrix<real, dim, 1>::Zero();
                            }
                        }
                    }
                    if (tid == P - 1) {
                        for (unsigned int y=0; y<grid.num[1]; y++) {
                            for (unsigned int z=0; z<grid.num[2]; z++) {
                                t_h[grid.ind[OL + chunk][y][z]] = Eigen::Matrix<real, dim, 1>::Zero();
                            }
                        }
                    }

                     /* save results to scratchpad in parallel */
                    for (int k = 0; k < num_copy; k++) {
                        if (l_copy_list[k].hasto_record(n * OL + m)) {
                            unsigned int pos = l_copy_list[k].get_position();
                            unsigned int cols = l_copy_list[k].get_cols();
                            int base_idx = tid * chunk_base - OL;
                            record::type t = l_copy_list[k].get_type();
                            unsigned int ridx = l_copy_list[k].get_row_idx();
                            unsigned int cidx = l_copy_list[k].get_col_idx();
                            int off_r = l_copy_list[k].get_offset_scratch_real
                            (n * OL + m, base_idx*(grid.num[1])*(grid.num[2]) - pos);

#pragma omp simd
                            for (int i = OL; i < chunk + OL; i++) {
                                int idx = base_idx + i;
                                if ((idx >= pos) && (idx < pos + cols)) {
                                    if (t == record::type::electric) {
                                        for (unsigned int y=0; y<grid.num[1]; y++) {
                                            for (unsigned int z=0; z<grid.num[2]; z++) {
                                                m_result_scratch[off_r + grid.ind[i][y][z]] = t_e[grid.ind[i][y][z]][ridx];
                                            }
                                        }
                                    } else if (t == record::type::magnetic) {
                                        for (unsigned int y=0; y<grid.num[1]; y++) {
                                            for (unsigned int z=0; z<grid.num[2]; z++) {
                                                m_result_scratch[off_r + grid.ind[i][y][z]] = t_h[grid.ind[i][y][z]][ridx];
                                            }
                                        }
                                    } else if (t == record::type::inversion) {
                                        for (unsigned int y=0; y<grid.num[1]; y++) {
                                            for (unsigned int z=0; z<grid.num[2]; z++) {
                                                m_result_scratch[off_r + grid.ind[i][y][z]] =
                                                    t_d[grid.ind[i][y][z]](num_lvl * (num_lvl - 1));
                                            }
                                        }
                                    } else if (t == record::type::density) {

                                        /* right now only populations */
                                        real temp = 1.0/num_lvl;
                                        for (int l = num_lvl * (num_lvl - 1);
                                             l < num_adj; l++) {
                                            temp += 0.5 * t_d[grid.ind[i][grid.num[1]/2][grid.num[2]/2]](l) *
                                                m_generators[l](ridx, cidx).real();
                                        }
                                        m_result_scratch[off_r + i] = temp;

                                        /* TODO: coherences
                                         * remove 1/3
                                         * consider only two/one corresponding
                                         * entry */

                                    } else {
                                        /* TODO handle trouble */

                                        /* TODO calculate regular populations */
                                    }
                                }
                            }

                            /*
                             *(m_result_scratch +
                             l_copy_list[k].get_scratch_real
                             (n * OL + m, idx - pos)) =
                             *l_copy_list[k].get_real(i, tid);
                             if (cle.is_complex()) {
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
