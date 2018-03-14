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

#include <iostream>
#include <cmath>
#include <omp.h>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <unsupported/Eigen/MatrixFunctions>
#include <solver_openmp_2lvl_os.hpp>
#include <common_openmp.hpp>

namespace mbsolve {

static solver_factory<solver_openmp_2lvl_os> factory("openmp-2lvl-os");

/* TODO: necessary? */
#if 0
unsigned int num_grid_points;
unsigned int num_time_steps;
real time_step_size;

    num_grid_points = m_scenario.NumGridPoints;
    num_time_steps = m_scenario.NumTimeSteps;
    time_step_size = m_scenario.TimeStepSize;
#endif

solver_openmp_2lvl_os::solver_openmp_2lvl_os(std::shared_ptr<const device> dev,
                                             std::shared_ptr<scenario> scen) :
solver_int(dev, scen)
{
    /* TODO: scenario, device sanity check */
    /*
     * device.length > 0 (-> regions.size() > 0)
     * required materials found?
     * no gap in regions
     *
     */

    /* TODO: solver params
     * courant number
     */

    /* initialize Eigen library for multi-threaded application */
    Eigen::initParallel();

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

        /* copy other constants */
        m_sim_consts[i].M_CE = materials[i].M_CE;
        m_sim_consts[i].M_CH = materials[i].M_CH;
        m_sim_consts[i].M_CP = 0.5 * materials[i].M_CP;
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

    /* allocate data arrays */
    m_d = (Eigen::Vector3d *) mb_aligned_alloc(sizeof(Eigen::Vector3d) *
                                               scen->get_num_gridpoints(0));
    m_h = (real *) mb_aligned_alloc(sizeof(real) *
                                    (scen->get_num_gridpoints(0) + 1));
    m_e = (real *) mb_aligned_alloc(sizeof(real) * scen->get_num_gridpoints(0));
    m_mat_indices = (unsigned int *)
        mb_aligned_alloc(sizeof(unsigned int) * scen->get_num_gridpoints(0));

    /* set up indices array and initialize data arrays */
#pragma omp parallel for schedule(static)
    for (unsigned int i = 0; i < scen->get_num_gridpoints(0); i++) {
        /* determine index of material */
        int idx = -1;
        real x = i * scen->get_gridpoint_size(0);
        for (const auto& reg : dev->get_regions()) {
            if ((x >= reg->get_start()) && (x <= reg->get_end())) {
                idx = id_to_idx[reg->get_material()->get_id()];
                break;
            }
        }
        /* TODO: assert/bug if idx == -1 */
        if ((idx < 0) || (idx >= dev->get_used_materials().size())) {
            std::cout << "At index " << i << std::endl;
            throw std::invalid_argument("region not found");
        }
        m_mat_indices[i] = idx;

        /* TODO: evaluate flexible initialization in scenario */
        m_d[i][0] = 0.0;
        m_d[i][1] = 0.0;
        m_d[i][2] = m_sim_consts[idx].inversion_init;
        m_e[i] = 0.0;
        m_h[i] = 0.0;
        if (i == scen->get_num_gridpoints(0) - 1) {
            m_h[i + 1] = 0.0;
        }
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
}

solver_openmp_2lvl_os::~solver_openmp_2lvl_os()
{
    mb_aligned_free(m_h);
    mb_aligned_free(m_e);
    mb_aligned_free(m_d);
    mb_aligned_free(m_mat_indices);
    mb_aligned_free(m_result_scratch);
    delete[] m_source_data;
}

const std::string&
solver_openmp_2lvl_os::get_name() const
{
    return factory.get_name();
}

void
solver_openmp_2lvl_os::run() const
{
    /*
#pragma offload target(mic) in(gsc, num_grid_points, num_time_steps) \
  in(time_step_size) \
  inout(m_e:length(m_scenario.NumGridPoints)) \
  inout(m_h:length(m_scenario.NumGridPoints + 1)) \
  inout(m_dm11:length(m_scenario.NumGridPoints)) \
  inout(m_dm12r:length(m_scenario.NumGridPoints)) \
  inout(m_dm12i:length(m_scenario.NumGridPoints)) \
  inout(m_dm22:length(m_scenario.NumGridPoints)) \
  in(region_indices:length(m_scenario.NumGridPoints))
  {
    */
#pragma omp parallel
    {
          /* main loop */
          for (int n = 0; n < m_scenario->get_num_timesteps(); n++) {

              /* update e in parallel */
              //#pragma omp for simd schedule(static)
#pragma omp for schedule(static)
              for (int i = 0; i < m_scenario->get_num_gridpoints(0); i++) {
                  unsigned int mat_idx = m_mat_indices[i];

                  real j = m_sim_consts[mat_idx].sigma * m_e[i];

                  real p_t = m_sim_consts[mat_idx].M_CP
                      * m_sim_consts[mat_idx].d12 *
                      (m_sim_consts[mat_idx].w12 * m_d[i][1] -
                       m_sim_consts[mat_idx].gamma12 * m_d[i][0]);

                  m_e[i] = m_e[i] + m_sim_consts[mat_idx].M_CE *
                      (-j - p_t + (m_h[i + 1] - m_h[i])
                       * m_sim_consts[mat_idx].d_x_inv);
              }

              /* apply sources */
              for (const auto& src : m_sim_sources) {
                  /* TODO: support other source types than hard sources */
                  if (src.type == source::type::hard_source) {
                      m_e[src.x_idx[0]] = m_source_data[src.data_base_idx + n];
                  } else if (src.type == source::type::soft_source) {
                      m_e[src.x_idx[0]] += m_source_data[src.data_base_idx + n];
                  } else {
                  }
              }

              /* update h and dm in parallel */
              //#pragma omp for simd schedule(static)
#pragma omp for schedule(static)
              for (int i = 0; i < m_scenario->get_num_gridpoints(0); i++) {
                  unsigned int mat_idx = m_mat_indices[i];

                  if (i > 0) {
                      m_h[i] += m_sim_consts[mat_idx].M_CH *
                          (m_e[i] - m_e[i - 1]);
                  }

                  real arg = 2.0 * m_sim_consts[mat_idx].d_t *
                      m_sim_consts[mat_idx].d12 * m_e[i];
                  real s = sin(arg);
                  real c = cos(arg);

                  Eigen::Matrix3d prop_U1;
                  prop_U1(0, 0) = + 1.0;
                  prop_U1(1, 1) = c;
                  prop_U1(2, 2) = c;
                  prop_U1(1, 2) = s;
                  prop_U1(2, 1) = -s;

                  Eigen::Vector3d temp = m_d[i];

                  /* first time-independent half-step */
                  temp = m_sim_consts[mat_idx].prop_U02 * temp
                      + m_sim_consts[mat_idx].equi_corr;

                  /* time-dependent step */
                  temp = prop_U1 * temp;

                  /* second time-independent half-step */
                  temp = m_sim_consts[mat_idx].prop_U02 * temp
                      + m_sim_consts[mat_idx].equi_corr;

                  m_d[i] = temp;
              }

              /* apply boundary condition */
              m_h[0] = 0;
              m_h[m_scenario->get_num_gridpoints(0)] = 0;

              /* save results to scratchpad in parallel */
              for (const auto& cle : m_copy_list) {
                  if (cle.hasto_record(n)) {
                      unsigned int pos = cle.get_position();
                      unsigned int cols = cle.get_cols();
                      record::type t = cle.get_type();
                      unsigned int o_r = cle.get_offset_scratch_real(n, 0);

#pragma omp for schedule(static)
                      for (int i = pos; i < pos + cols; i++) {
                          if (t == record::type::electric) {
                              m_result_scratch[o_r + i - pos] = m_e[i];
                          } else if (t == record::type::inversion) {
                              m_result_scratch[o_r + i - pos] = m_d[i][2];
                          } else {
                              /* TODO handle trouble */
                          }
                      }
                      /* TODO handle imaginary */

                  }
              }
          }
    }

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
