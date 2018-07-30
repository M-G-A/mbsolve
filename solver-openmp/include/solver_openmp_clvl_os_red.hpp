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

#define ABSORBING_BOUNDARY 0
#define CURRENT_MODEL 1

#ifndef MBSOLVE_SOLVER_OPENMP_3LVL_OS_RED_H
#define MBSOLVE_SOLVER_OPENMP_3LVL_OS_RED_H

#include <iostream>
#include <omp.h>
#include <solver.hpp>
#include <internal/common_fdtd_2lvl.hpp>
#include <internal/copy_list_entry.hpp>

namespace mbsolve {

/* struct def shouldn't be here -> mbsolve-lib/types? */
struct sim_grid{
    unsigned int ***ind;
    unsigned int *num;
};

template<unsigned int num_lvl, unsigned int dim>
class sim_constants_clvl_os
{
    static const unsigned int num_adj = num_lvl * num_lvl - 1;

    typedef Eigen::Matrix<complex, num_adj, num_adj> complex_matrix_t;
    typedef Eigen::Matrix<real, num_adj, num_adj> real_matrix_t;
    typedef Eigen::Matrix<real, num_adj, 1> real_vector_t;

public:
    /* constant propagators */
    //    complex_matrix_t B_1;
    //    complex_matrix_t B_2;

    bool has_qm;
    bool has_dipole;

    /* analytic solution precalc */
    real_matrix_t coeff_1[dim][num_adj/2];
    real_matrix_t coeff_2[dim][num_adj/2];
    real theta[dim][num_adj/2];

    /* rodrigues formula precalc */
    real_matrix_t U2[dim];
    real theta_1[dim];

    /* constant propagator A_0 = exp(M dt/2) */
    real_matrix_t A_0;

    /* unitary transformation matrix */
    complex_matrix_t B[dim];

    /* required for polarization calc ? */
    real_matrix_t M;
    real_matrix_t U[dim];
    real_vector_t d_in;
    real_vector_t d_eq;

    /* dipole moments */
    real_vector_t v[dim];

    /* diagonalized interaction propagator */
    /* TODO: special type for diagonal matrix? */
    /* TODO: vector would do, right? */
    Eigen::Matrix<complex, num_adj, 1> L[dim];

    /* electromagnetic constants */
    real M_CE;                            // add direction-dependent constants
    real M_CH;
    real M_CP;
    real sigma;

    /* simulation constants */
    real d_r_inv[dim];
    real d_t;

    /* initialization constants */
    real_vector_t d_init;

};

template<unsigned int num_lvl, unsigned int dim>
class solver_openmp_clvl_os_red : public solver_int
{
    static const unsigned int num_adj = num_lvl * num_lvl - 1;

    typedef Eigen::Matrix<complex, num_lvl, num_lvl> qm_operator_t;
    typedef Eigen::Matrix<complex, num_adj, num_adj> complex_matrix_t;
    typedef Eigen::Matrix<real, num_adj, num_adj> real_matrix_t;
    typedef Eigen::Matrix<real, num_adj, 1> real_vector_t;
    
    
public:
    
    solver_openmp_clvl_os_red(std::shared_ptr<const device> dev,
                              std::shared_ptr<scenario> scen);

    ~solver_openmp_clvl_os_red();

    const std::string& get_name() const;

    void run() const;
    
    int test;

private:
    const std::string m_name;

    /* TODO: rule of three. make copy constructor etc. private?
     * or implement correctly
     */

    /*
     * Position-dependent density matrix in adjoint representation.
     */
    real_vector_t **m_d;

    std::vector<qm_operator_t > m_generators;

    Eigen::Matrix<real, dim, 1> **m_h;
    Eigen::Matrix<real, dim, 1> **m_e;
    Eigen::Matrix<real, dim, 1> **m_p;
 
#if ABSORBING_BOUNDARY==1
    real *m_e_0;
    real *m_e_L;
    real m_s[6];
#endif
    
#if CURRENT_MODEL!=0
    real **m_w;
#endif

    sim_grid grid;

    real *m_result_scratch;

    unsigned int m_scratch_size;

    real *m_source_data;

    unsigned int **m_mat_indices;

#ifdef XEON_PHI_OFFLOAD
    copy_list_entry_dev *l_copy_list;
#else
    copy_list_entry *l_copy_list;
#endif
    sim_source *l_sim_sources;
    sim_constants_clvl_os<num_lvl,dim> *l_sim_consts;

    std::vector<sim_constants_clvl_os<num_lvl,dim> > m_sim_consts;

    std::vector<sim_source> m_sim_sources;

    std::vector<copy_list_entry> m_copy_list;

    
    void
    setup_generators()
    {
        m_generators.clear();

        /* set up SU(N) generators u -- real part off-diagonal elements */
        for (int k = 0; k < num_lvl; k++) {
            for (int j = 0; j < k; j++) {
                qm_operator_t g;
                g(j, k) = 1;
                g(k, j) = 1;
                m_generators.push_back(g);
            }
        }

        /* set up SU(N) generators v -- imag part off-diagonal elements */
        for (int k = 0; k < num_lvl; k++) {
            for (int j = 0; j < k; j++) {
                qm_operator_t g;
                g(j, k) = complex(0, -1);
                g(k, j) = complex(0, +1);
                m_generators.push_back(g);
            }
        }

        /* set up SU(N) generators w -- main-diagonal elements */
        for (int l = 0; l < num_lvl - 1; l++) {
            qm_operator_t g;
            int j = 0;

            for (j = 0; j <= l; j++) {
                g(j, j) = 1.0;
            }
            g(j, j) = -(l + 1);

            g *= -sqrt(2.0/((l + 1) * (l + 2)));
            m_generators.push_back(g);
        }
    }

    real_vector_t
    get_adj_op(const qm_operator_t& op)
    {
        real_vector_t ret;
        for (int i = 0; i < num_adj; i++) {
            qm_operator_t m;
            m = op * m_generators[i];
            ret[i] = m.real().trace();
        }
        return ret;
    }

    real_matrix_t
    get_adj_sop(qm_operator_t (*G)(const qm_operator_t&))
    {
        real_matrix_t ret;

        for (int i = 0; i < num_adj; i++) {
            for (int j = 0; j < num_adj; j++) {
                qm_operator_t result = G(m_generators[j]) * m_generators[i];
                complex c = 0.5 * result.trace();
                ret(i, j) = c.real();
            }
        }
        return ret;
    }

    real_vector_t
    get_adj_deq(qm_operator_t (*G)(const qm_operator_t&))
    {
        real_vector_t ret;
        for (int i = 0; i < num_adj; i++) {
            qm_operator_t m;
            m = G(qm_operator_t::Identity()) * m_generators[i];
            ret[i] = m.real().trace() * 1.0/num_lvl;
        }
        return ret;
    }

    real_matrix_t
    get_adj_liouvillian(const qm_operator_t& H) {
        real_matrix_t ret;

        for (int i = 0; i < num_adj; i++) {
            for (int j = 0; j < num_adj; j++) {
                qm_operator_t result = H *
                    (m_generators[i] * m_generators[j] -
                     m_generators[j] * m_generators[i]);
                complex c = complex(0, 1) * 0.5 * result.trace();
                ret(i, j) = c.real() * 1.0/HBAR;
            }
        }
        return ret;
    }

};

typedef solver_openmp_clvl_os_red<3,1> solver_openmp_3lvl_os_red;

}

#endif
