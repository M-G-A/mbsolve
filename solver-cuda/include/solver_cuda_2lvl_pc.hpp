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

#ifndef MBSOLVE_SOLVER_CUDA_2LVL_PC_H
#define MBSOLVE_SOLVER_CUDA_2LVL_PC_H

#include <cuda.h>
#include <internal/common_fdtd_2lvl.hpp>
#include <internal/copy_list_entry.hpp>
#include <solver_int.hpp>

namespace mbsolve {

class solver_cuda_2lvl_pc : public solver_int
{
public:
    solver_cuda_2lvl_pc(std::shared_ptr<const device> dev,
                        std::shared_ptr<scenario> scen);

    ~solver_cuda_2lvl_pc();

    const std::string& get_name() const;

    void run() const;

private:
    /* data buffers on GPU */
    real *m_h;
    real *m_e;
    real *m_d;

    /* material indices on GPU */
    unsigned int *m_mat_indices;

    /* data for sources on GPU */
    real *m_source_data;

    /* result scratchpad memory on GPU */
    unsigned int m_scratch_size;
    real *m_result_scratch;

    /* scenario data on CPU */
    std::vector<sim_constants_2lvl> m_sim_consts;
    std::vector<sim_source> m_sim_sources;
    std::vector<copy_list_entry> m_copy_list;
};

}

#endif
