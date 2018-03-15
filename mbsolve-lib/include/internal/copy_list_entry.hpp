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

#ifndef MBSOLVE_COPY_LIST_ENTRY_H
#define MBSOLVE_COPY_LIST_ENTRY_H

#include <memory>
//#include <mbsolve.hpp>
#include <record.hpp>
#include <result.hpp>
#include <scenario.hpp>
#include <types.hpp>

/* TODO: needs improvement */
#ifdef XEON_PHI_OFFLOAD
#define __mb_on_device __attribute__((target(mic)))
#else
#ifdef __CUDACC__
#define __mb_on_device __host__ __device__
#include <math.h>
#else
#define __mb_on_device
#endif
#endif

namespace mbsolve {

/**
 * This internal class is the device-side equivalent to \ref copy_list_entry.
 * It stores all data required to copy result data from device-side
 * raw arrays to device-side scratchpad memory.
 * \ingroup MBSOLVE_LIB
 */
class copy_list_entry_dev
{
    friend class copy_list_entry;
private:
    unsigned int m_offset_scratch_real;
    unsigned int m_offset_scratch_imag;

    unsigned int m_rows;
    unsigned int m_cols=1;
    real m_interval_ratio;
    int m_position_idx[3];

    real m_timestep;
    real m_interval;

    record::type m_type;
    unsigned int m_col_idx;
    unsigned int m_row_idx;

    bool m_is_complex;
    
    unsigned int m_grid_num[3] = {1,1,1};

    /*TODO make members private -> friend/nested with copy_list_entry? */

public:
    __mb_on_device bool hasto_record(unsigned int iteration) const {
        int last = floor(m_interval_ratio * (iteration - 1));
        int next = floor(m_interval_ratio * (iteration + 0));

        return (last != next);
    }
    
    __mb_on_device unsigned int has2rec_r(unsigned int x, unsigned int y, unsigned int z, unsigned int t) const {
        unsigned int ret = 1;
        unsigned int ratio[3] = {x,y,z};
//(x >= m_position_idx[0]+2) && (x < m_position_idx[0]+2 + m_cols) &&
        for (unsigned int dim_num=0; dim_num<3; dim_num++) {
            if (m_position_idx[dim_num]<0) {
                ret = (ratio[dim_num]%(-m_position_idx[dim_num])==0) ? ret : 0;
                ratio[dim_num] = -m_position_idx[dim_num];
            } else {
                ret = (ratio[dim_num] == m_position_idx[dim_num]) ? ret : 0;
                ratio[dim_num] = m_position_idx[dim_num];
            }
        }
        if (ret == 1) {
            ret=1+get_offset_scratch_real(t)+x/ratio[0]
                *(m_grid_num[1]*m_grid_num[2])+y/ratio[1]
                *m_grid_num[2]+z/ratio[2];
        }
        return ret;
    }

    __mb_on_device bool is_complex() const {
        return m_is_complex;
    }

    __mb_on_device record::type get_type() const {
        return m_type;
    }

    __mb_on_device unsigned int get_col_idx() const {
        return m_col_idx;
    }

    __mb_on_device unsigned int get_row_idx() const {
        return m_row_idx;
    }

    __mb_on_device unsigned int get_position() const {
        return m_position_idx[0];
    }

    __mb_on_device unsigned int get_cols() const {
        return m_cols;
    }

    __mb_on_device unsigned int
    get_offset_scratch_real(unsigned int iteration,
                            unsigned int gridpoint = 0) const {
        return m_offset_scratch_real +
            floor(m_interval_ratio * iteration) * m_cols + gridpoint;
    }

    __mb_on_device unsigned int
    get_offset_scratch_imag(unsigned int iteration,
                            unsigned int gridpoint = 0) const {
        return m_offset_scratch_imag +
            floor(m_interval_ratio * iteration) * m_cols + gridpoint;
    }
};

/**
 * This internal class stores all data required to copy result data from
 * raw arrays to the \ref result class.
 * \ingroup MBSOLVE_LIB
 */
class copy_list_entry
{
private:
    std::shared_ptr<result> m_result;
    std::shared_ptr<const record> m_record;

    /* m_dev members */

    copy_list_entry_dev m_dev;

public:
    copy_list_entry(std::shared_ptr<const record> rec,
                    std::shared_ptr<const scenario> scen,
                    unsigned int offset_scratch) :
        m_record(rec)
    {
        m_dev.m_timestep = scen->get_timestep_size();
        
        if (rec->get_interval() <= scen->get_timestep_size()) {
            m_dev.m_rows = scen->get_num_timesteps();
            m_dev.m_interval = scen->get_timestep_size();
            m_dev.m_interval_ratio = 1.0;
        } else {
            m_dev.m_rows = floor(scen->get_endtime()/rec->get_interval()) + 1;
            m_dev.m_interval = rec->get_interval();
            m_dev.m_interval_ratio = scen->get_timestep_size()/
                rec->get_interval();
        }

        for (unsigned int dim_num=0; dim_num<3; dim_num++) {
            if (rec->get_position(dim_num) < 0.0) {
                /* copy complete grid */
                m_dev.m_position_idx[dim_num] = rec->get_position(dim_num);
                m_dev.m_grid_num[dim_num] = (scen->get_num_gridpoints(dim_num)/((int) -rec->get_position(dim_num)));
                m_dev.m_cols *= m_dev.m_grid_num[dim_num];
            } else {
                m_dev.m_position_idx[dim_num] = std::round(rec->get_position(dim_num)/
                                                     scen->get_gridpoint_size(dim_num));
            }
        }

	/* create result */
        m_result = std::make_shared<result>(rec->get_name(), m_dev.m_cols,
                                            m_dev.m_rows, m_dev.m_grid_num);

        m_dev.m_type = rec->get_type();
        m_dev.m_timestep = scen->get_timestep_size();
        m_dev.m_is_complex = rec->is_complex();

        m_dev.m_offset_scratch_real = offset_scratch;
        m_dev.m_offset_scratch_imag = offset_scratch + m_dev.m_cols *
            m_dev.m_rows;

        m_dev.m_col_idx = rec->get_col();
        m_dev.m_row_idx = rec->get_row();
    }

    const copy_list_entry_dev& get_dev() const { return m_dev; }

    record::type get_type() const { return m_dev.get_type(); }

    unsigned int get_col_idx() const { return m_dev.get_col_idx(); }

    unsigned int get_row_idx() const { return m_dev.get_row_idx(); }

    bool hasto_record(unsigned int iteration) const {
        return m_dev.hasto_record(iteration);
    }
    
    unsigned int has2rec_r(unsigned int x, unsigned int y, unsigned int z, unsigned int t) const {
        return m_dev.has2rec_r(x,y,z,t);
    }

    bool is_complex() const {
        return m_dev.m_is_complex;
    }

    unsigned int get_interval() const { return m_dev.m_interval; }

    unsigned int get_position() const { return m_dev.m_position_idx[0]; }

    unsigned int get_cols() const { return m_dev.m_cols; }

    unsigned int get_rows() const { return m_dev.m_rows; }

    unsigned int get_size() const { return m_dev.m_cols * m_dev.m_rows; }

    std::shared_ptr<const record> get_record() const { return m_record; };

    std::shared_ptr<result> get_result() const { return m_result; }

    std::vector<real>::iterator
    get_result_real(unsigned int iteration, unsigned int gridpoint = 0) const {
        return m_result->get_data_real(floor(m_dev.m_interval_ratio *
                                             iteration),
                                       gridpoint);
    }

    std::vector<real>::iterator
    get_result_imag(unsigned int iteration, unsigned int gridpoint = 0) const {
        return m_result->get_data_imag(floor(m_dev.m_interval_ratio *
                                             iteration),
                                       gridpoint);
    }

    unsigned int
    get_offset_scratch_imag(unsigned int iteration,
                            unsigned int gridpoint = 0) const {
        return m_dev.get_offset_scratch_imag(iteration, gridpoint);
    }

    unsigned int
    get_offset_scratch_real(unsigned int iteration,
                            unsigned int gridpoint = 0) const {
        return m_dev.get_offset_scratch_real(iteration, gridpoint);
    }
};

}

#endif
