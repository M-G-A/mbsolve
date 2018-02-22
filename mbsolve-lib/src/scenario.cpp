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

#include <scenario.hpp>

namespace mbsolve {

scenario::scenario(unsigned int dim, const std::string& name,
                   unsigned int *num_gridpoints, real endtime) :
    m_dim(dim), m_name(name), m_num_gridpoints(num_gridpoints),
    m_endtime(endtime), m_dm_init_type(dm_init_type::lower_full)
{
    m_gridpoint_size = new real [dim];
}

void
scenario::add_record(std::shared_ptr<record> rec)
{
    m_records.push_back(rec);
}

const std::vector<std::shared_ptr<record> >&
scenario::get_records() const
{
    return m_records;
}

void
scenario::add_source(std::shared_ptr<source> src)
{
    m_sources.push_back(src);
}

const std::vector<std::shared_ptr<source> >&
scenario::get_sources() const
{
    return m_sources;
}

const std::string&
scenario::get_name() const
{
    return m_name;
}

unsigned int
scenario::get_num_timesteps() const
{
    return m_num_timesteps;
}

void
scenario::set_num_timesteps(unsigned int value)
{
    m_num_timesteps = value;
}

unsigned int
scenario::get_num_gridpoints(unsigned int dim_num) const
{
    if (dim_num>m_dim-1) {
        return 3;
    }
    return m_num_gridpoints[dim_num];
}

void
scenario::set_num_gridpoints(unsigned int value, unsigned int dim_num)
{
    m_num_gridpoints[dim_num] = value;
}

real
scenario::get_timestep_size() const
{
    return m_timestep_size;
}

void
scenario::set_timestep_size(real value)
{
    m_timestep_size = value;
}

real
scenario::get_gridpoint_size(unsigned int dim_num) const
{
    return m_gridpoint_size[dim_num];
}

void
scenario::set_gridpoint_size(real value, int dim_num)
{
    m_gridpoint_size[dim_num] = value;
}

real
scenario::get_endtime() const
{
    return m_endtime;
}

void
scenario::set_endtime(real value)
{
    m_endtime = value;
}

scenario::dm_init_type
scenario::get_dm_init_type() const
{
    return m_dm_init_type;
}

void
scenario::set_dm_init_type(scenario::dm_init_type type)
{
    m_dm_init_type = type;
}

}
