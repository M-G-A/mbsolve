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

#include <device.hpp>

namespace mbsolve {

device::device(const std::string& name, real width) : m_name(name),m_width(width)
{
}

device::device(const std::string& file, const std::vector<material *>& mats)
{

}

device::~device()
{
}

void
device::add_region(std::shared_ptr<region> reg) {
    m_regions.push_back(reg);
    m_used_materials.insert(reg->get_material()->get_id());
}

const std::vector<std::shared_ptr<region> >&
device:: get_regions() const
{
    return m_regions;
}

const std::set<std::string>&
device::get_used_materials() const
{
    return m_used_materials;
}

real
device::get_length(unsigned int dim_num) const {
    real total = 0.0;
    for (auto r : m_regions) {
        total += r->get_length();
    }
    /* device width and height - ToDo add extra-variable*/
    if ((dim_num==1)||(dim_num==2)){
        total=m_width;
    }
    return total;
}

const std::string&
device::get_name() const
{
    return m_name;
}

real
device::get_minimum_permittivity() const
{
    real min = 1e42;
    for (auto r : m_regions) {
        real eps_r = r->get_material()->get_rel_permittivity();
        if (eps_r < min) {
            min = eps_r;
        }
    }
    return min;
}

}
