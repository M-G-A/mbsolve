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

#ifndef SCENARIO_H
#define SCENARIO_H

#include <string>
#include <vector>
#include <Record.hpp>
#include <Source.hpp>

namespace mbsolve {

class Scenario
{
public:
    std::string Name;

    unsigned int NumTimeSteps;

    unsigned int NumGridPoints;

    real TimeStepSize;

    real GridPointSize;

    real SimEndTime;

    std::vector<Record> Records;

    /* TODO: add sources vector */
    //std::vector<ISource *> Sources;

    /* TODO: change real to Quantity */

};

}

#endif
