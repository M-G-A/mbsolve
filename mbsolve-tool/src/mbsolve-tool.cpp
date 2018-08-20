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

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>
#include <mbsolve.hpp>

namespace po = boost::program_options;
namespace ti = boost::timer;

static std::string device_file;
static std::string output_file;
static std::string scenario_file;
static std::string solver_method;
static std::string writer_method;
static mbsolve::real sim_endtime;
static unsigned int dim;
static unsigned int num_gridpoints[3];

static void parse_args(int argc, char **argv)
{
    po::options_description desc("Allowed options");
    desc.add_options()
	("help,h", "Print usage")
	("device,d", po::value<std::string>(&device_file)->required(),
	 "Set device settings file")
	("output,o", po::value<std::string>(&output_file), "Set output file")
	("scenario,s", po::value<std::string>(&scenario_file),
	 "Set scenario settings file")
	("method,m", po::value<std::string>(&solver_method)->required(),
	 "Set solver method")
	("writer,w", po::value<std::string>(&writer_method)->required(),
	 "Set writer")
        ("endtime,e", po::value<mbsolve::real>(&sim_endtime),
         "Set simulation end time")
        ("x-gridpoints,x", po::value<unsigned int>(&num_gridpoints[0]),
         "Set number of spatial grid points in x direction")
        ("y-gridpoints,y", po::value<unsigned int>(&num_gridpoints[1]),
         "Set number of spatial grid points in y direction")
        ("z-gridpoints,z", po::value<unsigned int>(&num_gridpoints[2]),
         "Set number of spatial grid points in z direction")
        ("Dimension,D", po::value<unsigned int>(&dim),
         "Set dimension");

    po::variables_map vm;
    try {
	po::store(po::parse_command_line(argc, argv, desc), vm);
	if (vm.count("help")) {
	    std::cout << desc;
	    exit(0);
	}
	po::notify(vm);
    } catch (po::error& e) {
	std::cerr << "Error: " << e.what() << std::endl;
	std::cerr << desc;
	exit(1);
    }

    if (vm.count("device")) {
        device_file = vm["device"].as<std::string>();
        std::cout << "Using device file " << device_file << std::endl;
    }
    if (vm.count("scenario")) {
        scenario_file = vm["scenario"].as<std::string>();
        std::cout << "Using scenario file " << scenario_file << std::endl;
    }
    
    std::cout << "simulating in "<< dim << "D" << std::endl;
}

/* song2005 relaxation superoperator */
Eigen::Matrix<mbsolve::complex, 3, 3>
relax_sop_song3(const Eigen::Matrix<mbsolve::complex, 3, 3>& arg)
{
    Eigen::Matrix<mbsolve::complex, 3, 3> ret =
        Eigen::Matrix<mbsolve::complex, 3, 3>::Zero();

    mbsolve::real d7_eq = 0.0;
    mbsolve::real d8_eq = 0.0;
    mbsolve::real T_1 = 1e-10;

    ret(0, 0) = 1.0/3 * (1/T_1 * (arg(2, 2) + arg(1, 1) - 2.0 * arg(0, 0)
                                  - d7_eq - d8_eq));
    ret(1, 1) = 1.0/3 * (1/T_1 * (arg(2, 2) + arg(1, 1) - 2.0 * arg(0, 0)
                                  - d7_eq - d8_eq))
        - 1/T_1 * (arg(1, 1) - arg(0, 0) - d7_eq);
    ret(2, 2) = 1.0/3 * (1/T_1 * (arg(2, 2) + arg(1, 1) - 2.0 * arg(0, 0)
                                  - d7_eq - d8_eq))
        - 1/T_1 * (arg(2, 2) - arg(0, 0) - d8_eq);
    ret(1, 0) = -1/T_1 * arg(1, 0);
    ret(0, 1) = -1/T_1 * arg(0, 1);
    ret(2, 0) = -1/T_1 * arg(2, 0);
    ret(0, 2) = -1/T_1 * arg(0, 2);
    ret(2, 1) = -1/T_1 * arg(2, 1);
    ret(1, 2) = -1/T_1 * arg(1, 2);

    return ret;
}

/* ziolkowski1995 relaxation superoperator */
Eigen::Matrix<mbsolve::complex, 2, 2>
relax_sop_ziolk2(const Eigen::Matrix<mbsolve::complex, 2, 2>& arg)
{
    Eigen::Matrix<mbsolve::complex, 2, 2> ret =
        Eigen::Matrix<mbsolve::complex, 2, 2>::Zero();

    ret(0, 0) = +1e10 * arg(1, 1);
    ret(1, 1) = -1e10 * arg(1, 1);
    ret(0, 1) = -1e10 * arg(0, 1);
    ret(1, 0) = -1e10 * arg(1, 0);

    return ret;
}

/* tzenov2018 absorber relaxation superoperator */
Eigen::Matrix<mbsolve::complex, 2, 2>
relax_sop_tzenov2018_abs(const Eigen::Matrix<mbsolve::complex, 2, 2>& arg)
{
    Eigen::Matrix<mbsolve::complex, 2, 2> ret =
        Eigen::Matrix<mbsolve::complex, 2, 2>::Zero();

    mbsolve::real scattering_rate = 1.0/3e-12;
    mbsolve::real dephasing_rate = 1.0/160e-15;

    /* at equilibrium, the lower level is fully populated */
    ret(0, 0) = +scattering_rate * arg(1, 1);
    ret(1, 1) = -scattering_rate * arg(1, 1);

    /* dephasing of coherence terms */
    ret(0, 1) = -dephasing_rate * arg(0, 1);
    ret(1, 0) = -dephasing_rate * arg(1, 0);

    return ret;
}

/* tzenov2018 gain relaxation superoperator */
Eigen::Matrix<mbsolve::complex, 2, 2>
relax_sop_tzenov2018_gain(const Eigen::Matrix<mbsolve::complex, 2, 2>& arg)
{
    Eigen::Matrix<mbsolve::complex, 2, 2> ret =
        Eigen::Matrix<mbsolve::complex, 2, 2>::Zero();

    mbsolve::real scattering_rate = 1.0/10e-12;
    mbsolve::real dephasing_rate = 1.0/200e-15;

    /* at equilibrium, the upper level is fully populated */
    ret(0, 0) = -scattering_rate * arg(0, 0);
    ret(1, 1) = +scattering_rate * arg(0, 0);

    /* dephasing of coherence terms */
    ret(0, 1) = -dephasing_rate * arg(0, 1);
    ret(1, 0) = -dephasing_rate * arg(1, 0);

    return ret;
}

int main(int argc, char **argv)
{
    dim=1;
    for (unsigned int dim_num=0; dim_num<3; dim_num++) {
        num_gridpoints[dim_num]=1;
    }
    
    /* parse command line arguments */
    parse_args(argc, argv);

    try {
	ti::cpu_timer timer;
	double total_time = 0;

        std::shared_ptr<mbsolve::device> dev;
        std::shared_ptr<mbsolve::scenario> scen;
        
        double dt=25e-15;
        double position[3]={-2,-1,-1};
        
        /* source */
        double pol[3]={0.0,0.0,0.0};
        double source_pos[3]={0.0,0.5,0.0}; //percentage for 2nd and 3rd dim
        double sigma=0.22;
        double A = 1;//use 1e-5 for gaussian-beam;
        double f = 1;
        int s = 0; //0=line,1=gaussian,2=gaussian-beam - source
        int n = 0; //gaussian-beam mode
        int m = 0; //gaussian-beam mode
        double width = 50e-6;

        if (device_file == "song2005") {
            /* Song setup */

            Eigen::Matrix<mbsolve::complex, 3, 3> H, *u, d_init;

            H <<0, 0, 0,
                0, 2.3717, 0,
                0, 0, 2.4165;
            H = H * mbsolve::HBAR * 1e15;

            // mbsolve::real g = 1.0;
            mbsolve::real g = sqrt(2);

            u = new Eigen::Matrix<mbsolve::complex, 3, 3> [dim];
            for (unsigned int dim_num=0; dim_num<dim-1; dim_num++) {
                u[dim_num]=Eigen::Matrix<mbsolve::complex, 3, 3>::Zero();
            }
            u[dim-1] <<0, 1.0, g,
                1.0, 0, 0,
                g, 0, 0;
            u[dim-1] = u[dim-1] * mbsolve::E0 * 9.2374e-11;

            d_init << 1, 0, 0,
                0, 0, 0,
                0, 0, 0;

            auto qm = std::make_shared<mbsolve::qm_desc_3lvl>
                (dim, 6e24, H, u, &relax_sop_song3, d_init);

            auto mat_vac = std::make_shared<mbsolve::material>("Vacuum");
            mbsolve::material::add_to_library(mat_vac);
            auto mat_ar = std::make_shared<mbsolve::material>("AR_Song", qm);
            mbsolve::material::add_to_library(mat_ar);

            dev = std::make_shared<mbsolve::device>("Song",width);
            dev->add_region(std::make_shared<mbsolve::region>
                            ("Active region", mat_ar, 0, 150e-6));

           /* default settings */
            if (num_gridpoints[0] == 1) {
                num_gridpoints[0] = 32768;
            }
            if (sim_endtime < 1e-21) {
                sim_endtime = 80e-15;
            }

            /* Song basic scenario */
            scen = std::make_shared<mbsolve::scenario>
                    (dim, "Basic", num_gridpoints, sim_endtime);
            if (s == 0) {
                auto source = std::make_shared<mbsolve::sech_pulse>
                ("sech", source_pos, mbsolve::source::hard_source, pol,
                 A*3.5471e9, f*3.8118e14, 17.248, 1.76/5e-15, -M_PI/2);
                scen->add_source(source);
            } else if (s == 1) {
                auto source = std::make_shared<mbsolve::gauss>
                //("sech", 0.0, mbsolve::source::hard_source, 4.2186e9/2, 2e14,
                ("sech", source_pos, mbsolve::source::hard_source, pol,
                 A*3.5471e9, f*3.8118e14, 17.248, sigma, 1.76/5e-15, -M_PI/2);
                scen->add_source(source);
            } else {
                auto source = std::make_shared<mbsolve::gauss_beam>
                //("sech", 0.0, mbsolve::source::hard_source, 4.2186e9/2, 2e14,
                ("gauss_beam", source_pos, mbsolve::source::hard_source, pol,
                 A*3.5471e9, f*3.8118e14, 17.248, sigma, n, m);
                scen->add_source(source);
            }

            scen->add_record(std::make_shared<mbsolve::record>
                             ("d11", mbsolve::record::type::density, 1, 1,
                              dt, position));
            scen->add_record(std::make_shared<mbsolve::record>
                             ("d22", mbsolve::record::type::density, 2, 2,
                              dt, position));
            scen->add_record(std::make_shared<mbsolve::record>
                             ("d33", mbsolve::record::type::density, 3, 3,
                              dt, position));

        } else if (device_file == "ziolkowski1995") {
            /* set up quantum mechanical description */
            std::shared_ptr<mbsolve::qm_description> qm;

            if ((solver_method == "openmp-2lvl-os") ||
                (solver_method == "openmp-2lvl-os-old") ||
                (solver_method == "openmp-2lvl-pc") ||
                (solver_method == "openmp-2lvl-pc-red")) {
                /* Ziolkowski setup in old 2-lvl desc */
                /* TODO: transform to new description */
                qm = std::make_shared<mbsolve::qm_desc_2lvl>
                    (1e24, 2 * M_PI * 2e14, 6.24e-11, 0.5e10, 1.0e10);

            } else {
                /* Ziolkowski setup in new 2-lvl desc */

                Eigen::Matrix<mbsolve::complex, 2, 2> H, *u, d_init;

                H <<-0.5, 0,
                    0, 0.5;
                H = H * mbsolve::HBAR * 2 * M_PI * 2e14;
                
                u = new Eigen::Matrix<mbsolve::complex, 2, 2> [dim];
                for (unsigned int dim_num=0; dim_num<dim-1; dim_num++) {
                    u[dim_num]=Eigen::Matrix<mbsolve::complex, 2, 2>::Zero();
                }
                u[dim-1] <<0, 1.0,
                    1.0, 0;
                u[dim-1] = u[dim-1] * mbsolve::E0 * 6.24e-11 * (-1.0);
                
                d_init << 1, 0,
                    0, 0;
                qm = std::make_shared<mbsolve::qm_desc_clvl<2> >
                    (dim, 1e24, H, u, &relax_sop_ziolk2, d_init);
            }

            /* materials */
            auto mat_vac = std::make_shared<mbsolve::material>("Vacuum");
            auto mat_ar = std::make_shared<mbsolve::material>
                ("AR_Ziolkowski", qm);
            mbsolve::material::add_to_library(mat_vac);
            mbsolve::material::add_to_library(mat_ar);

            /* set up device */
            dev = std::make_shared<mbsolve::device>("Ziolkowski",width);
            dev->add_region(std::make_shared<mbsolve::region>
                            ("Vacuum left", mat_vac, 0, 7.5e-6));
            dev->add_region(std::make_shared<mbsolve::region>
                            ("Active region", mat_ar, 7.5e-6, 142.5e-6));
            dev->add_region(std::make_shared<mbsolve::region>
                            ("Vacuum right", mat_vac, 142.5e-6, 150e-6));

            /* default settings */
            if (num_gridpoints[0] == 1) {
                num_gridpoints[0] = 500;
            }
            if (sim_endtime < 1e-21) {
                sim_endtime = 200e-15;
            }

            /* Ziolkowski basic scenario */
            scen = std::make_shared<mbsolve::scenario>
                (dim, "Basic", num_gridpoints, sim_endtime);

            if (s == 0) {
                auto source = std::make_shared<mbsolve::sech_pulse>
                //("sech", 0.0, mbsolve::source::hard_source, 4.2186e9/2, 2e14,
                ("sech", source_pos, mbsolve::source::hard_source, pol,
                 A*4.2186e9, f*2e14, 10, 2e14);
                scen->add_source(source);
            } else if (s == 1) {
                auto source = std::make_shared<mbsolve::gauss>
                ("gauss", source_pos, mbsolve::source::hard_source, pol,
                 A*4.2186e9, f*2e14, 10, sigma, 2e14);
                scen->add_source(source);
            }else{
                auto source = std::make_shared<mbsolve::gauss_beam>
                ("gauss_beam", source_pos, mbsolve::source::hard_source, pol,
                 A*4.2186e9, f*2e14, 10, sigma, n, m);
                scen->add_source(source);
            }

            scen->add_record(std::make_shared<mbsolve::record>
                             ("inv12", dt, position));

        } else if (device_file == "tzenov2018-cpml") {
            /* set up quantum mechanical descriptions */
            std::shared_ptr<mbsolve::qm_description> qm_gain;
            std::shared_ptr<mbsolve::qm_description> qm_absorber;

            if (solver_method == "openmp-2lvl-os-red") {
                /* 2-lvl description */
                Eigen::Matrix<mbsolve::complex, 2, 2> H;
                Eigen::Matrix<mbsolve::complex, 2, 2> *u_gain, *u_abs;
                Eigen::Matrix<mbsolve::complex, 2, 2> d_init;

                /* Hamiltonian */
                H <<-0.5, 0,
                    0, 0.5;
                H = H * mbsolve::HBAR * 2 * M_PI * 3.4e12;

                /* dipole moment operator */
                u_gain = new Eigen::Matrix<mbsolve::complex, 2, 2> [dim];
                u_abs = new Eigen::Matrix<mbsolve::complex, 2, 2> [dim];
                for (unsigned int dim_num=0; dim_num<dim-1; dim_num++) {
                    u_gain[dim_num]=Eigen::Matrix<mbsolve::complex, 2, 2>::Zero();
                    u_abs[dim_num]=Eigen::Matrix<mbsolve::complex, 2, 2>::Zero();
                }
                
                u_gain[dim-1] << 0, 1.0, 1.0, 0;
                u_gain[dim-1] = u_gain[dim-1] * mbsolve::E0 * 2e-9*1e-2;
                u_abs[dim-1] << 0, 1.0, 1.0, 0;
                u_abs[dim-1] = u_abs[dim-1] * mbsolve::E0 * 6e-9;

                /* initial value density matrix */
                d_init << 0.5, 0.001,
                    0.001, 0.5;

                qm_gain = std::make_shared<mbsolve::qm_desc_clvl<2> >
                    (dim, 5e21, H, u_gain, &relax_sop_tzenov2018_gain, d_init);
                qm_absorber = std::make_shared<mbsolve::qm_desc_clvl<2> >
                    (dim, 1e21, H, u_abs, &relax_sop_tzenov2018_abs, d_init);

            } else if (solver_method == "openmp-3lvl-os-red") {


            } else {
                throw std::invalid_argument("Solver not suitable!");
            }

            /* materials */
            auto mat_absorber = std::make_shared<mbsolve::material>
                ("Absorber", qm_absorber, 12.96, 1.0, 500);
            auto mat_gain = std::make_shared<mbsolve::material>
                ("Gain", qm_gain, 12.96, 1.0, 500);
            mbsolve::material::add_to_library(mat_absorber);
            mbsolve::material::add_to_library(mat_gain);

            /* set up device */
            dev = std::make_shared<mbsolve::device>("tzenov-cpml",width);
            dev->add_region(std::make_shared<mbsolve::region>
                            ("Gain R", mat_gain, 0, 0.5e-3));
            dev->add_region(std::make_shared<mbsolve::region>
                            ("Absorber", mat_absorber, 0.5e-3, 0.625e-3));
            dev->add_region(std::make_shared<mbsolve::region>
                            ("Gain L", mat_gain, 0.625e-3, 1.125e-3));

            /* default settings */
            if (num_gridpoints[0] == 0) {
                num_gridpoints[0] = 8912;
            }
            if (sim_endtime < 1e-21) {
                sim_endtime = 2e-9;
            }

            /* basic scenario */
            scen = std::make_shared<mbsolve::scenario>
                (dim, "basic", num_gridpoints, sim_endtime);

            scen->add_record(std::make_shared<mbsolve::record>
                             ("inv12", dt, position));

        } else {
            throw std::invalid_argument("Specified device not found!");
        }
        
        if (dim==1) {
            scen->add_record(std::make_shared<mbsolve::record>("e_z", mbsolve::record::type::electric,1,0, dt, position));
            scen->add_record(std::make_shared<mbsolve::record>("h_y", mbsolve::record::type::magnetic,1,0, dt, position));
        }else if(dim==2) {
            scen->add_record(std::make_shared<mbsolve::record>("e_x", mbsolve::record::type::electric,1,0, dt, position));
            scen->add_record(std::make_shared<mbsolve::record>("e_z", mbsolve::record::type::electric,2,0, dt, position));
            scen->add_record(std::make_shared<mbsolve::record>("h_y", mbsolve::record::type::magnetic,1,0, dt, position));
            //            scen->add_record(std::make_shared<mbsolve::record>("h_y",
            //                    mbsolve::record::type::magnetic,2,0, dt, position));
        }else{
            scen->add_record(std::make_shared<mbsolve::record>("e_x", mbsolve::record::type::electric,1,0, dt, position));
            scen->add_record(std::make_shared<mbsolve::record>("e_y", mbsolve::record::type::electric,2,0, dt, position));
            scen->add_record(std::make_shared<mbsolve::record>("e_z", mbsolve::record::type::electric,3,0, dt, position));
            scen->add_record(std::make_shared<mbsolve::record>("h_x", mbsolve::record::type::magnetic,1,0, dt, position));
            scen->add_record(std::make_shared<mbsolve::record>("h_y", mbsolve::record::type::magnetic,2,0, dt, position));
            scen->add_record(std::make_shared<mbsolve::record>("h_z", mbsolve::record::type::magnetic,3,0, dt, position));
        }
        if ((solver_method == "openmp-2lvl-os-red")||
            (solver_method == "openmp-3lvl-os-red")){
            if (dim==2) {
                solver_method=solver_method+"-2d";
            }else if (dim==3){
                solver_method=solver_method+"-3d";
            }
        } else {
            if (dim>1) {
                std::cout << "Dimension-argument beeing ignored. Solver methode only implemented in 1D" << std::endl;
            }
        }
        
	/* tic */
	timer.start();

        mbsolve::writer writer(writer_method);
	mbsolve::solver solver(solver_method, dev, scen);

        /* toc */
	timer.stop();
	ti::cpu_times times = timer.elapsed();
	std::cout << "Time required (setup): " << 1e-9 * times.wall
		  << std::endl;
	total_time +=1e-9 * times.wall;

	std::cout << solver.get_name() << std::endl;

	/* tic */
	timer.start();

	/* execute solver */
	solver.run();

	/* toc */
	timer.stop();
	times = timer.elapsed();
	std::cout << "Time required (run): " << 1e-9 * times.wall << std::endl;
	total_time +=1e-9 * times.wall;

	/* grid point updates per second */
	double gpups = 1e-6 * 1e9/times.wall * scen->get_num_gridpoints() *
            scen->get_endtime()/scen->get_timestep_size();
	std::cout << "Performance: " << gpups << " MGPU/s" << std::endl;

	/* tic */
	timer.start();

	/* write results */
	writer.write(output_file, solver.get_results(), dev, scen);

	/* toc */
	timer.stop();
	times = timer.elapsed();
	std::cout << "Time required (write): " << 1e-9 * times.wall
		  << std::endl;
	total_time +=1e-9 * times.wall;

	std::cout << "Time required (total): " << total_time << std::endl;

    } catch (std::exception& e) {
	std::cout << "Error: " << e.what() << std::endl;
	exit(1);
    }

    exit(0);
}
