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

#ifndef MBSOLVE_SOURCE_H
#define MBSOLVE_SOURCE_H

#include <string>
#include <vector>
#include <types.hpp>

namespace mbsolve {

/**
 * Base class for all sources in \ref scenario.
 * \ingroup MBSOLVE_LIB
 */
class source
{
public:
    enum type { hard_source, soft_source };

protected:
    std::string m_name;

    /* amplitude */
    real m_ampl;

    /* (carrier) frequency */
    real m_freq;

    /* phase */
    real m_phase;

    /* position */
    real *m_position;

    /* internal resistance */
    real m_r_i;

    type m_type;
    
    real *m_pol;

public:

    source(const std::string& name, real *position, type source_type, real *pol,
           real ampl, real freq, real phase = 0) :
        m_name(name), m_position(position), m_type(source_type), m_pol(pol),
        m_ampl(ampl), m_freq(freq), m_phase(phase)
    {
    }

    /* TODO: get_value/calc_value : simplify */

    real get_value(real t, unsigned int dim_num = 0, real y = 0.0, real z = 0.0, real current_value = 0.0) const
    {
        /* calculate source value */
        real val = m_ampl * calc_value(t,dim_num,y,z);

        /* if type == thevenin, consider internal resistance */

        /* else if soft source */
        //return current_value + val;

        /* else -> hard source */
        return val;
    }

    /* calculate new value */
    virtual real calc_value(real /* t */,unsigned int /* dim_num */, real /* y */, real /* z */) const
    {
        return 0.0;
    }

    real get_position(unsigned int dim_num = 0) const {
        return m_position[dim_num];
    }

    type get_type() const {
        return m_type;
    }

    /* TODO: specify dm entry/field etc? */

    /* TODO: add position. how?
	   virtual const Quantity& position
    */
    /* TODO: add source type: hard, soft, thevenin */
    /* TODO: for thevenin: add internal resistance */


};
/*
class sine_source : public source
{
private:

    real calc_value(real t) const
    {
        return sin(2 * M_PI * m_freq * t + m_phase);
    }

public:
    sine_source(const std::string& name, real ampl, real freq, real phase) :
        source(name, ampl, freq, phase)
    {
    }

    };*/

class sech_pulse : public source
{
private:

    real m_beta;
    real m_phase_sin;

public:
    sech_pulse(const std::string& name, real *position, type source_type,
               real *pol,
               real ampl, real freq,
               real phase,
               real beta, real phase_sin = 0.0) :
        source(name, position, source_type, pol, ampl, freq, phase), m_beta(beta),
        m_phase_sin(phase_sin)
    {
    }

    real calc_value(real t, unsigned int dim_num, real y, real z) const
    {
        real ret = 0.0;
        ret = 1/std::cosh(m_beta * t - m_phase) *
            sin(2 * M_PI * m_freq * t - m_phase_sin);
        if ((pow(y-m_position[1],2)<0.01) && (pow(z-m_position[2],2)<0.01)) {
            
        }
        return ret*m_pol[dim_num];
    }

};

class single_cycle_pulse : public source
{
private:
    real m_beta;

public:
    single_cycle_pulse(const std::string& name, real *position,
                       type source_type, real *pol,
                       real ampl, real freq,
                       real phase,
                       real beta) :
        source(name, position, source_type, pol, ampl, freq, phase), m_beta(beta)
    {
    }

    real calc_value(real t, unsigned int dim_num, real y, real z) const
    {
        return 1/std::cosh(m_beta * (t - m_phase)) *
            sin(2 * M_PI * m_freq * (t - m_phase - 1/(m_freq * 4)))*m_pol[dim_num];
    }

};

class gauss : public source
{
private:

    real m_beta;
    real m_sigma;
    real m_phase_sin;

public:
    gauss(const std::string& name, real *position, type source_type, real *pol,
               real ampl, real freq,
               real phase, real sigma, real beta, real phase_sin = 0.0) :
        source(name, position, source_type, pol, ampl, freq, phase), m_sigma(sigma),
        m_beta(beta), m_phase_sin(phase_sin)
    {
    }

    real calc_value(real t, unsigned int dim_num, real y_p, real z_p) const
    {
        real ret;
        real y = y_p - m_position[1];
        real z = z_p - m_position[2];
        
        ret = 1/std::cosh(m_beta * t - m_phase) * sin(2 * M_PI * m_freq * t - m_phase_sin);
        ret *= 1/(std::sqrt(2*M_PI*pow(m_sigma,2)))*
        std::exp(-1*pow(y,2)/(2*pow(m_sigma,2)));
        ret *= 1/(std::sqrt(2*M_PI*pow(m_sigma,2)))*
        std::exp(-1*pow(z,2)/(2*pow(m_sigma,2)));
        
        return ret*m_pol[dim_num];
    }

};
    
class gauss_beam : public source
{
private:

    real m_z;
    real m_z_0;
    real m_lamda;
    real m_w_0;
    int m_n;
    int m_m;
    
    real hermite(real xy,int nm) const
    {
        real ret=2*xy;
        real old=1;
        if (nm==0) {
            ret=1;
        }
        for (int i=1; i<nm; i++) {
            real inter=2*xy*ret-2*i*old;
            old = ret;
            ret = inter;
        }
        
        return ret;
    }

public:
    gauss_beam(const std::string& name, real *position, type source_type, real *pol,
               real ampl, real freq, real phase, real min_width,
               int mode_n = 1, int mode_m = 1, real z = 1) :
        source(name, position, source_type, pol, ampl, freq, phase), m_w_0(min_width),
        m_n(mode_n), m_m(mode_m), m_z(z)
    {
        m_lamda = 299792458 / m_freq; // / std::sqrt(mu_r*eps_r);
//        m_w_0 = std::sqrt(m_lamda * m_z_0 / M_PI);
        m_z_0 = M_PI * pow(m_w_0,2)/m_lamda;
        
    }

    real calc_value(real t, unsigned int dim_num, real y_p, real z_p) const
    {
        complex ret;
        complex i = complex(0.0,1.0);//sqrt(-1);
        real y = y_p - m_position[1];
        real z = z_p - m_position[2];
        
        real m_w = m_w_0 * std::sqrt(1+pow(m_z/m_z_0,2));
        real m_zeta = std::atan(m_z/m_z_0);
        real m_R = m_z * (1+pow(m_z_0/m_z,2));
        real m_k = 2 * M_PI/m_lamda;
        
        ret = m_ampl * m_w_0/m_w * hermite(M_SQRT2*y/m_w,m_n)
            * hermite(M_SQRT2*z/m_w,m_m)
            * std::exp(-(pow(y,2)+pow(z,2))/pow(m_w,2))
            * std::exp(- i*m_k*(pow(y,2) + pow(z,2))/(2*m_R))
            * std::exp(- i * m_k * m_z)
            * std::exp(i * ((real)(m_n+m_m+1)) * m_zeta)
            * 1.0/std::cosh(2e14 * t - m_phase)
            *  sin(2 * M_PI * m_freq * t);
//        std::exp(i* m_k * 299792458.0 * t);
//        std::cout << ret << "; ";
        return (m_pol[dim_num]*ret).real();
    }

};


/* TODO: custom functor source / callback function? */

}

#endif
