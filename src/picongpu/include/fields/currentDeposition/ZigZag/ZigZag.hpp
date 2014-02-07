/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera
 *
 * This file is part of PIConGPU. 
 * 
 * PIConGPU is free software: you can redistribute it and/or modify 
 * it under the terms of the GNU General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
 * 
 * PIConGPU is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License for more details. 
 * 
 * You should have received a copy of the GNU General Public License 
 * along with PIConGPU.  
 * If not, see <http://www.gnu.org/licenses/>. 
 */

#pragma once

#include "simulation_defines.hpp"
#include "types.h"
#include "dimensions/DataSpace.hpp"
#include "dimensions/TVec.h"
#include "basicOperations.hpp"



namespace picongpu
{
namespace currentSolverZigZag
{
using namespace PMacc;

/**
 * \class Esirkepov implements the current deposition algorithm from T.Zh. Esirkepov
 * for an arbitrary particle assign function given as a template parameter.
 * See available shapes at "intermediateLib/particleShape".
 * paper: "Exact charge conservation scheme for Particle-in-Cell simulation
 *  with an arbitrary form-factor"
 */
template<typename ParticleAssign, typename NumericalCellType>
struct ZigZag
{
    static const int supp = ParticleAssign::support;

    static const int currentLowerMargin = supp / 2 + 1;
    static const int currentUpperMargin = (supp + 1) / 2 + 1;
    typedef PMacc::math::CT::Int<currentLowerMargin, currentLowerMargin, currentLowerMargin> LowerMargin;
    typedef PMacc::math::CT::Int<currentUpperMargin, currentUpperMargin, currentUpperMargin> UpperMargin;

    /* begin and end border is calculated for a particle with a support which travels
     * to the negative direction.
     * Later on all coordinates shifted thus we can solve the charge calculation
     * independend from the position of the particle. That means we must not look
     * if a particle position is >0.5 oder not (this is done by coordinate shifting to this defined range)
     * 
     * (supp + 1) % 2 is 1 for even supports else 0
     */
   // static const int begin = -supp / 2 + (supp + 1) % 2 - 1;
   // static const int end = supp / 2;
    static const int begin = supp / 2 + 1;
    static const int end = (supp + 1) / 2 + 1;

    float_X charge;

    template<typename DataBoxJ, typename PosType, typename VelType, typename ChargeType >
    DINLINE void operator()(DataBoxJ dataBoxJ,
                            const PosType pos1,
                            const VelType velocity,
                            const ChargeType charge, const float3_X& cellSize, const float_X deltaTime)
    {

        const float3_X deltaPos = float3_X(velocity.x() * deltaTime / cellSize.x(),
                                           velocity.y() * deltaTime / cellSize.y(),
                                           velocity.z() * deltaTime / cellSize.z());
        const PosType pos2 = pos1 - deltaPos;

        float3_X pos[2];
        pos[0] = (pos1) * cellSize;
        pos[1] = (pos2) * cellSize;

        float3_X I[2];
        float3_X W[2];
        float3_X r;
        float3_X F[2];

        for (int l = 0; l < 2; ++l)
        {
            for (uint32_t d = 0; d < DIM3; ++d)
            {
                I[l][d] = math::floor(pos[l][d] / cellSize[d]);
            }
        }
        for (uint32_t d = 0; d < DIM3; ++d)
        {
            r[d] = calc_r(I[0][d], I[1][d], cellSize[d], pos[0][d], pos[1][d]);

        }
        for (int l = 0; l < 2; ++l)
        {
            for (uint32_t d = 0; d < DIM3; ++d)
            {
                W[l][d] = calc_W(pos[l][d], r[d], I[l][d], cellSize[d]);
            }
        }
        for (uint32_t d = 0; d < DIM3; ++d)
        {
            F[0][d] = calc_F(pos[0][d], r[d], deltaTime, charge);
            F[1][d] = calc_F(-pos[1][d], -r[d], deltaTime, charge);
        }

        float_X frac = float_X(1.0) / float_X(CELL_VOLUME);
        for (int l = 0; l < 2; ++l)
        {

            /*x*/
            DataSpace<DIM3> jIdx_1((I[l].x()),
                                   I[l].y(),
                                   I[l].z());
            printf("[%i] %i %i %i\n",l,jIdx_1.x(),jIdx_1.y(),jIdx_1.z());
            float_X j_1 = frac * F[l].x()*(float_X(1.0) - W[l].y())*(float_X(1.0) - W[l].z());
            atomicAddWrapper(&(dataBoxJ(jIdx_1).x()), j_1);

            float_X j_2 = frac * F[l].x()*(W[l].y())*(float_X(1.0) - W[l].z());
            DataSpace<DIM3> jIdx_2((I[l].x()),
                                   I[l].y() + float_X(1.0),
                                   I[l].z());
            atomicAddWrapper(&(dataBoxJ(jIdx_2).x()), j_2);

            float_X j_3 = frac * F[l].x() * (float_X(1.0) - W[l].y())*(W[l].z());
            DataSpace<DIM3> jIdx_3((I[l].x()),
                                   I[l].y(),
                                   I[l].z() + float_X(1.0));
            atomicAddWrapper(&(dataBoxJ(jIdx_3).x()), j_3);

            float_X j_4 = frac * F[l].x() * (W[l].y())*(W[l].z());
            DataSpace<DIM3> jIdx_4((I[l].x() ),
                                   I[l].y() + float_X(1.0),
                                   I[l].z() + float_X(1.0));
            atomicAddWrapper(&(dataBoxJ(jIdx_4).x()), j_4);

            /*y*/
            DataSpace<DIM3> jIdx_5((I[l].x()),
                                   I[l].y() ,
                                   I[l].z());
            float_X j_5 = frac * F[l].y()*(float_X(1.0) - W[l].x())*(float_X(1.0) - W[l].z());
            atomicAddWrapper(&(dataBoxJ(jIdx_5).y()), j_5);

            float_X j_6 = frac * F[l].y()*(W[l].x())*(float_X(1.0) - W[l].z());
            DataSpace<DIM3> jIdx_6((I[l].x() + float_X(1.0)),
                                   I[l].y() ,
                                   I[l].z());
            atomicAddWrapper(&(dataBoxJ(jIdx_6).y()), j_6);

            float_X j_7 = frac * F[l].y() * (float_X(1.0) - W[l].x())*(W[l].z());
            DataSpace<DIM3> jIdx_7((I[l].x()),
                                   I[l].y() ,
                                   I[l].z() + float_X(1.0));
            atomicAddWrapper(&(dataBoxJ(jIdx_7).y()), j_7);

            float_X j_8 = frac * F[l].y() * (W[l].x())*(W[l].z());
            DataSpace<DIM3> jIdx_8((I[l].x() + float_X(1.0)),
                                   I[l].y() ,
                                   I[l].z() + float_X(1.0));
            atomicAddWrapper(&(dataBoxJ(jIdx_8).y()), j_8);

            /*z*/
            DataSpace<DIM3> jIdx_9(I[l].x(),
                                   I[l].y(),
                                   I[l].z() );
            float_X j_9 = frac * F[l].z()*(float_X(1.0) - W[l].x())*(float_X(1.0) - W[l].y());
            atomicAddWrapper(&(dataBoxJ(jIdx_9).z()), j_9);

            float_X j_10 = frac * F[l].z()*(W[l].x())*(float_X(1.0) - W[l].y());
            DataSpace<DIM3> jIdx_10((I[l].x() + float_X(1.0)),
                                    I[l].y(),
                                    I[l].z() );
            atomicAddWrapper(&(dataBoxJ(jIdx_10).z()), j_10);

            float_X j_11 = frac * F[l].z() * (float_X(1.0) - W[l].x())*(W[l].y());
            DataSpace<DIM3> jIdx_11((I[l].x()),
                                    I[l].y() + float_X(1.0),
                                    I[l].z());
            atomicAddWrapper(&(dataBoxJ(jIdx_11).z()), j_11);

            float_X j_12 = frac * F[l].z() * (W[l].x())*(W[l].y());
            DataSpace<DIM3> jIdx_12((I[l].x() + float_X(1.0)),
                                    I[l].y() + float_X(1.0),
                                    I[l].z());
            atomicAddWrapper(&(dataBoxJ(jIdx_12).z()), j_12);
              
        }
    }


private:

    DINLINE float_X
    calc_r(float_X i_1, float_X i_2, float_X delta_x, float_X x_1, float_X x_2)
    {

        const float_X min_1 = ::min(i_1*delta_x, i_2 * delta_x) + delta_x;
        const float_X max_1 = ::max(i_1*delta_x, i_2 * delta_x);
        const float_X max_2 = ::max(max_1, (x_1 + x_2) / float_X(2.));
        const float_X x_r = ::min(min_1, max_2);
        return x_r;
    }

    DINLINE float_X
    calc_W(float_X x, float_X x_r, float_X i, float_X cell)
    {
        return (x + x_r) / (float_X(2.0) * cell) - i;
    }

    /* for F_2 call with -x and -x_r*/
    DINLINE float_X
    calc_F(float_X x, float_X x_r, const float_X& delta_t, const float_X& q)
    {
        return q * (x_r - x) / delta_t;
    }
};

} //namespace currentSolverZigZag

namespace traits
{

/*Get margin of a solver
 * class must define a LowerMargin and UpperMargin 
 */
template<typename ParticleShape, typename NumericalCellType>
struct GetMargin<picongpu::currentSolverZigZag::ZigZag<ParticleShape, NumericalCellType> >
{
private:
    typedef picongpu::currentSolverZigZag::ZigZag<ParticleShape, NumericalCellType> Solver;
public:
    typedef typename Solver::LowerMargin LowerMargin;
    typedef typename Solver::UpperMargin UpperMargin;
};

} //namespace traits

} //namespace picongpu


