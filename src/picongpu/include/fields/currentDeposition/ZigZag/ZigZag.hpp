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
#include <cuSTL/cursor/tools/twistVectorFieldAxes.hpp>
#include "algorithms/FieldToParticleInterpolation.hpp"


namespace picongpu
{
namespace currentSolverZigZag
{
using namespace PMacc;

/**
 * \class ZigZag charge conservation method
 * 1. order paper: "A new charge conservation method in electromagnetic particle-in-cell simulations"
 *                 by T. Umeda, Y. Omura, T. Tominaga, H. Matsumoto
 * 2. order paper: "Charge conservation methods for computing current densities in electromagnetic particle-in-cell simulations" 
 *                 by T. Umeda, Y. Omura, H. Matsumoto
 * 3. order paper: "High-Order Interpolation Algorithms for Charge Conservation in Particle-in-Cell Simulation"
 *                 by Jinqing Yu, Xiaolin Jin, Weimin Zhou, Bin Li, Yuqiu Gu
 */
template<uint32_t T_Dim,typename T_ParticleShape>
struct ZigZag
{
    typedef T_ParticleShape ParticleShape;
    typedef typename ParticleShape::ChargeAssignment ParticleAssign;
    static const int supp = ParticleAssign::support;

    static const int currentLowerMargin = supp / 2 + 1;
    static const int currentUpperMargin = (supp + 1) / 2 + 1;
    typedef PMacc::math::CT::Int<currentLowerMargin, currentLowerMargin, currentLowerMargin> LowerMargin;
    typedef PMacc::math::CT::Int<currentUpperMargin, currentUpperMargin, currentUpperMargin> UpperMargin;

    static const int begin = -supp / 2 + (supp + 1) % 2;
    static const int end = begin + supp + supp % 2;

    static const int dir_begin = -(supp) / 2;
    static const int dir_end = dir_begin + (supp);

    /* begin and end border is calculated for a particle with a support which travels
     * to the negative direction.
     * Later on all coordinates shifted thus we can solve the charge calculation
     * independend from the position of the particle. That means we must not look
     * if a particle position is >0.5 oder not (this is done by coordinate shifting to this defined range)
     * 
     * (supp + 1) % 2 is 1 for even supports else 0
     */

    float_X charge;

    template<typename DataBoxJ, typename PosType, typename VelType, typename ChargeType >
    DINLINE void operator()(DataBoxJ dataBoxJ,
                            const PosType pos1,
                            const VelType velocity,
                            const ChargeType charge, const float_X deltaTime)
    {

        const float3_X deltaPos = float3_X(velocity.x() * deltaTime / cellSize.x(),
                                           velocity.y() * deltaTime / cellSize.y(),
                                           velocity.z() * deltaTime / cellSize.z());

        float3_X pos[2];
        pos[0] = (pos1 - deltaPos);
        pos[1] = (pos1);

        DataSpace<DIM3> I[2];
        float3_X r;


        for (int l = 0; l < 2; ++l)
        {
            for (uint32_t d = 0; d < DIM3; ++d)
            {
                I[l][d] = math::floor(pos[l][d]);
            }
        }
        for (uint32_t d = 0; d < DIM3; ++d)
        {
            r[d] = calc_r(I[0][d], I[1][d], pos[0][d], pos[1][d]);

        }
        const float_X volume_reci = float_X(1.0) / float_X(CELL_VOLUME);

        using namespace cursor::tools;
        // floatD_X pos_tmp(pos1);

        BOOST_AUTO(cursorJ, dataBoxJ.toCursor());

        for (int l = 0; l < 2; ++l)
        {
            float3_X IcP;
            /* sign= 1 if l=0
             * sign=-1 if l=1
             */
            const float_X sign = float_X(1.0) - float_X(2.0) * float_X(l);
            for (uint32_t d = 0; d < DIM3; ++d)
            {
                IcP[d] = calc_InCellPos(pos[l][d], r[d], I[l][d]);
            }

            BOOST_AUTO(cursorJ, dataBoxJ.shift(precisionCast<int>(I[l])).toCursor());

            helper(cursorJ, IcP, sign * pos[l][0], sign * r[0], volume_reci, cellSize.x(), deltaTime, charge);
            helper(twistVectorFieldAxes<PMacc::math::CT::Int < 1, 0, 2 > >(cursorJ), float3_X(IcP[1], IcP[0], IcP[2]), sign * pos[l][1], sign * r[1], volume_reci, cellSize.y(), deltaTime, charge);
            helper(twistVectorFieldAxes<PMacc::math::CT::Int < 2, 0, 1 > >(cursorJ), float3_X(IcP[2], IcP[0], IcP[1]), sign * pos[l][2], sign * r[2], volume_reci, cellSize.z(), deltaTime, charge);
        }
    }

    template<typename JCurser>
    DINLINE void helper(JCurser dataBoxJ,
                        const float3_X& pos,
                        const float_X pos_x,
                        const float_X r,
                        const float_X volume_reci,
                        const float_X cellLength,
                        const float_X deltaTime,
                        const float_X charge)
    {


        typedef typename ParticleShape::CloudShape::ChargeAssignment CloudShapeAssign;
        PMACC_AUTO(shape, ParticleAssign());
        PMACC_AUTO(cloudShapeAssign, CloudShapeAssign());


        for (int x = dir_begin; x < dir_end; ++x)
        {
            const float_X F = calc_F(pos_x, r, deltaTime, charge) * cloudShapeAssign(float_X(x) + float_X(0.5) - pos.x());

            for (int y = begin; y < end; ++y)

            {
                const float_X shape_y = shape(float_X(y) - pos.y());
                for (int z = begin; z < end; ++z)
                {
                    const DataSpace<DIM3> jIdx(x, y, z);
                    float_X j = cellLength * volume_reci * F * shape(float_X(z) - pos.z()) * shape_y;
                    if (j != float_X(0.0))
                        atomicAddWrapper(&((*dataBoxJ(jIdx)).x()), j);
                }
            }
        }
    }

private:

    DINLINE float_X
    calc_r(float_X i_1, float_X i_2, float_X x_1, float_X x_2)
    {

        const float_X min_1 = ::min(i_1, i_2) + float_X(1.0);
        const float_X max_1 = ::max(i_1, i_2);
        const float_X max_2 = ::max(max_1, (x_1 + x_2) / float_X(2.));
        const float_X x_r = ::min(min_1, max_2);
        return x_r;
    }

    DINLINE float_X
    calc_InCellPos(float_X x, float_X x_r, float_X i)
    {
        return (x + x_r) / (float_X(2.0)) - i;
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
template<uint32_t T_Dim,typename T_ParticleShape>
struct GetMargin<picongpu::currentSolverZigZag::ZigZag<T_Dim, T_ParticleShape> >
{
private:
    typedef picongpu::currentSolverZigZag::ZigZag<T_Dim, T_ParticleShape> Solver;
public:
    typedef typename Solver::LowerMargin LowerMargin;
    typedef typename Solver::UpperMargin UpperMargin;
};

} //namespace traits

} //namespace picongpu


