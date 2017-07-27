/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Rene Widera
 *
 * This file is post of PIConGPU.
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
#include "pmacc_types.hpp"
#include "dimensions/DataSpace.hpp"
#include "basicOperations.hpp"
#include <cuSTL/cursor/tools/twistVectorFieldAxes.hpp>
#include "algorithms/FieldToParticleInterpolation.hpp"
#include "algorithms/ShiftCoordinateSystem.hpp"
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/copy.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/back_inserter.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/equal_to.hpp>
#include <boost/mpl/if.hpp>
#include "compileTime/AllCombinations.hpp"
#include "fields/currentDeposition/ZigZag/EvalAssignmentFunction.hpp"

namespace picongpu
{
namespace currentSolver
{
using namespace PMacc;

/**\class ZigZag charge conservation method
 *
 * @see ZigZag.def for paper references
 */
template<typename T_ParticleShape, uint32_t T_dim>
struct ZigZagShape
{
    /* cloud shape: describe the form factor of a posticle
     * assignment shape: integral over the cloud shape (this shape is defined by the user in
     * species.posam for a species)
     */
    typedef particles::shapes::CIC ParticleShape;
    typedef typename ParticleShape::ChargeAssignmentOnSupport ParticleAssign;
    typedef typename ParticleShape::CloudShape::ChargeAssignmentOnSupport CloudShape;
    BOOST_STATIC_CONSTEXPR int supp = ParticleAssign::support;

    BOOST_STATIC_CONSTEXPR int currentLowerMargin = supp / 2 + 1 - (supp + 1) % 2;
    BOOST_STATIC_CONSTEXPR int currentUpperMargin = (supp + 1) / 2 + 1;
    typedef typename PMacc::math::CT::make_Int<simDim, currentLowerMargin>::type LowerMargin;
    typedef typename PMacc::math::CT::make_Int<simDim, currentUpperMargin>::type UpperMargin;


    /** add current of a moving posticle to the global current
     *
     * @posam dataBoxJ DataBox with current field
     * @posam pos1 current position of the posticle
     * @posam velocity velocity of the macro posticle
     * @posam charge charge of the macro posticle
     * @posam deltaTime dime difference of one simulation time step
     */
    template<typename DataBoxJ, typename PosType, typename VelType, typename ChargeType >
    DINLINE void operator()(DataBoxJ dataBoxJ,
                            const PosType pos1,
                            const VelType velocity,
                            const ChargeType charge, const float_X deltaTime)
    {

        floatD_X deltaPos;
        for (uint32_t d = 0; d < simDim; ++d)
            deltaPos[d] = (velocity[d] * deltaTime) / cellSize[d];

        /*note: all positions are normalized to the grid*/
        floatD_X pos[2];
        pos[0] = (pos1 - deltaPos);
        pos[1] = (pos1);

        DataSpace<simDim> I[2];
        floatD_X relayPoint;


        for (int l = 0; l < 2; ++l)
        {
            for (uint32_t d = 0; d < simDim; ++d)
            {
                I[l][d] = math::floor(pos[l][d]);
            }
        }
        for (uint32_t d = 0; d < simDim; ++d)
        {
            relayPoint[d] = calc_relayPoint(I[0][d], I[1][d], pos[0][d], pos[1][d]);
        }
        constexpr float_X volume_reci = float_X(1.0) / float_X(CELL_VOLUME);

        auto j = dataBoxJ.toCursor();
#       pragma unroll 2
        for(int l = 0; l< 2; ++l)
        {
            //const int l = pId;
            float_X sign = float_X(1.) - float_X(2.) * l;

            int x = I[l].x();
            int y = I[l].y();
            int z = I[l].z();

            floatD_X deltaPos(( relayPoint - pos[l] ) * sign);
            float_X tmpX = deltaPos.y() * deltaPos.z() * (float_X(1.0) / float_X(12.0));
            float_X tmpY = deltaPos.x() * deltaPos.z() * (float_X(1.0) / float_X(12.0));
            float_X tmpZ = deltaPos.x() * deltaPos.y() * (float_X(1.0) / float_X(12.0));

            float_X const tmp = deltaPos.x() * deltaPos.y() * deltaPos.z() * (float_X(1.0) / float_X(12.0));
            atomicAddWrapper(
                &( (*j( x, y, z )).x() ),
                CloudShape()( distance(relayPoint.x(), pos[l].x(), x + float_X( 0.5 )) ) *
                volume_reci * cellSize.x() * charge  / DELTA_T * ( deltaPos.x() *
                ParticleAssign()( distance(relayPoint.y(), pos[l].y(), y) ) *
                ParticleAssign()( distance(relayPoint.z(), pos[l].z(), z) ) +
                tmp * deltaPos.x())
            );

            // X
            atomicAddWrapper(
                &( (*j( x, y, z )).x() ),
                CloudShape()( distance(relayPoint.x(), pos[l].x(), x + float_X( 0.5 )) ) *
                volume_reci * cellSize.x() * F1(deltaPos.x(), charge) * (
                ParticleAssign()( distance(relayPoint.y(), pos[l].y(), y) ) *
                ParticleAssign()( distance(relayPoint.z(), pos[l].z(), z) ) +
                tmpX)
            );
            atomicAddWrapper(
                &( (*j( x, y + 1, z )).x() ),
                CloudShape()( distance(relayPoint.x(), pos[l].x(), x + float_X( 0.5 )) ) *
                volume_reci * cellSize.x() * F1(deltaPos.x(), charge) * (
                ParticleAssign()( distance(relayPoint.y(), pos[l].y(), y + 1 ) ) *
                ParticleAssign()( distance(relayPoint.z(), pos[l].z(), z) ) -
                tmpX)
            );
            atomicAddWrapper(
                &( (*j( x, y, z + 1 )).x() ),
                CloudShape()( distance(relayPoint.x(), pos[l].x(), x + float_X( 0.5 )) ) *
                volume_reci * cellSize.x() * F1(deltaPos.x(), charge) * (
                ParticleAssign()( distance(relayPoint.y(), pos[l].y(), y) ) *
                ParticleAssign()( distance(relayPoint.z(), pos[l].z(), z + 1 ) ) -
                tmpX)
            );
            atomicAddWrapper(
                &( (*j( x, y + 1, z + 1 )).x() ),
                CloudShape()( distance(relayPoint.x(), pos[l].x(), x + float_X( 0.5 )) ) *
                volume_reci * cellSize.x() * F1(deltaPos.x(), charge) * (
                ParticleAssign()( distance(relayPoint.y(), pos[l].y(), y + 1) ) *
                ParticleAssign()( distance(relayPoint.z(), pos[l].z(), z + 1) ) +
                tmpX)
            );
            // Y
            atomicAddWrapper(
                &( (*j(x, y, z )).y() ),
                CloudShape()( distance(relayPoint.y(), pos[l].y(), y + float_X( 0.5 )) ) *
                volume_reci * cellSize.y() * F1(deltaPos.y(), charge) * (
                ParticleAssign()( distance(relayPoint.x(), pos[l].x(), x) ) *
                ParticleAssign()( distance(relayPoint.z(), pos[l].z(), z) ) +
                tmpY)
            );
            atomicAddWrapper(
                &( (*j( x + 1, y, z )).y() ),
                CloudShape()( distance(relayPoint.y(), pos[l].y(), y + float_X( 0.5 )) ) *
                volume_reci * cellSize.y() * F1(deltaPos.y(), charge) * (
                ParticleAssign()( distance(relayPoint.x(), pos[l].x(), x + 1) ) *
                ParticleAssign()( distance(relayPoint.z(), pos[l].z(), z) ) -
                tmpY)
            );
            atomicAddWrapper(
                &( (*j( x, y, z + 1 )).y() ),
                CloudShape()( distance(relayPoint.y(), pos[l].y(), y + float_X( 0.5 )) ) *
                volume_reci * cellSize.y() * F1(deltaPos.y(), charge) * (
                ParticleAssign()( distance(relayPoint.x(), pos[l].x(), x) ) *
                ParticleAssign()( distance(relayPoint.z(), pos[l].z(), z + 1) ) -
                tmpY)
            );
            atomicAddWrapper(
                &( (*j( x + 1, y, z + 1 )).y() ),
                CloudShape()( distance(relayPoint.y(), pos[l].y(), y + float_X( 0.5 )) ) *
                volume_reci * cellSize.y() * F1(deltaPos.y(), charge) * (
                ParticleAssign()( distance(relayPoint.x(), pos[l].x(), x + 1) ) *
                ParticleAssign()( distance(relayPoint.z(), pos[l].z(), z + 1) ) +
                tmpY)
            );

            // Z
            atomicAddWrapper(
                &( (*j(x, y, z )).z() ),
                CloudShape()( distance(relayPoint.z(), pos[l].z(), z + float_X( 0.5 )) ) *
                volume_reci * cellSize.z() * F1(deltaPos.z(), charge) * (
                ParticleAssign()( distance(relayPoint.x(), pos[l].x(), x) ) *
                ParticleAssign()( distance(relayPoint.y(), pos[l].y(), y) ) +
                tmpZ)
            );
            atomicAddWrapper(
                &( (*j( x + 1, y, z )).z() ),
                CloudShape()( distance(relayPoint.z(), pos[l].z(), z + float_X( 0.5 )) ) *
                volume_reci * cellSize.z() * F1(deltaPos.z(), charge) * (
                ParticleAssign()( distance(relayPoint.x(), pos[l].x(), x + 1) ) *
                ParticleAssign()( distance(relayPoint.y(), pos[l].y(), y) ) -
                tmpZ)
            );

            atomicAddWrapper(
                &( (*j(x, y + 1, z )).z() ),
                CloudShape()( distance(relayPoint.z(), pos[l].z(), z + float_X( 0.5 )) ) *
                volume_reci * cellSize.z() * F1(deltaPos.z(), charge) * (
                ParticleAssign()( distance(relayPoint.x(), pos[l].x(), x) ) *
                ParticleAssign()( distance(relayPoint.y(), pos[l].y(), y + 1) ) -
                tmpZ)
            );
            atomicAddWrapper(
                &( (*j( x + 1, y + 1, z )).z() ),
                CloudShape()( distance(relayPoint.z(), pos[l].z(), z + float_X( 0.5 )) ) *
                volume_reci * cellSize.z() * F1(deltaPos.z(), charge) * (
                ParticleAssign()( distance(relayPoint.x(), pos[l].x(), x + 1) ) *
                ParticleAssign()( distance(relayPoint.y(), pos[l].y(), y + 1) ) +
                tmpZ)
            );

        }
    }

    static PMacc::traits::StringProperty getStringProperties()
    {
        PMacc::traits::StringProperty propList( "name", "ZigZagCIC" );
        return propList;
    }

private:

    DINLINE float_X
    F1(const float_X deltaPosDir, const float_X charge) const
    {
        return charge * deltaPosDir / DELTA_T;
    }

    DINLINE float_X
    distance(const float_X x_r, const float_X x, const float_X i) const
    {
        return i - (x + x_r)/float_X(2.0);
    }

    /** calculate virtual point were we split our posticle trajectory
     *
     * The relay point calculation differs from the paper version in the point
     * that the trajectory of a posticle which does not leave the cell is not splitted.
     * The relay point for a posticle which does not leave the cell is set to the
     * current position `x_2`
     *
     * @posam i_1 grid point which is less than x_1 (`i_1=floor(x_1)`)
     * @posam i_2 grid point which is less than x_2 (`i_2=floor(x_2)`)
     * @posam x_1 begin position of the posticle trajectory
     * @posam x_2 end position of the posticle trajectory
     * @return relay point for posticle trajectory
     */
    DINLINE float_X
    calc_relayPoint(const float_X i_1, const float_X i_2, const float_X x_1, const float_X x_2) const
    {
        return i_1 == i_2 ? (x_1 + x_2) / float_X(2.0) : ::max(i_1, i_2);
    }
};

} //namespace currentSolver

} //namespace picongpu
