/**
 * Copyright 2016 Rene Widera
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

#include "cuSTL/cursor/Cursor.hpp"

#include "fields/currentDeposition/EmZ/EmZ.def"
#include "fields/currentDeposition/RelayPoint.hpp"
#include "fields/currentDeposition/EmZ/DepositCurrent.hpp"
#include "fields/currentDeposition/Esirkepov/Line.hpp"

namespace picongpu
{
namespace currentSolver
{

template<
    typename T_ParticleShape
>
struct EmZ
{
    typedef typename T_ParticleShape::ChargeAssignmentOnSupport ParticleAssign;
    BOOST_STATIC_CONSTEXPR int supp = ParticleAssign::support;

    BOOST_STATIC_CONSTEXPR int currentLowerMargin = supp / 2 + 1 - (supp + 1) % 2;
    BOOST_STATIC_CONSTEXPR int currentUpperMargin = (supp + 1) / 2 + 1;
    typedef typename PMacc::math::CT::make_Int<simDim, currentLowerMargin>::type LowerMargin;
    typedef typename PMacc::math::CT::make_Int<simDim, currentUpperMargin>::type UpperMargin;

    /* begin and end border is calculated for the current time step were the old
     * position of the particle in the previous time step is smaller than the current position
     * Later on all coordinates are shifted thus we can solve the charge calculation
     * in support + 1 steps.
     *
     * For the case were previous position is greater than current position we correct
     * begin and end on runtime and add +1 to begin and end.
     */
    BOOST_STATIC_CONSTEXPR int begin = -currentLowerMargin + 1;
    BOOST_STATIC_CONSTEXPR int end = begin + supp;


    /** deposit the current of a particle
     *
     * @tparam DataBoxJ any PMacc DataBox
     *
     * @param dataBoxJ box shifted to the cell of particle
     * @param posEnd position of the particle after it is pushed
     * @param velocity velocity of the particle
     * @param charge charge of the particle
     * @param deltaTime time of one time step
     */
    template<
        typename DataBoxJ
    >
    DINLINE void
    operator()(
        DataBoxJ dataBoxJ,
        const floatD_X posEnd,
        const float3_X velocity,
        const float_X charge,
        const float_X /* deltaTime */
    )
    {
        floatD_X deltaPos;
        for ( uint32_t d = 0; d < simDim; ++d )
            deltaPos[d] = ( velocity[d] * DELTA_T ) / cellSize[d];

        /*note: all positions are normalized to the grid*/
        const floatD_X posStart( posEnd - deltaPos );

        DataSpace<simDim> I[2];
        floatD_X relayPoint;

        /* calculate the relay point for the trajectory splitting */
        for ( uint32_t d = 0; d < simDim; ++d )
        {
            BOOST_CONSTEXPR_OR_CONST bool isSupportEven = ( supp % 2 == 0 );
            relayPoint[d] = RelayPoint< isSupportEven >()(
                I[0][d],
                I[1][d],
                posStart[d],
                posEnd[d]
            );
        }

        Line< floatD_X > line;
        const float_X chargeDensity = charge / CELL_VOLUME;

        /* Esirkepov implementation for the current deposition */
        emz::DepositCurrent<
            ParticleAssign,
            begin,
            end
        > deposit;

        /* calculate positions for the second virtual particle */
        for (uint32_t d = 0; d < simDim; ++d)
        {
            line.m_pos0[d] = calc_InCellPos(
                posStart[d],
                I[0][d]
            );
            line.m_pos1[d] = calc_InCellPos(
                relayPoint[d],
                I[0][d]
            );
        }

        const bool twoParticlesNeeded = I[0] != I[1];

        deposit(
            dataBoxJ.shift( I[0] ).toCursor(),
            line,
            chargeDensity,
            velocity.z() * ( twoParticlesNeeded ? float_X(0.5) : float_X(1.0) )
        );

        /* detect if there is a second virtual particle */
        if( twoParticlesNeeded )
        {
            /* calculate positions for the second virtual particle */
            for (uint32_t d = 0; d < simDim; ++d)
            {
                /* switched start and end point */
                line.m_pos1[d] = calc_InCellPos(
                    posEnd[d],
                    I[1][d]
                );
                line.m_pos0[d] = calc_InCellPos(
                    relayPoint[d],
                    I[1][d]
                );
            }
            deposit(
                dataBoxJ.shift( I[1] ).toCursor(),
                line,
                chargeDensity,
                velocity.z() * float_X(0.5)
            );
        }
    }

    static PMacc::traits::StringProperty
    getStringProperties()
    {
        PMacc::traits::StringProperty propList( "name", "EmZ" );
        return propList;
    }


    /** get normalized in cell particle position
     *
     * @param x position of the particle
     * @param i shift of grid (only integral positions are allowed)
     * @return in cell position
     */
    DINLINE float_X
    calc_InCellPos(
        const float_X x,
        const float_X i
    ) const
    {
        return x - i;
    }
};

} //namespace currentSolver

} //namespace picongpu
