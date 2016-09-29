/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Rene Widera
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
#include "pmacc_types.hpp"
#include "cuSTL/cursor/Cursor.hpp"
#include "basicOperations.hpp"
#include <cuSTL/cursor/tools/twistVectorFieldAxes.hpp>
#include <cuSTL/cursor/compile-time/SafeCursor.hpp>
#include "fields/currentDeposition/Esirkepov/Esirkepov.def"
#include "fields/currentDeposition/Esirkepov/Line.hpp"

namespace picongpu
{
namespace currentSolver
{
    using namespace PMacc;

    template<bool isEven>
    struct Calc_Relay
    {
       /** calculate virtual point were we split our particle trajectory
        *
        * The relay point calculation differs from the paper version in the point
        * that the trajectory of a particle which does not leave the cell is not splitted.
        * The relay point for a particle which does not leave the cell is set to the
        * current position `x_2`
        *
        * @param i_1 grid point which is less than x_1 (`i_1=floor(x_1)`)
        * @param i_2 grid point which is less than x_2 (`i_2=floor(x_2)`)
        * @param x_1 begin position of the particle trajectory
        * @param x_2 end position of the particle trajectory
        * @return relay point for particle trajectory
        */
        DINLINE float_X
        operator()( int& i_1, int& i_2, const float_X x_1, const float_X x_2) const
        {
            i_1 = math::floor(x_1);
            i_2 = math::floor(x_2);
            /* paper version:
             *   i_1 == i_2 ? (x_1 + x_2) / float_X(2.0) : ::max(i_1, i_2);
             */
            return i_1 == i_2 ? x_2 : ::max(i_1, i_2);
        }
    };

    template<>
    struct Calc_Relay<false>
    {
       /** calculate virtual point were we split our particle trajectory
        *
        * The relay point calculation differs from the paper version in the point
        * that the trajectory of a particle which does not leave the cell is not splitted.
        * The relay point for a particle which does not leave the cell is set to the
        * current position `x_2`
        *
        * @param i_1 grid point which is less than x_1 (`i_1=floor(x_1)`)
        * @param i_2 grid point which is less than x_2 (`i_2=floor(x_2)`)
        * @param x_1 begin position of the particle trajectory
        * @param x_2 end position of the particle trajectory
        * @return relay point for particle trajectory
        */
        DINLINE float_X
        operator()( int& i_1, int& i_2, const float_X x_1, const float_X x_2) const
        {
            i_1 = math::floor(x_1 + float_X(0.5));
            i_2 = math::floor(x_2 + float_X(0.5));

            /* paper version:
             *   i_1 == i_2 ? (x_1 + x_2) / float_X(2.0) : ::max(i_1, i_2);
             */
            return i_1 == i_2 ? x_2 : float_X(i_1+i_2)/float_X(2.0);
        }
    };


template<typename T_ParticleShape>
struct Esirkepov<T_ParticleShape, DIM3>
{
    typedef typename T_ParticleShape::ChargeAssignmentOnSupport ParticleAssign;
    BOOST_STATIC_CONSTEXPR int supp = ParticleAssign::support;

    BOOST_STATIC_CONSTEXPR int currentLowerMargin = supp / 2 + 1 - (supp + 1) % 2;
    BOOST_STATIC_CONSTEXPR int currentUpperMargin = (supp + 1) / 2 + 1;
    typedef PMacc::math::CT::Int<currentLowerMargin, currentLowerMargin, currentLowerMargin> LowerMargin;
    typedef PMacc::math::CT::Int<currentUpperMargin, currentUpperMargin, currentUpperMargin> UpperMargin;

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


    /* At the moment Esirkepov only support YeeCell were W is defined at origin (0,0,0)
     *
     * \todo: please fix me that we can use CenteredCell
     */
    template<typename DataBoxJ, typename PosType, typename VelType, typename ChargeType >
    DINLINE void operator()(DataBoxJ dataBoxJ,
                            const PosType pos1,
                            const VelType velocity,
                            const ChargeType charge,
                            const float_X deltaTime)
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

        for (uint32_t d = 0; d < simDim; ++d)
        {
            relayPoint[d] = Calc_Relay<supp%2==0>()(I[0][d], I[1][d], pos[0][d], pos[1][d]);
        }

        const float_X currentVolumeDensity = charge / (CELL_VOLUME * DELTA_T);

        floatD_X inCellPosStart;
        floatD_X inCellPosEnd;

        for (uint32_t d = 0; d < simDim; ++d)
        {
            const float_X pos_tmp = pos[0][d];
            const float_X tmpRelayPoint = relayPoint[d];
            inCellPosStart[d] = calc_InCellPos(pos_tmp, I[0][d]);
            inCellPosEnd[d] = calc_InCellPos(tmpRelayPoint, I[0][d]);
        }
        calc( dataBoxJ.shift(
            precisionCast<int>(I[0])).toCursor(),
            Line<float3_X>( inCellPosStart, inCellPosEnd ),
            currentVolumeDensity
        );

        for (uint32_t d = 0; d < simDim; ++d)
        {
            const float_X pos_tmp = pos[1][d];
            const float_X tmpRelayPoint = relayPoint[d];
            inCellPosStart[d] = calc_InCellPos(pos_tmp, I[1][d]);
            inCellPosEnd[d] = calc_InCellPos(tmpRelayPoint, I[1][d]);
        }
        calc( dataBoxJ.shift(
            precisionCast<int>(I[1])).toCursor(),
            Line<float3_X>( inCellPosEnd, inCellPosStart ), /* switched start and end point */
            currentVolumeDensity
        );

    }

    /* At the moment Esirkepov only support YeeCell were W is defined at origin (0,0,0)
     *
     * \todo: please fix me that we can use CenteredCell
     */
    template<typename T_Cursor >
    DINLINE void calc( const T_Cursor& cursorJ,
                       const Line<float3_X>& line,
                       const float_X currentVolumeDensity)
    {
        /**
         * \brief the following three calls separate the 3D current deposition
         * into three independent 1D calls, each for one direction and current component.
         * Therefore the coordinate system has to be rotated so that the z-direction
         * is always specific.
         */
        using namespace cursor::tools;
        cptCurrent1D(twistVectorFieldAxes<PMacc::math::CT::Int < 1, 2, 0 > >(cursorJ), rotateOrigin < 1, 2, 0 > (line), cellSize.x()*currentVolumeDensity);
        cptCurrent1D(twistVectorFieldAxes<PMacc::math::CT::Int < 2, 0, 1 > >(cursorJ), rotateOrigin < 2, 0, 1 > (line), cellSize.y()*currentVolumeDensity);
        cptCurrent1D(cursorJ, line, cellSize.z()*currentVolumeDensity);

    }

    /**
     * deposites current in z-direction
     * \param cursorJ cursor pointing at the current density field of the particle's cell
     * \param line trajectory of the particle from to last to the current time step
     * \param cellEdgeLength length of edge of the cell in z-direction
     */
    template<typename CursorJ, typename T_Line>
    DINLINE void cptCurrent1D(CursorJ cursorJ,
                              const T_Line line,
                              const float_X currentSurfaceDensity)
    {

        if(line.m_pos0[2] == line.m_pos1[2])
            return;
        /* pick every cell in the xy-plane that is overlapped by particle's
         * form factor and deposit the current for the cells above and beneath
         * that cell and for the cell itself.
         */
        for (int i = begin ; i < end ; ++i)
        {
            for (int j = begin ; j < end ; ++j)
            {
                /* This is the implementation of the FORTRAN W(i,j,k,3)/ C style W(i,j,k,2) version from
                 * Esirkepov paper. All coordinates are rotated before thus we can
                 * always use C style W(i,j,k,2).
                 */
                float_X tmp =
                    -currentSurfaceDensity * (
                        S0(line, i, 0) * S0(line, j, 1) +
                        float_X(0.5) * DS(line, i, 0) * S0(line, j, 1) +
                        float_X(0.5) * S0(line, i, 0) * DS(line, j, 1) +
                        (float_X(1.0) / float_X(3.0)) * DS(line, i, 0) * DS(line, j, 1)
                    );

                float_X accumulated_J = float_X(0.0);
                for (int k = begin ; k < end ; ++k)
                {
                    const float_X W = DS(line, k, 2) * tmp;
                    /* We multiply with `cellEdgeLength` due to the fact that the attribute for the
                     * in-cell particle `position` (and it's change in DELTA_T) is normalize to [0,1) */
                    accumulated_J += W;
                    /* the branch divergence here still over-compensates for the fewer collisions in the (expensive) atomic adds */
                    if (accumulated_J != float_X(0.0))
                        atomicAddWrapper(&((*cursorJ(i, j, k)).z()), accumulated_J);
                }
            }
        }
    }

    /** calculate S0 (see paper)
     * @param line element with previous and current position of the particle
     * @param gridPoint used grid point to evaluate assignment shape
     * @param d dimension range {0,1,2} means {x,y,z}
     *          different to Esirkepov paper, here we use C style
     */
    DINLINE float_X S0(const Line<float3_X>& line, const float_X gridPoint, const uint32_t d)
    {
        return ParticleAssign()(gridPoint - line.m_pos0[d]);
    }

    /** calculate DS (see paper)
     * @param line element with previous and current position of the particle
     * @param gridPoint used grid point to evaluate assignment shape
     * @param d dimension range {0,1,2} means {x,y,z}]
     *          different to Esirkepov paper, here we use C style
     */
    DINLINE float_X DS(const Line<float3_X>& line, const float_X gridPoint, const uint32_t d)
    {
        return ParticleAssign()(gridPoint - line.m_pos1[d]) - ParticleAssign()(gridPoint - line.m_pos0[d]);
    }

    static PMacc::traits::StringProperty getStringProperties()
    {
        PMacc::traits::StringProperty propList( "name", "Esirkepov" );
        return propList;
    }


    /** get normalized average in cell particle position
     *
     * @param x position of the particle
     * @param i shift of grid
     * @return in cell position
     */
    DINLINE float_X
    calc_InCellPos(const float_X x, const float_X i) const
    {
        return x - i;
    }
};

} //namespace currentSolver

} //namespace picongpu

#include "fields/currentDeposition/Esirkepov/Esirkepov2D.hpp"
