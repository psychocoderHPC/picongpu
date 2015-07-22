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

#include "DirSplitting.def"

#include "simulation_defines.hpp"

#include <fields/MaxwellSolver/DirSplitting/DirSplitting.kernel>
#include <math/vector/Int.hpp>
#include <dataManagement/DataConnector.hpp>
#include <fields/FieldB.hpp>
#include <fields/FieldE.hpp>
#include "math/Vector.hpp"
#include <cuSTL/algorithm/kernel/ForeachBlock.hpp>
#include <lambda/Expression.hpp>
#include <cuSTL/cursor/NestedCursor.hpp>

namespace picongpu
{
namespace dirSplitting
{
using namespace PMacc;

/** Check Directional Splitting grid and time conditions
 *
 * This is a workaround that the condition check is only
 * triggered if the current used solver is `DirSplitting`
 */
template<typename T_UsedSolver, typename T_Dummy = void>
struct ConditionCheck
{
};

template<typename T_Dummy>
struct ConditionCheck<DirSplitting, T_Dummy>
{
    /* Directional Splitting conditions:
     *
     * using SI units to avoid round off errors
     */
    PMACC_CASSERT_MSG(DirectionSplitting_Set_dX_equal_dt_times_c____check_your_gridConfig_param_file,
                      (SI::SPEED_OF_LIGHT_SI * SI::DELTA_T_SI) == SI::CELL_WIDTH_SI);
    PMACC_CASSERT_MSG(DirectionSplitting_use_cubic_cells____check_your_gridConfig_param_file,
                      SI::CELL_HEIGHT_SI == SI::CELL_WIDTH_SI);
#if (SIMDIM == DIM3)
    PMACC_CASSERT_MSG(DirectionSplitting_use_cubic_cells____check_your_gridConfig_param_file,
                      SI::CELL_DEPTH_SI == SI::CELL_WIDTH_SI);
#endif
};

class DirSplitting : private ConditionCheck<fieldSolver::FieldSolver>
{
private:

    template<typename OrientationTwist, uint32_t T_pase, typename CursorE, typename CursorB, typename CursorJ, typename GridSize>
    void propagate(CursorE cursorE, CursorB cursorB, CursorJ cursorJ, GridSize gridSize) const
    {
        using namespace cursor::tools;
        using namespace PMacc::math::tools;

        PMACC_AUTO(gridSizeTwisted, twistVectorAxes<OrientationTwist>(gridSize));

        /* twist components of the supercell */
        typedef PMacc::math::CT::Int <
            PMacc::math::CT::At<SuperCellSize, typename OrientationTwist::x>::type::value,
            PMacc::math::CT::At<SuperCellSize, typename OrientationTwist::y>::type::value,
            PMacc::math::CT::At<SuperCellSize, typename OrientationTwist::z>::type::value
            > BlockDim;

        algorithm::kernel::ForeachBlock<BlockDim> foreach;
        foreach(zone::SphericZone<3>(PMacc::math::Size_t<3>(BlockDim::x::value, gridSizeTwisted.y(), gridSizeTwisted.z())),
                cursor::make_NestedCursor(twistVectorFieldAxes<OrientationTwist>(cursorE)),
                cursor::make_NestedCursor(twistVectorFieldAxes<OrientationTwist>(cursorB)),
                cursor::make_NestedCursor(twistVectorFieldAxes<OrientationTwist>(cursorJ)),
                DirSplittingKernel<BlockDim, T_pase>((int) gridSizeTwisted.x()));
    }
public:

    DirSplitting(MappingDesc)
    {
    }

    void update_beforeCurrent(uint32_t currentStep) const
    {

        DataConnector &dc = Environment<>::get().DataConnector();

        FieldE& fieldE = dc.getData<FieldE > (FieldE::getName(), true);
        FieldB& fieldB = dc.getData<FieldB > (FieldB::getName(), true);
        FieldJ& fieldJ = dc.getData<FieldJ > (FieldJ::getName(), true);


        if (laserProfile::INIT_TIME > float_X(0.0))
            fieldE.laserManipulation(currentStep);

        typedef SuperCellSize GuardDim;

        BOOST_AUTO(fieldE_coreBorder,
                   fieldE.getGridBuffer().getDeviceBuffer().
                   cartBuffer().view(GuardDim().toRT(),
                                     -GuardDim().toRT()));
        BOOST_AUTO(fieldB_coreBorder,
                   fieldB.getGridBuffer().getDeviceBuffer().
                   cartBuffer().view(GuardDim().toRT(),
                                     -GuardDim().toRT()));

        BOOST_AUTO(fieldJ_coreBorder,
                   fieldJ.getGridBuffer().getDeviceBuffer().
                   cartBuffer().view(GuardDim().toRT(),
                                     -GuardDim().toRT()));

        using namespace cursor::tools;
        using namespace PMacc::math::tools;

        PMacc::math::Size_t<3> gridSize = fieldE_coreBorder.size();


        typedef PMacc::math::CT::Int<0, 1, 2> Orientation_X;
        typedef PMacc::math::CT::Int<1, 2, 0> Orientation_Y;
        typedef PMacc::math::CT::Int<2, 0, 1> Orientation_Z;


        EventTask eRfieldE;
        EventTask eRfieldB;
        propagate<Orientation_X, 0>(
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldJ_coreBorder.origin(),
                                    gridSize);

        propagate<Orientation_Y, 0>(
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldJ_coreBorder.origin(),
                                    gridSize);
        propagate<Orientation_Z, 0>(
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldJ_coreBorder.origin(),
                                    gridSize);

        eRfieldE = fieldE.asyncCommunication(__getTransactionEvent());
        eRfieldB = fieldB.asyncCommunication(__getTransactionEvent());
        __setTransactionEvent(eRfieldE);
        __setTransactionEvent(eRfieldB);

        propagate<Orientation_X, 1>(
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldJ_coreBorder.origin(),
                                    gridSize);

        propagate<Orientation_Y, 1>(
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldJ_coreBorder.origin(),
                                    gridSize);
        propagate<Orientation_Z, 1>(
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldJ_coreBorder.origin(),
                                    gridSize);

        propagate<Orientation_X, 2>(
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldJ_coreBorder.origin(),
                                    gridSize);

        /*   propagate<Orientation_X, 1>(
                                       fieldE_coreBorder.origin(),
                                       fieldB_coreBorder.origin(),
                                       fieldJ_coreBorder.origin(),
                                       gridSize);

           propagate<Orientation_X, 2>(
                                       fieldE_coreBorder.origin(),
                                       fieldB_coreBorder.origin(),
                                       fieldJ_coreBorder.origin(),
                                       gridSize);
           propagate<Orientation_X, 3>(
                                       fieldE_coreBorder.origin(),
                                       fieldB_coreBorder.origin(),
                                       fieldJ_coreBorder.origin(),
                                       gridSize);

           eRfieldE = fieldE.asyncCommunication(__getTransactionEvent());
           eRfieldB = fieldB.asyncCommunication(__getTransactionEvent());
           __setTransactionEvent(eRfieldE);
           __setTransactionEvent(eRfieldB);

           propagate<Orientation_Y, 0>(
                                       fieldE_coreBorder.origin(),
                                       fieldB_coreBorder.origin(),
                                       fieldJ_coreBorder.origin(),
                                       gridSize);

           propagate<Orientation_Y, 1>(
                                       fieldE_coreBorder.origin(),
                                       fieldB_coreBorder.origin(),
                                       fieldJ_coreBorder.origin(),
                                       gridSize);

           propagate<Orientation_Y, 2>(
                                       fieldE_coreBorder.origin(),
                                       fieldB_coreBorder.origin(),
                                       fieldJ_coreBorder.origin(),
                                       gridSize);
           propagate<Orientation_Y, 3>(
                                       fieldE_coreBorder.origin(),
                                       fieldB_coreBorder.origin(),
                                       fieldJ_coreBorder.origin(),
                                       gridSize);

           eRfieldE = fieldE.asyncCommunication(__getTransactionEvent());
           eRfieldB = fieldB.asyncCommunication(__getTransactionEvent());
           __setTransactionEvent(eRfieldE);
           __setTransactionEvent(eRfieldB);

           propagate<Orientation_Z, 0>(
                                       fieldE_coreBorder.origin(),
                                       fieldB_coreBorder.origin(),
                                       fieldJ_coreBorder.origin(),
                                       gridSize);

           propagate<Orientation_Z, 1>(
                                       fieldE_coreBorder.origin(),
                                       fieldB_coreBorder.origin(),
                                       fieldJ_coreBorder.origin(),
                                       gridSize);

           propagate<Orientation_Z, 2>(
                                       fieldE_coreBorder.origin(),
                                       fieldB_coreBorder.origin(),
                                       fieldJ_coreBorder.origin(),
                                       gridSize);
           propagate<Orientation_Z, 3>(
                                       fieldE_coreBorder.origin(),
                                       fieldB_coreBorder.origin(),
                                       fieldJ_coreBorder.origin(),
                                       gridSize);
         * */

        //#### J

        /*  propagate<Orientation_X, 2>(
                                      fieldE_coreBorder.origin(),
                                      fieldB_coreBorder.origin(),
                                      fieldJ_coreBorder.origin(),
                                      gridSize);

          propagate<Orientation_Y, 2>(
                                      fieldE_coreBorder.origin(),
                                      fieldB_coreBorder.origin(),
                                      fieldJ_coreBorder.origin(),
                                      gridSize);

          propagate<Orientation_Z, 2>(
                                      fieldE_coreBorder.origin(),
                                      fieldB_coreBorder.origin(),
                                      fieldJ_coreBorder.origin(),
                                      gridSize);
         */

        /*   eRfieldE = fieldE.asyncCommunication(__getTransactionEvent());
           eRfieldB = fieldB.asyncCommunication(__getTransactionEvent());
          __setTransactionEvent(eRfieldE);
          __setTransactionEvent(eRfieldB);

          propagate<Orientation_X, 1>(
                                      fieldE_coreBorder.origin(),
                                      fieldB_coreBorder.origin(),
                                      fieldJ_coreBorder.origin(),
                                      gridSize);

          propagate<Orientation_Y, 1>(
                                      fieldE_coreBorder.origin(),
                                      fieldB_coreBorder.origin(),
                                      fieldJ_coreBorder.origin(),
                                      gridSize);

          propagate<Orientation_Z, 1>(
                                      fieldE_coreBorder.origin(),
                                      fieldB_coreBorder.origin(),
                                      fieldJ_coreBorder.origin(),
                                      gridSize);
         */

        //#### J

        /* propagate<Orientation_X, 3>(
                                     fieldE_coreBorder.origin(),
                                     fieldB_coreBorder.origin(),
                                     fieldJ_coreBorder.origin(),
                                     gridSize);

         propagate<Orientation_Y, 3>(
                                     fieldE_coreBorder.origin(),
                                     fieldB_coreBorder.origin(),
                                     fieldJ_coreBorder.origin(),
                                     gridSize);

         propagate<Orientation_Z, 3>(
                                     fieldE_coreBorder.origin(),
                                     fieldB_coreBorder.origin(),
                                     fieldJ_coreBorder.origin(),
                                     gridSize);
         */


        eRfieldE = fieldE.asyncCommunication(__getTransactionEvent());
        eRfieldB = fieldB.asyncCommunication(__getTransactionEvent());
        __setTransactionEvent(eRfieldE);
        __setTransactionEvent(eRfieldB);

        dc.releaseData(FieldE::getName());
        dc.releaseData(FieldB::getName());
        dc.releaseData(FieldJ::getName());

    }

    void update_afterCurrent(uint32_t currentStep) const
    {
        DataConnector &dc = Environment<>::get().DataConnector();

        FieldE& fieldE = dc.getData<FieldE > (FieldE::getName(), true);
        FieldB& fieldB = dc.getData<FieldB > (FieldB::getName(), true);

        EventTask eRfieldE = fieldE.asyncCommunication(__getTransactionEvent());
        EventTask eRfieldB = fieldB.asyncCommunication(__getTransactionEvent());
        __setTransactionEvent(eRfieldE);
        __setTransactionEvent(eRfieldB);

        dc.releaseData(FieldE::getName());
        dc.releaseData(FieldB::getName());
    }
};

} // dirSplitting

} // picongpu
