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
#include "fields/currentInterpolation/CurrentInterpolation.hpp"

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

    template<uint32_t pass, typename OrientationTwist, typename CursorE, typename CursorB, typename CursorE2, typename CursorB2, typename CursorJ, typename GridSize>
    void propagate(CursorE cursorE, CursorB cursorB, CursorE2 cursorE2, CursorB2 cursorB2, CursorJ cursorJ, GridSize gridSize) const
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
                cursor::make_NestedCursor(twistVectorFieldAxes<OrientationTwist>(cursorE2)),
                cursor::make_NestedCursor(twistVectorFieldAxes<OrientationTwist>(cursorB2)),
                cursor::make_NestedCursor(twistVectorFieldAxes<OrientationTwist>(cursorJ)),
                DirSplittingKernel<pass, BlockDim>((int) gridSizeTwisted.x()));
    }
public:

    DirSplitting(MappingDesc)
    {
    }

    void update_beforeCurrent(uint32_t currentStep) const
    {


    }

    void update_afterCurrent(uint32_t currentStep) const
    {
        DataConnector &dc = Environment<>::get().DataConnector();

        FieldE& fieldE = dc.getData<FieldE > (FieldE::getName(), true);
        FieldB& fieldB = dc.getData<FieldB > (FieldB::getName(), true);
        FieldJ& fieldJ = dc.getData<FieldJ > (FieldJ::getName(), true);

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

        BOOST_AUTO(fieldE_coreBorder2,
                   fieldE.getGridBuffer2().getDeviceBuffer().
                   cartBuffer().view(GuardDim().toRT(),
                                     -GuardDim().toRT()));
        BOOST_AUTO(fieldB_coreBorder2,
                   fieldB.getGridBuffer2().getDeviceBuffer().
                   cartBuffer().view(GuardDim().toRT(),
                                     -GuardDim().toRT()));

        using namespace cursor::tools;
        using namespace PMacc::math::tools;

        PMacc::math::Size_t<3> gridSize = fieldE_coreBorder.size();





        __startOperation(ITask::TASK_HOST);
        __startOperation(ITask::TASK_CUDA);
        fieldE.sync();
        fieldB.sync();
        //fieldE.reset(currentStep);
        //fieldB.reset(currentStep);
        __setTransactionEvent(fieldE.asyncCommunication(__getTransactionEvent()));
        __setTransactionEvent(fieldB.asyncCommunication(__getTransactionEvent()));



        typedef PMacc::math::CT::Int<0, 1, 2> Orientation_X;
        typedef PMacc::math::CT::Int<1, 2, 0> Orientation_Y;
        typedef PMacc::math::CT::Int<2, 0, 1> Orientation_Z;


        propagate<0, Orientation_X>(
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldJ_coreBorder.origin(),
                                    gridSize);

        propagate<0, Orientation_Y>(
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldJ_coreBorder.origin(),
                                    gridSize);

        propagate<0, Orientation_Z>(
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldJ_coreBorder.origin(),
                                    gridSize);

        /*       propagate<1, Orientation_X>(
                                           fieldE_coreBorder2.origin(),
                                           fieldB_coreBorder2.origin(),
                                           fieldE_coreBorder2.origin(),
                                           fieldB_coreBorder2.origin(),
                                           fieldJ_coreBorder.origin(),
                                           gridSize);

               propagate<1, Orientation_Y>(
                                           fieldE_coreBorder2.origin(),
                                           fieldB_coreBorder2.origin(),
                                           fieldE_coreBorder2.origin(),
                                           fieldB_coreBorder2.origin(),
                                           fieldJ_coreBorder.origin(),
                                           gridSize);

               propagate<1, Orientation_Z>(
                                           fieldE_coreBorder2.origin(),
                                           fieldB_coreBorder2.origin(),
                                           fieldE_coreBorder2.origin(),
                                           fieldB_coreBorder2.origin(),
                                           fieldJ_coreBorder.origin(),
                                           gridSize);

         */
        __startOperation(ITask::TASK_HOST);
        __startOperation(ITask::TASK_CUDA);

        __setTransactionEvent(fieldE.asyncCommunication(__getTransactionEvent()));
        __setTransactionEvent(fieldB.asyncCommunication(__getTransactionEvent()));

        propagate<1, Orientation_X>(
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldJ_coreBorder.origin(),
                                    gridSize);

        propagate<1, Orientation_Y>(
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldJ_coreBorder.origin(),
                                    gridSize);

        propagate<1, Orientation_Z>(
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldJ_coreBorder.origin(),
                                    gridSize);


        propagate<2, Orientation_X>(
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldJ_coreBorder.origin(),
                                    gridSize);

        propagate<2, Orientation_Y>(
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldJ_coreBorder.origin(),
                                    gridSize);

        propagate<2, Orientation_Z>(
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldJ_coreBorder.origin(),
                                    gridSize);

        propagate<3, Orientation_X>(
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldJ_coreBorder.origin(),
                                    gridSize);

        propagate<3, Orientation_Y>(
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldJ_coreBorder.origin(),
                                    gridSize);

        propagate<3, Orientation_Z>(
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldE_coreBorder.origin(),
                                    fieldB_coreBorder.origin(),
                                    fieldJ_coreBorder.origin(),
                                    gridSize);
        /*

                propagate<2, Orientation_X>(
                                            fieldE_coreBorder2.origin(),
                                            fieldB_coreBorder2.origin(),
                                            fieldE_coreBorder.origin(),
                                            fieldB_coreBorder.origin(),
                                            fieldJ_coreBorder.origin(),
                                            gridSize);

                propagate<2, Orientation_Y>(
                                            fieldE_coreBorder2.origin(),
                                            fieldB_coreBorder2.origin(),
                                            fieldE_coreBorder.origin(),
                                            fieldB_coreBorder.origin(),
                                            fieldJ_coreBorder.origin(),
                                            gridSize);

                propagate<2, Orientation_Z>(
                                            fieldE_coreBorder2.origin(),
                                            fieldB_coreBorder2.origin(),
                                            fieldE_coreBorder.origin(),
                                            fieldB_coreBorder.origin(),
                                            fieldJ_coreBorder.origin(),
                                            gridSize);

         */
        /*
                propagate<1, Orientation_Y>(
                                            fieldE_coreBorder2.origin(),
                                            fieldB_coreBorder2.origin(),
                                            fieldE_coreBorder.origin(),
                                            fieldB_coreBorder.origin(),
                                            fieldJ_coreBorder.origin(),
                                            gridSize);

         */
        /*
                __startOperation(ITask::TASK_HOST);
                __startOperation(ITask::TASK_CUDA);
                fieldE.sync();
                fieldB.sync();
                __setTransactionEvent(fieldE.asyncCommunication(__getTransactionEvent()));
                __setTransactionEvent(fieldB.asyncCommunication(__getTransactionEvent()));

         */

        /*
         *   propagate<0, Orientation_Y>(
                                      fieldE_coreBorder2.origin(),
                                      fieldB_coreBorder2.origin(),
                                      fieldE_coreBorder.origin(),
                                      fieldB_coreBorder.origin(),
                                      fieldJ_coreBorder.origin(),
                                      gridSize);
         * */


        /*
                __startOperation(ITask::TASK_HOST);
                __startOperation(ITask::TASK_CUDA);
                fieldE.sync();
                fieldB.sync();
                __setTransactionEvent(fieldE.asyncCommunication(__getTransactionEvent()));
                __setTransactionEvent(fieldB.asyncCommunication(__getTransactionEvent()));
         */

        /*
          propagate<0, Orientation_Z>(
                                      fieldE_coreBorder2.origin(),
                                      fieldB_coreBorder2.origin(),
                                      fieldE_coreBorder.origin(),
                                      fieldB_coreBorder.origin(),
                                      fieldJ_coreBorder.origin(),
                                      gridSize);
         */
        /*

                __startOperation(ITask::TASK_HOST);
                __startOperation(ITask::TASK_CUDA);
                fieldE.sync();
                fieldB.sync();
                __setTransactionEvent(fieldE.asyncCommunication(__getTransactionEvent()));
                __setTransactionEvent(fieldB.asyncCommunication(__getTransactionEvent()));


         */


        /*
                propagate<1, Orientation_Z>(
                                            fieldE_coreBorder2.origin(),
                                            fieldB_coreBorder2.origin(),
                                            fieldE_coreBorder.origin(),
                                            fieldB_coreBorder.origin(),
                                            fieldJ_coreBorder.origin(),
                                            gridSize);
         * */

        //if (currentStep == 0)
        {
            // fieldJ.addCurrentToEMF < CORE + BORDER > (currentInterpolation::NoneDS<simDim, 1>());
            //   fieldJ.addCurrentToEMF < CORE + BORDER > (currentInterpolation::NoneDS<simDim, 0>());
            //  fieldJ.addCurrentToEMF < CORE + BORDER > (currentInterpolation::NoneDS<simDim, 2>());
        }


        /* algorithm::kernel::ForeachBlock<SuperCellSize> foreachHalf;
         foreachHalf(zone::SphericZone<3>(PMacc::math::Size_t<3>(gridSize)),
                     cursor::make_NestedCursor(fieldE_coreBorder.origin()),
                     cursor::make_NestedCursor(fieldB_coreBorder.origin()),
                     HalfKernel<SuperCellSize>());
         */

        if (laserProfile::INIT_TIME > float_X(0.0))
            dc.getData<FieldE > (FieldE::getName(), true).laserManipulation(currentStep);

        __startOperation(ITask::TASK_HOST);
        __startOperation(ITask::TASK_CUDA);
        fieldE.sync();
        fieldB.sync();
        __setTransactionEvent(fieldE.asyncCommunication(__getTransactionEvent()));
        __setTransactionEvent(fieldB.asyncCommunication(__getTransactionEvent()));

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
