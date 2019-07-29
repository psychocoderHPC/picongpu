/* Copyright 2013-2019 Axel Huebl, Heiko Burau, Rene Widera
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

#include "picongpu/fields/MaxwellSolver/DirSplitting/DirSplitting.def"
#include "picongpu/simulation_defines.hpp"
#include "picongpu/fields/MaxwellSolver/DirSplitting/DirSplitting.kernel"
#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/fields/cellType/Centered.hpp"
#include "picongpu/fields/LaserPhysics.hpp"

#include <pmacc/cuSTL/algorithm/kernel/ForeachBlock.hpp>
#include <pmacc/cuSTL/cursor/NestedCursor.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/math/vector/Int.hpp>
#include <pmacc/math/vector/TwistComponents.hpp>
#include <pmacc/math/vector/compile-time/TwistComponents.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/cuSTL/cursor/navigator/compile-time/TwistAxesNavigator.hpp>
#include <pmacc/cuSTL/cursor/accessor/TwistAxesAccessor.hpp>


namespace picongpu
{
namespace fields
{
namespace maxwellSolver
{
namespace dirSplitting
{
    /** Check Directional Splitting grid and time conditions
     *
     * This is a workaround that the condition check is only
     * triggered if the current used solver is `DirSplitting`
     */
    template<typename T_UsedSolver, typename T_Dummy = void>
    struct ConditionCheck
    {
    };

    template<typename T_CurrentInterpolation, typename T_Dummy>
    struct ConditionCheck<DirSplitting< T_CurrentInterpolation >, T_Dummy>
    {
        /* Directional Splitting conditions:
         *
         * using SI units to avoid round off errors
         *
         * The compiler is allowed to evaluate an expression those not depends on a template parameter
         * even if the class is never instantiated. In that case static assert is always
         * evaluated (e.g. with clang), this results in an error if the condition is false.
         * http://www.boost.org/doc/libs/1_60_0/doc/html/boost_staticassert.html
         *
         * A workaround is to add a template dependency to the expression.
         * `sizeof(ANY_TYPE) != 0` is always true and defers the evaluation.
         */
        PMACC_CASSERT_MSG(DirectionSplitting_Set_dX_equal_dt_times_c____check_your_grid_param_file,
                          (SI::SPEED_OF_LIGHT_SI * SI::DELTA_T_SI) == SI::CELL_WIDTH_SI &&
                          (sizeof(T_Dummy) != 0));
        PMACC_CASSERT_MSG(DirectionSplitting_use_cubic_cells____check_your_grid_param_file,
                          SI::CELL_HEIGHT_SI == SI::CELL_WIDTH_SI &&
                          (sizeof(T_Dummy) != 0));
#if (SIMDIM == DIM3)
        PMACC_CASSERT_MSG(DirectionSplitting_use_cubic_cells____check_your_grid_param_file,
                          SI::CELL_DEPTH_SI == SI::CELL_WIDTH_SI &&
                          (sizeof(T_Dummy) != 0));
#endif
    };

    namespace result_of
    {
        template<typename T_SpaceTwist, typename T_ComponentTwist, typename T_Cursor>
        struct twistVectorForDirSplitting
        {
            typedef pmacc::cursor::Cursor<pmacc::cursor::TwistAxesAccessor<T_Cursor, T_ComponentTwist>,
                           pmacc::cursor::CT::TwistAxesNavigator<T_SpaceTwist>,
                           T_Cursor> type;
        };
    } // namespace result_of

    template<typename T_SpaceTwist, typename T_ComponentTwist, typename T_Cursor>
    HDINLINE
    typename result_of::twistVectorForDirSplitting<T_SpaceTwist, T_ComponentTwist, T_Cursor>::type
    twistVectorForDirSplitting(const T_Cursor& cursor)
    {
        return typename result_of::twistVectorForDirSplitting<T_SpaceTwist, T_ComponentTwist, T_Cursor>::type(
            pmacc::cursor::TwistAxesAccessor<T_Cursor, T_ComponentTwist>(),
            pmacc::cursor::CT::TwistAxesNavigator<T_SpaceTwist>(),
            cursor);
    }

} // namespace dirSplitting

    template< typename T_CurrentInterpolation >
    class DirSplitting: private dirSplitting::ConditionCheck< DirSplitting< T_CurrentInterpolation > >
    {
    private:
        template<uint32_t pass, typename SpaceTwist, typename OrientationTwist,typename JSpaceTwist,typename CursorE, typename CursorB, typename CursorJ, typename GridSize>
        void propagate(CursorE cursorE, CursorB cursorB,CursorJ cursorJ,CursorE old_cursorE, CursorB old_cursorB, GridSize gridSize) const
        {
            using namespace cursor::tools;
            using namespace pmacc::math;

            using SpaceTwistSimDim = typename CT::shrinkTo<SpaceTwist,simDim>::type;
            auto gridSizeTwisted = twistComponents<
                SpaceTwistSimDim
            >(gridSize);

            /* twist components of the supercell */
            typedef typename pmacc::math::CT::make_Vector<simDim,boost::mpl::integral_c<int,0> >::type ZeroVector;

            typedef typename pmacc::math::CT::AssignIfInRange<
                ZeroVector,
                boost::mpl::integral_c<int,0>,
                typename pmacc::math::CT::At<SuperCellSize,typename SpaceTwist::x>::type
            >::type VectorWith_X;

            typedef typename pmacc::math::CT::AssignIfInRange<
                VectorWith_X,
                boost::mpl::integral_c<int,1>,
                typename pmacc::math::CT::At<SuperCellSize,typename SpaceTwist::y>::type
            >::type VectorWith_XY;

            typedef typename pmacc::math::CT::AssignIfInRange<
                VectorWith_XY,
                boost::mpl::integral_c<int,2>,
                typename pmacc::math::CT::At<SuperCellSize,typename SpaceTwist::z>::type
            >::type BlockDim;


            pmacc::math::Size_t<simDim> zoneSize(gridSizeTwisted);
            zoneSize.x()=BlockDim::x::value;

            algorithm::kernel::ForeachBlock<BlockDim> foreach;
            foreach(zone::SphericZone<simDim>(zoneSize),
                    cursor::make_NestedCursor(dirSplitting::twistVectorForDirSplitting<SpaceTwistSimDim, OrientationTwist>(cursorE)),
                    cursor::make_NestedCursor(dirSplitting::twistVectorForDirSplitting<SpaceTwistSimDim, OrientationTwist>(cursorB)),
                    cursor::make_NestedCursor(dirSplitting::twistVectorForDirSplitting<SpaceTwistSimDim, OrientationTwist>(cursorJ)),
                    cursor::make_NestedCursor(dirSplitting::twistVectorForDirSplitting<SpaceTwistSimDim, OrientationTwist>(old_cursorE)),
                    cursor::make_NestedCursor(dirSplitting::twistVectorForDirSplitting<SpaceTwistSimDim, OrientationTwist>(old_cursorB)),
                    DirSplittingKernel<pass,BlockDim,JSpaceTwist>((int)gridSizeTwisted.x()));
        }

    public:

        using CellType = cellType::Centered;
        using CurrentInterpolation = T_CurrentInterpolation;

        DirSplitting(MappingDesc) {}

        void update_afterCurrent(uint32_t currentStep) const
        {
            typedef SuperCellSize GuardDim;

            DataConnector &dc = Environment<>::get().DataConnector();

            auto fieldE = dc.get< FieldE >( FieldE::getName(), true );
            auto fieldB = dc.get< FieldB >( FieldB::getName(), true );
            auto fieldJ = dc.get< FieldJ >( FieldJ::getName(), true );

            using namespace cursor::tools;



            if (laserProfiles::Selected::INIT_TIME > float_X(0.0))
                LaserPhysics{}(currentStep);

            __setTransactionEvent(fieldE->asyncCommunication(__getTransactionEvent()));
            __setTransactionEvent(fieldB->asyncCommunication(__getTransactionEvent()));

            auto fieldE_coreBorder =
                fieldE->getGridBuffer().getDeviceBuffer().
                       cartBuffer().view(GuardDim().toRT(),
                                         -GuardDim().toRT());
            auto fieldB_coreBorder =
                fieldB->getGridBuffer().getDeviceBuffer().
                cartBuffer().view(GuardDim().toRT(),
                                  -GuardDim().toRT());

            auto old_fieldE_coreBorder =
                fieldE->getGridBuffer2().getDeviceBuffer().
                       cartBuffer().view(GuardDim().toRT(),
                                         -GuardDim().toRT());
            auto old_fieldB_coreBorder =
                fieldB->getGridBuffer2().getDeviceBuffer().
                cartBuffer().view(GuardDim().toRT(),
                                  -GuardDim().toRT());

            auto fieldJ_coreBorder =
                fieldJ->getGridBuffer().getDeviceBuffer().
                cartBuffer().view(GuardDim().toRT(),
                                  -GuardDim().toRT());

            typedef pmacc::math::CT::Int<0,1,2> Orientation_XJ;
            typedef pmacc::math::CT::Int<0,1,2> Space_XJ;

            pmacc::math::Size_t<simDim> gridSize = fieldE_coreBorder.size();

            typedef pmacc::math::CT::Int<0,1,2> Orientation_X;
            typedef pmacc::math::CT::Int<0,1,2> Space_X;
            typedef pmacc::math::CT::Int<0,2,1> JDir_X;
            propagate<1, Space_X,Orientation_X,JDir_X>(
                      fieldE_coreBorder.origin(),
                      fieldB_coreBorder.origin(),
                      fieldJ_coreBorder.origin(),
                      old_fieldE_coreBorder.origin(),
                      old_fieldB_coreBorder.origin(),
                      gridSize);

            __setTransactionEvent(fieldE->asyncCommunication(__getTransactionEvent()));
            __setTransactionEvent(fieldB->asyncCommunication(__getTransactionEvent()));

            typedef pmacc::math::CT::Int<1,2,0> Orientation_Y;
            typedef pmacc::math::CT::Int<1,0,2> Space_Y;
            typedef pmacc::math::CT::Int<0,1,2> JDir_Y;
            propagate<1, Space_Y,Orientation_Y,JDir_Y>(
                      fieldE_coreBorder.origin(),
                      fieldB_coreBorder.origin(),
                      fieldJ_coreBorder.origin(),
                      old_fieldE_coreBorder.origin(),
                      old_fieldB_coreBorder.origin(),
                      gridSize);

            __setTransactionEvent(fieldE->asyncCommunication(__getTransactionEvent()));
            __setTransactionEvent(fieldB->asyncCommunication(__getTransactionEvent()));


    #if (SIMDIM==DIM3)
            //! \todo: currently 3D: check this code if someone enable 3D
            typedef pmacc::math::CT::Int<2,0,1> Orientation_Z;
            typedef pmacc::math::CT::Int<2,0,1> Space_Z;
            typedef pmacc::math::CT::Int<0,2,1> JDir_Z;
            propagate<0,Space_Z,Orientation_Z,JDir_Z>(
                      fieldE_coreBorder.origin(),
                      fieldB_coreBorder.origin(),
                      fieldJ_coreBorder.origin(),
                      old_fieldE_coreBorder.origin(),
                      old_fieldB_coreBorder.origin(),
                      gridSize);

            __setTransactionEvent(fieldE->asyncCommunication(__getTransactionEvent()));
            __setTransactionEvent(fieldB->asyncCommunication(__getTransactionEvent()));
    #endif

            propagate<1, Space_Y,Orientation_Y,JDir_Y>(
                    fieldE_coreBorder.origin(),
                    fieldB_coreBorder.origin(),
                    fieldJ_coreBorder.origin(),
                    old_fieldE_coreBorder.origin(),
                    old_fieldB_coreBorder.origin(),
                    gridSize);

            __setTransactionEvent(fieldE->asyncCommunication(__getTransactionEvent()));
            __setTransactionEvent(fieldB->asyncCommunication(__getTransactionEvent()));

            propagate<1, Space_X,Orientation_X,JDir_X>(
                    fieldE_coreBorder.origin(),
                    fieldB_coreBorder.origin(),
                    fieldJ_coreBorder.origin(),
                    old_fieldE_coreBorder.origin(),
                    old_fieldB_coreBorder.origin(),
                    gridSize);

            __setTransactionEvent(fieldE->asyncCommunication(__getTransactionEvent()));
            __setTransactionEvent(fieldB->asyncCommunication(__getTransactionEvent()));
        }

        void update_beforeCurrent(uint32_t) const
        {

        }

        static pmacc::traits::StringProperty getStringProperties()
        {
            pmacc::traits::StringProperty propList( "name", "DS" );
            return propList;
        }
    };


} // namespace maxwellSolver
} // namespace fields
} // namespace picongpu
