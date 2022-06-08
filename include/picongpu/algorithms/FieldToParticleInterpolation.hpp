/* Copyright 2013-2022 Axel Huebl, Heiko Burau, Rene Widera
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/algorithms/ShiftCoordinateSystem.hpp"
#include "picongpu/particles/shapes.hpp"

#include <pmacc/attribute/unroll.hpp>
#include <pmacc/cuSTL/algorithm/functor/GetComponent.hpp>
#include <pmacc/cuSTL/cursor/FunctorCursor.hpp>
#include <pmacc/math/Vector.hpp>

namespace picongpu
{
    /** interpolate field which are defined on a grid to a point inside of a grid
     *
     * interpolate around a point from -AssignmentFunction::support/2 to
     * (AssignmentFunction::support+1)/2
     *
     * @tparam GridShiftMethod functor which shift coordinate system that al value are
     * located on corner
     * @tparam AssignmentFunction AssignmentFunction which is used for interpolation
     * @tparam InterpolationMethod functor for interpolation method
     */
    template<class T_Shape, class InterpolationMethod>
    struct FieldToParticleInterpolation
    {
        using AssignmentFunction = typename T_Shape::ChargeAssignmentOnSupport;
        static constexpr int supp = AssignmentFunction::support;

        static constexpr int lowerMargin = supp / 2;
        static constexpr int upperMargin = (supp + 1) / 2;
        using LowerMargin = typename pmacc::math::CT::make_Int<simDim, lowerMargin>::type;
        using UpperMargin = typename pmacc::math::CT::make_Int<simDim, upperMargin>::type;

        PMACC_CASSERT_MSG(
            __FieldToParticleInterpolation_supercell_is_too_small_for_stencil,
            pmacc::math::CT::min<typename pmacc::math::CT::mul<SuperCellSize, GuardSize>::type>::type::value
                    >= lowerMargin
                && pmacc::math::CT::min<typename pmacc::math::CT::mul<SuperCellSize, GuardSize>::type>::type::value
                    >= upperMargin);

        /*(supp + 1) % 2 is 1 for even supports else 0*/
        static constexpr int begin = -supp / 2 + (supp + 1) % 2;
        static constexpr int end = begin + supp - 1;

        /** Get particle assignment shape functors.
         *
         * @param pos Position of the particle relative to the located cell. The position must be shifted to the
         *            assignment cell. The supported range of the position is defined by assignment function and
         *            depends on the particle support.
         * @return Three dimensional array with particle shape assignment functors.
         */
        HDINLINE auto getShapeFunctors(float3_X const& pos) const
        {
            pmacc::memory::Array<shapes::Cached<AssignmentFunction>, 3> result;
            result[0] = shapes::Cached<AssignmentFunction>(pos.x(), true);
            result[1] = shapes::Cached<AssignmentFunction>(pos.y(), true);
            result[2] = shapes::Cached<AssignmentFunction>(pos.z(), true);
            return result;
        };

        /** Get particle assignment shape functors.
         *
         * @param pos Position of the particle relative to the located cell. The position must be shifted to the
         *            assignment cell. The supported range of the position is defined by assignment function and
         *            depends on the particle support.
         * @return Two dimensional array with particle shape assignment functors.
         */
        HDINLINE auto getShapeFunctors(float2_X const& pos) const
        {
            pmacc::memory::Array<shapes::Cached<AssignmentFunction>, 2> result;
            result[0] = shapes::Cached<AssignmentFunction>(pos.x(), true);
            result[1] = shapes::Cached<AssignmentFunction>(pos.y(), true);
            return result;
        };

        template<class Cursor, class VecVector>
        HDINLINE typename Cursor::ValueType operator()(
            Cursor field,
            const floatD_X& particlePos,
            const VecVector& fieldPos)
        {
            /**\brief:
             * The following calls seperate the vector interpolation into
             * independent scalar interpolations.
             */
            using Supports = typename pmacc::math::CT::make_Int<simDim, supp>::type;

            using FunctorComponent = decltype(pmacc::cursor::make_FunctorCursor(
                field,
                pmacc::algorithm::functor::GetComponent<float_X>(0)));

            pmacc::memory::Array<FunctorComponent, 3> fieldComponent;

            using ShapeArray = decltype(getShapeFunctors(particlePos));
            pmacc::memory::Array<ShapeArray, 3> positionShifted;

            PMACC_UNROLL(Cursor::ValueType::dim)
            for(uint32_t i = 0; i < Cursor::ValueType::dim; i++)
            {
                fieldComponent[i]
                    = pmacc::cursor::make_FunctorCursor(field, pmacc::algorithm::functor::GetComponent<float_X>(i));

                floatD_X particlePosShifted = particlePos;
                ShiftCoordinateSystem<Supports>()(fieldComponent[i], particlePosShifted, fieldPos[i]);
                positionShifted[i] = getShapeFunctors(particlePosShifted);
            }

            return InterpolationMethod::template interpolate<begin, end>(fieldComponent, positionShifted);
        }

        static pmacc::traits::StringProperty getStringProperties()
        {
            GetStringProperties<InterpolationMethod> propList;
            return propList;
        }
    };

    namespace traits
    {
        /*Get margin of a solver
         * class must define a LowerMargin and UpperMargin
         */
        template<class AssignMethod, class InterpolationMethod>
        struct GetMargin<picongpu::FieldToParticleInterpolation<AssignMethod, InterpolationMethod>>
        {
        private:
            using Interpolation = picongpu::FieldToParticleInterpolation<AssignMethod, InterpolationMethod>;

        public:
            using LowerMargin = typename Interpolation::LowerMargin;
            using UpperMargin = typename Interpolation::UpperMargin;
        };

    } // namespace traits

} // namespace picongpu
