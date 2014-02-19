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
#include <cuSTL/cursor/FunctorCursor.hpp>
#include "math/vector/compile-time/Int.hpp"

namespace picongpu
{

/** interpolate field which are defined on a grid to a point inside of a grid
 * 
 * interpolate around of a point from -AssignmentFunction::support/2 to 
 * (AssignmentFunction::support+1)/2
 * 
 * \tparam GridShiftMethod functor which shift coordinate system that al value are
 * located on corner
 * \tparam AssignmentFunction AssignmentFunction which is used for interpolation
 * \tparam InterpolationMethod functor for interpolation method
 */
template<class Assignment, class InterpolationMethod>
struct FieldToParticleInterpolation
{
    typedef FieldToParticleInterpolation<Assignment,InterpolationMethod> ThisType;
    typedef typename Assignment::ChargeAssignment AssignmentFunction;

    static const int supp = AssignmentFunction::support;

    static const int lowerMargin = supp / 2;
    static const int upperMargin = (supp + 1) / 2;
    typedef PMacc::math::CT::Int<lowerMargin, lowerMargin, lowerMargin> LowerMargin;
    typedef PMacc::math::CT::Int<upperMargin, upperMargin, upperMargin> UpperMargin;

    /*(supp + 1) % 2 is 1 for even supports else 0*/
    static const int begin = -supp / 2 + (supp + 1) % 2;
    static const int end = supp / 2;

    template<class Cursor, class VecVector_ >
    HDINLINE float3_X operator()(Cursor field, const float3_X& particlePos,
                                 const VecVector_ & fieldPos)
    {
        using namespace lambda;
        DECLARE_PLACEHOLDERS() // declares _1, _2, _3, ... in device code

        /**\brief:
         * The following three calls seperate the vector interpolation into three
         * independent scalar interpolations. In each call the coordinate system
         * is turned so that E_scalar does the interpolation for the z-component.
         */

        /** _1[mpl::int_<0>()] means: 
         * Create a functor which returns [0] applied on the first paramter.
         * Here it is: return the x-component of the field-vector.
         * _1[mpl::int_<0>()] is equivalent to _1[0] but has no runtime cost.
         */

        BOOST_AUTO(field_x, PMacc::cursor::make_FunctorCursor(field, _1[mpl::int_ < 0 > ()]));
        float3_X pos_tmp(particlePos);
        shift(field_x, pos_tmp, fieldPos.x());
        float_X result_x = InterpolationMethod::template interpolate<AssignmentFunction, begin, end > (field_x, pos_tmp);

        BOOST_AUTO(field_y, PMacc::cursor::make_FunctorCursor(field, _1[mpl::int_ < 1 > ()]));
        pos_tmp = particlePos;
        shift(field_y, pos_tmp, fieldPos.y());
        float_X result_y = InterpolationMethod::template interpolate<AssignmentFunction, begin, end > (field_y, pos_tmp);

        BOOST_AUTO(field_z, PMacc::cursor::make_FunctorCursor(field, _1[mpl::int_ < 2 > ()]));
        pos_tmp = particlePos;
        shift(field_z, pos_tmp, fieldPos.z());
        float_X result_z = InterpolationMethod::template interpolate<AssignmentFunction, begin, end > (field_z, pos_tmp);

        return float3_X(result_x, result_y, result_z);
    }

private:

    /**shift to new coordinat system
     * 
     * shift cursor and vector to new coordinate system
     * @param curser curser to memory
     * @param vector short vector with coordinates in old system
     * @param fieldPos vector with relative coordinates for shift ( value range [0.0;0.5] )
     */
    template<typename Cursor, typename Vector >
    HDINLINE void shift(Cursor& cursor, Vector& vector, const float3_X & fieldPos)
    {

        if (supp % 2 == 0)
        {
            //even support

            /* for any direction
             * if fieldPos == 0.5 and vector<0.5 
             * shift curser+(-1) and new_vector=old_vector-(-1)
             * 
             * (vector.x() < fieldPos.x()) is equal ((fieldPos == 0.5) && (vector<0.5))
             */
            float3_X coordinate_shift(
                                      -float_X(vector.x() < fieldPos.x()),
                                      -float_X(vector.y() < fieldPos.y()),
                                      -float_X(vector.z() < fieldPos.z())
                                      );
            cursor = cursor(
                            PMacc::math::Int < 3 > (
                                                    coordinate_shift.x(),
                                                    coordinate_shift.y(),
                                                    coordinate_shift.z()
                                                    ));
            //same as: vector = vector - fieldPos - coordinate_shift;
            vector -= (fieldPos + coordinate_shift);
        }
        else
        {
            //odd support

            /* for any direction
             * if fieldPos < 0.5 and vector> 0.5 
             * shift curser+1 and new_vector=old_vector-1
             */
            float3_X coordinate_shift(
                                      float_X(vector.x() > float_X(0.5) && fieldPos.x() != float_X(0.5)),
                                      float_X(vector.y() > float_X(0.5) && fieldPos.y() != float_X(0.5)),
                                      float_X(vector.z() > float_X(0.5) && fieldPos.z() != float_X(0.5))
                                      );
            cursor = cursor(
                            PMacc::math::Int < 3 > (
                                                    coordinate_shift.x(),
                                                    coordinate_shift.y(),
                                                    coordinate_shift.z()
                                                    ));
            //same as: vector = vector - fieldPos - coordinate_shift;
            vector -= (fieldPos + coordinate_shift);

        }
    }
};

namespace traits
{

/*Get margin of a solver
 * class must define a LowerMargin and UpperMargin 
 */
template<class AssignMethod, class InterpolationMethod>
struct GetMargin<picongpu::FieldToParticleInterpolation<AssignMethod, InterpolationMethod> >
{
private:
    typedef picongpu::FieldToParticleInterpolation<AssignMethod, InterpolationMethod> Interpolation;
public:
    typedef typename Interpolation::LowerMargin LowerMargin;
    typedef typename Interpolation::UpperMargin UpperMargin;
};

} //namespace traits

} //namespace picongpu


