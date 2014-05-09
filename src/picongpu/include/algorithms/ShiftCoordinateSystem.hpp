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

#include "types.h"
#include "types.h"
#include "math/vector/Int.hpp"

namespace picongpu
{

template<uint32_t T_support>
struct ShiftCoordinateSystem
{

    /**shift to new coordinat system
     *
     * shift cursor and vector to new coordinate system
     * @param curser curser to memory
     * @param vector short vector with coordinates in old system
     * @param fieldPos vector with relative coordinates for shift ( value range [0.0;0.5] )
     */
    template<typename Cursor, typename Vector >
    HDINLINE void operator()(Cursor& cursor, Vector& vector, const floatD_X & fieldPos)
    {


        if (T_support % 2 == 0)
        {
            const floatD_X v_pos = vector - fieldPos;
            PMacc::math::Int < simDim > intShift;
            for (uint32_t i = 0; i < simDim; ++i)
                intShift[i] = math::float2int_rd(v_pos[i]);
            cursor = cursor(intShift);
            for (uint32_t i = 0; i < simDim; ++i)
                vector[i] = v_pos[i] - float_X(intShift[i]);
        }
        else
        {
            //odd support
            const floatD_X v_pos = vector - fieldPos;
            PMacc::math::Int < simDim > intShift;
            for (uint32_t i = 0; i < simDim; ++i)
                intShift[i] = int(v_pos[i] >= float_X(0.5));
            cursor = cursor(intShift);
            for (uint32_t i = 0; i < simDim; ++i)
                vector[i] = v_pos[i] - float_X(intShift[i]);
        }
    }
};

template<uint32_t T_support, uint32_t dim>
struct ShiftCoordinateSystemOne
{

    /**shift to new coordinat system
     *
     * shift cursor and vector to new coordinate system
     * @param curser curser to memory
     * @param vector short vector with coordinates in old system
     * @param fieldPos vector with relative coordinates for shift ( value range [0.0;0.5] )
     */
    template<typename Cursor, typename Vector >
    HDINLINE void operator()(Cursor& cursor, Vector& vector, const floatD_X & fieldPos)
    {


        if (T_support % 2 == 0)
        {
            const float_X v_pos = vector[dim] - fieldPos[dim];
            DataSpace<simDim> intShift(DataSpace<simDim>::create(0));
            intShift[dim] = math::float2int_rd(v_pos);
            cursor = cursor.shift(intShift);
            vector[dim] = v_pos - float_X(intShift[dim]);
        }
        else
        {
            //odd support
            const float_X v_pos = vector[dim] - fieldPos[dim];
            DataSpace<simDim> intShift(DataSpace<simDim>::create(0));
            intShift[dim] = int(v_pos >= float_X(0.5));
            cursor = cursor.shift(intShift);
            vector[dim] = v_pos - float_X(intShift[dim]);
        }
    }
};

} // namespace picongpu
