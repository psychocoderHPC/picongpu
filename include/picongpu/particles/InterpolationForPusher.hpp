/* Copyright 2015-2021 Richard Pausch
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

namespace picongpu
{
    /** functor for particle field interpolator
     *
     * This functor is a simplification of the full
     * field to particle interpolator that can be used in the
     * particle pusher
     */
    template<
        typename T_Field2PartInt,
        typename T_MemoryTypex,
        typename T_MemoryTypey,
        typename T_MemoryTypez,
        typename T_FieldPosition>
    struct InterpolationForPusher
    {
        using Field2PartInt = T_Field2PartInt;

        HDINLINE
        InterpolationForPusher(
            const T_MemoryTypex& memx,
            const T_MemoryTypey& memy,
            const T_MemoryTypez& memz,
            const T_FieldPosition& fieldPos)
            : m_memx(memx)
            , m_memy(memy)
            , m_memz(memz)
            , m_fieldPos(fieldPos)
        {
        }
#if 0
        /* apply shift policy before interpolation */
        template<typename T_PosType, typename T_ShiftPolicy>
        HDINLINE float3_X operator()(const T_PosType& pos, const T_ShiftPolicy& shiftPolicy) const
        {
            return Field2PartInt()(shiftPolicy.memory(m_mem, pos), shiftPolicy.position(pos), m_fieldPos);
        }
#endif
        /* interpolation using given memory and position */
        template<typename T_PosType>
        HDINLINE float3_X operator()(const T_PosType& pos) const
        {
            return float3_X(
                Field2PartInt()(m_memx, pos, m_fieldPos, 0),
                Field2PartInt()(m_memy, pos, m_fieldPos, 1),
                Field2PartInt()(m_memz, pos, m_fieldPos, 2));
        }


    private:
        PMACC_ALIGN(m_memx, T_MemoryTypex);
        PMACC_ALIGN(m_memy, T_MemoryTypey);
        PMACC_ALIGN(m_memz, T_MemoryTypez);
        PMACC_ALIGN(m_fieldPos, const T_FieldPosition);
    };


    /** functor to create particle field interpolator
     *
     * required to get interpolator for pusher
     */
    template<typename T_Field2PartInt>
    struct CreateInterpolationForPusher
    {
        template<typename T_MemoryTypex, typename T_MemoryTypey, typename T_MemoryTypez, typename T_FieldPosition>
        HDINLINE InterpolationForPusher<T_Field2PartInt, T_MemoryTypex, T_MemoryTypey, T_MemoryTypez, T_FieldPosition>
        operator()(
            const T_MemoryTypex& memx,
            const T_MemoryTypey& memy,
            const T_MemoryTypez& memz,
            const T_FieldPosition& fieldPos)
        {
            return InterpolationForPusher<
                T_Field2PartInt,
                T_MemoryTypex,
                T_MemoryTypey,
                T_MemoryTypez,
                T_FieldPosition>(memx, memy, memz, fieldPos);
        }
    };

} // namespace picongpu
