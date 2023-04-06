/* Copyright 2013-2023 Axel Huebl, Rene Widera, Sergei Bastrakov
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

#include "picongpu/fields/boundary/IBoundary.hpp"
#include "picongpu/fields/boundary/impl/Exponential.hpp"

#include <cstdint>
#include <string>


namespace picongpu::fields::boundary
{
    /** Exponential damping field absorber
     */
    class Exponential : IBoundary
    {
        /** Thickness of the absorbing layer, in number of cells
         *
         * This setting applies to applies for all absorber kinds.
         * The absorber layer is located inside the global simulation area, near the outer borders.
         * Setting size to 0 results in disabling absorption at the corresponding boundary.
         * Note that for non-absorbing boundaries the actual thickness will be 0 anyways.
         * There are no requirements on thickness being a multiple of the supercell size.
         *
         * For PML the recommended thickness is between 6 and 16 cells.
         * For the exponential damping it is 32.
         *
         * Unit: number of cells.
         */
        vec3<vec2<uint32_t>> m_numCells;
        /** Define the strength of the absorber for all directions
         *
         * Elements corredponding to non-absorber borders will have no effect.
         *
         * Unit: none
         */
        vec3<vec2<float_64>> m_strength;

        friend class impl::Exponential;

    public:
        /** Create exponential absorber
         *
         * @param numCells
         * @param strength
         */
        Exponential(
            vec3<vec2<uint32_t>> const& numCells = {{12, 12}, {12, 12}, {12, 12}},
            vec3<vec2<float_64>> const& strength = {{1.0e-3, 1.0e-3}, {1.0e-3, 1.0e-3}, {1.0e-3, 1.0e-3}})
            : m_numCells(numCells)
            , m_strength(strength)
        {
        }

        impl::Exponential getSolver() const
        {
            return {*this};
        }

        vec3<vec2<uint32_t>> getNumCells() override
        {
            return m_numCells;
        }
    };

} // namespace picongpu::fields::boundary
