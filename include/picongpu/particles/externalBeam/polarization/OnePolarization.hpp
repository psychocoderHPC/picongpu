/* Copyright 2021 Pawel Ordyna
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

namespace picongpu
{
    namespace particles
    {
        namespace externalBeam
        {
            namespace polarization
            {
                namespace acc
                {
                    template<typename T_Param>
                    struct OnePolarization
                    {
                        /* Do Nothing
                         *
                         * @tparam T_Acc alpaka accelerator type
                         * @tparam T_MetaData type of the low level functor used to call the StartAttributes functor
                         *      which calls this functor.
                         * @tparam T_Particle pmacc::Particle, particle type
                         * @tparam T_Args pmacc::Particle, arbitrary number of particles types
                         *
                         * @param acc alpaka accelerator
                         * @param meta the instance of T_MetaData, provides domain info and the rng.
                         * @param particle particle to be manipulated
                         * @param ... unused particles
                         */
                        template<typename T_Acc, typename T_MetaData, typename T_Particle, typename... T_Args>
                        DINLINE void operator()(T_Acc const& acc, T_MetaData& meta, T_Particle& particle, T_Args&&...)
                            const
                        {
                            particle[polarizationAngle_] = T_Param::polarization;
                        }
                    };
                } // namespace acc
                template<typename T_Param>
                struct OnePolarization
                {
                    HINLINE explicit OnePolarization(uint32_t const& currentStep)
                    {
                    }

                    /** create functor for the accelerator
                     *
                     * @tparam T_Worker lockstep worker type
                     *
                     * @param worker lockstep worker
                     * @param localSupercellOffset offset (in superCells, without any guards) relative
                     *                        to the origin of the local domain
                     */
                    template<typename T_Worker>
                    DINLINE auto operator()(T_Worker const& worker, DataSpace<simDim> const& localSupercellOffset)
                        const
                    {
                        return acc::OnePolarization<T_Param>();
                    }
                };
            } // namespace polarization
        } // namespace externalBeam
    } // namespace particles
} // namespace picongpu
