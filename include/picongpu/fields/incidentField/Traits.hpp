/* Copyright 2020-2022 Sergei Bastrakov
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

#include <pmacc/meta/conversion/MakeSeq.hpp>

#include <cstdint>
#include <type_traits>


namespace picongpu
{
    namespace fields
    {
        namespace incidentField
        {
            namespace detail
            {
                /** Get type of incident field functor for the given profile type
                 *
                 * The resulting functor is set as ::type.
                 * These traits have to be specialized by all profiles.
                 *
                 * @tparam T_Profile profile type
                 *
                 * @{
                 */

                //! Get functor for incident E values
                template<typename T_Profile>
                struct GetFunctorIncidentE;

                //! Get functor for incident B values
                template<typename T_Profile>
                struct GetFunctorIncidentB;

                /** @} */

                /** Type of incident E/B functor for the given profile type
                 *
                 * These are helper aliases to wrap GetFunctorIncidentE/B.
                 * The latter present customization points.
                 *
                 * @tparam T_Profile profile type
                 *
                 * @{
                 */

                //! Functor for incident E values
                template<typename T_Profile>
                using FunctorIncidentE = typename GetFunctorIncidentE<T_Profile>::type;

                //! Functor for incident B values
                template<typename T_Profile>
                using FunctorIncidentB = typename GetFunctorIncidentB<T_Profile>::type;

                /** @} */

            } // namespace detail

            /** Get max E field amplitude for the given profile type
             *
             * The resulting value is set as ::value, in internal units.
             * This trait has to be specialized by all profiles.
             *
             * @tparam T_Profile profile type
             *
             * @{
             */

            //! Generic implementation for all profiles with parameter structs
            template<typename T_Profile>
            struct GetAmplitude
            {
                using FunctorE = detail::FunctorIncidentE<T_Profile>;
                static constexpr float_X value = FunctorE::Unitless::AMPLITUDE;
            };

            //! Specialization for None profile which has no amplitude
            template<>
            struct GetAmplitude<profiles::None>
            {
                static constexpr float_X value = 0.0_X;
            };

            //! Specialization for Free profile which has unknown amplitude
            template<typename T_FunctorIncidentE, typename T_FunctorIncidentB>
            struct GetAmplitude<profiles::Free<T_FunctorIncidentE, T_FunctorIncidentB>>
            {
                static constexpr float_X value = 0.0_X;
            };

            /** @} */

            /** Max E field amplitude in internal units for the given profile type
             *
             * @tparam T_Profile profile type
             */
            template<typename T_Profile>
            constexpr float_X amplitude = GetAmplitude<T_Profile>::value;

            //! Typelist of all enabled profiles, can contain duplicates
            using EnabledProfiles = pmacc::MakeSeq_t<
                XMin,
                XMax,
                YMin,
                YMax,
                std::conditional_t<simDim == 3, pmacc::MakeSeq_t<ZMin, ZMax>, pmacc::MakeSeq_t<>>>;

        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
