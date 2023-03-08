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

#include <pmacc/math/Vector.hpp>
#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/Resolve.hpp>
namespace picongpu
{
    namespace particles
    {
        template<
            typename T_Species,
            bool hasWavelength = pmacc::traits::HasFlag<typename T_Species::FrameType, wavelength<>>::type::value>
        struct GetAngFrequency
        {
            HDINLINE static constexpr float_X get()
            {
                using FrameType = typename Species::FrameType;
                using WavelengthFlag =
                    typename pmacc::traits::Resolve<typename GetFlagType<FrameType, wavelength<>>::type>::type;
                constexpr float_X wavelength = WavelengthFlag::getValue();
                return pmacc::math::Pi<float_X>::doubleValue * SPEED_OF_LIGHT / wavelength;
            }
            using Species = T_Species;
            HDINLINE float_X operator()() const
            {
                return get();
            }

            template<typename T_Particle>
            HDINLINE float_X operator()(const T_Particle& particle) const
            {
                return get();
            }
        };

        template<typename T_Species>
        struct GetAngFrequency<T_Species, false>
        {
            template<typename T_Particle>
            HDINLINE float_X operator()(const T_Particle& particle) const
            {
                float_X weighting = particle[weighting_];
                float_X momentum = math::abs(particle[momentum_]);
                return momentum / HBAR / weighting * SPEED_OF_LIGHT;
            }
        };

        /**
         * Returns the phase for a given timestep
         */
        template<typename T_Species>
        struct GetPhaseByTimestep
        {
            using Species = T_Species;

            HINLINE float_64 operator()(const uint32_t currentStep, float_64 phi_0 = 0.0) const
            {
                static const float_64 omega = GetAngFrequency<Species>()();
                /* phase phi = phi_0 - omega * t;
                 * Note: This MUST be calculated in double precision as single precision is inexact after ~100
                 * timesteps Double precision is enough for about 10^10 timesteps More timesteps (in SP&DP) are
                 * possible, if the product is implemented as a summation with summands reduced to 2*PI */
                static const float_64 phaseDiffPerTimestep = fmod(omega * DELTA_T, 2.0 * PI);
                // Reduce summands to range of 2*PI to avoid bit canceling
                float_64 dPhi = math::fmod(phaseDiffPerTimestep * static_cast<float_64>(currentStep), 2.0 * PI);
                phi_0 = math::fmod(phi_0, 2.0 * PI);
                float_64 result = phi_0 - dPhi;
                // Keep in range of [0,2*PI)
                if(result < 0)
                    result += 2.0 * PI;
                return result;
            }

            HDINLINE float_64
            operator()(const uint32_t currentStep, Species const& particle, float_64 phi_0 = 0.0) const
            {
                const float_64 omega = GetAngFrequency<Species>()(particle);
                /* phase phi = phi_0 - omega * t;
                 * Note: This MUST be calculated in double precision as single precision is inexact after ~100
                 * timesteps Double precision is enough for about 10^10 timesteps More timesteps (in SP&DP) are
                 * possible, if the product is implemented as a summation with summands reduced to 2*PI */
                const float_64 phaseDiffPerTimestep = math::fmod(omega * DELTA_T, 2.0 * PI);
                // Reduce summands to range of 2*PI to avoid bit canceling
                float_64 dPhi = math::fmod(phaseDiffPerTimestep * static_cast<float_64>(currentStep), 2.0 * PI);
                phi_0 = math::fmod(phi_0, 2.0 * PI);
                float_64 result = phi_0 - dPhi;
                // Keep in range of [0,2*PI)
                if(result < 0)
                    result += 2.0 * PI;
                return result;
            }
        };
    } // namespace particles
} // namespace picongpu
