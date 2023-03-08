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

#include "picongpu/particles/PhotonFunctors.hpp"
#include "picongpu/particles/externalBeam/beam/AxisSwap.hpp"

namespace picongpu::particles::externalBeam::phase
{
    namespace acc
    {
        template<typename T_ParamClass>
        struct FromPhotonMomentum
        {
            static constexpr float_64 phi0 = T_ParamClass::phi0;

            using SideCfg = typename T_ParamClass::ProbingBeam::SideCfg;
            template<uint32_t idx>
            using BeamToPicIdx_t = typename SideCfg::AxisSwapCT::template BeamToPicIdx<idx>::type;


            DINLINE explicit FromPhotonMomentum(uint32_t const& currentStep)
                : axisSwap(SideCfg::getAxisSwap())
                , currentStep_m(currentStep)
            {
            }
            /* Set phase
             *
             * @tparam T_Context start attributes context
             * @tparam T_Particle pmacc::Particle, particle type
             * @tparam T_Args pmacc::Particle, arbitrary number of particles types
             *
             * @param context start attributes context
             * @param particle particle to be manipulated
             * @param ... unused particles
             */
            template<typename T_Context, typename T_Particle, typename... T_Args>
            DINLINE void operator()(T_Context const& context, T_Particle& particle, T_Args&&...) const
            {
                float_X phase
                    = precisionCast<float_X>(GetPhaseByTimestep<T_Particle>()(currentStep_m, particle, phi0));

                static constexpr float_X cellSizeCT[3] = {CELL_WIDTH, CELL_HEIGHT, CELL_DEPTH};
                static constexpr float_X cellDepth{cellSizeCT[BeamToPicIdx_t<2u>::value]};

                /* this functor will be called only in the first cell (counting from the boundary where
                the photon beam enters the simulation. So global position along the propagation axis
                is just the in cell position. */
                const floatD_X position = particle[position_];
                // distance from the z_beam=0 plane (the simulation boundary) where the position contribution
                // to the plane wave phase is 0.
                const float_X distance{(axisSwap.rotate(position)).z() * cellDepth};
                const float_X waveNumber{GetAngFrequency<T_Particle>()() / SPEED_OF_LIGHT};
                const float_X spatialContribution = waveNumber * distance;
                phase += spatialContribution;
                particle[startPhase_] = phase;
            }

        private:
            PMACC_ALIGN(axisSwap, beam::AxisSwap);
            PMACC_ALIGN(currentStep_m, uint32_t);
        };
    } // namespace acc

    template<typename T_ParamClass>
    struct FromPhotonMomentum
    {
        HINLINE explicit FromPhotonMomentum(uint32_t const& currentStep)
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
        DINLINE auto operator()(T_Worker const& worker, DataSpace<simDim> const& localSupercellOffset) const
        {
            return acc::FromPhotonMomentum<T_ParamClass>();
        }
    };
} // namespace picongpu::particles::externalBeam::phase
