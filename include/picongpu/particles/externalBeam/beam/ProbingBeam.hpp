/* Copyright 2020-2021 Pawel Ordyna
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

#include "picongpu/particles/externalBeam/beam/CoordinateTransform.hpp"
#include "picongpu/particles/externalBeam/beam/ProbingBeam.def"
#include "picongpu/particles/externalBeam/beam/SqrtWrapper.hpp"
#include "picongpu/particles/externalBeam/beam/beamProfiles/profiles.hpp"
#include "picongpu/particles/externalBeam/beam/beamShapes/shapes.def"
#include "picongpu/particles/externalBeam/beam/beamShapes/shapes.hpp"


namespace picongpu
{
    namespace particles
    {
        namespace externalBeam
        {
            namespace beam
            {
                /** Defines the probing beam characteristic.
                 *
                 * @tparam T_BeamProfile Beam transverse profile.
                 * @tparam T_BeamShape Beam temporal shape.
                 * @tparam T_CoordinateTransform Coordinate transform from the pic
                 *      coordinate system to the beam coordinate system.
                 */
                template<typename T_BeamProfile, typename T_BeamShape, typename T_CoordinateTransform>
                HINLINE ProbingBeam<T_BeamProfile, T_BeamShape, T_CoordinateTransform>::ProbingBeam()
                    : coordinateTransform()
                {
                }

                /** Calculates the probing amplitude at a given position.
                 * @param position_b Position in the beam comoving coordinate system
                 *      (x, y, z__at_t_0 - c*t).
                 * @returns Probing wave amplitude scaling at position_b.
                 */
                template<typename T_BeamProfile, typename T_BeamShape, typename T_CoordinateTransform>
                HDINLINE float_X ProbingBeam<T_BeamProfile, T_BeamShape, T_CoordinateTransform>::operator()(
                    float3_X const& position_b) const
                {
                    float_X profileFactor = BeamProfile::getFactor(position_b[0], position_b[1]);

                    // negative time in front of the pulse positive time behind the pulse
                    float_X beamTime = -1.0_X * position_b[2] / SPEED_OF_LIGHT;
                    float_X shapeFactor = BeamShape::getFactor(beamTime);

                    return profileFactor * shapeFactor;
                }
            } // namespace beam
        } // namespace externalBeam
    } // namespace particles
} // namespace picongpu
