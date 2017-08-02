/* Copyright 2014-2017 Alexander Debus, Axel Huebl
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

#include <pmacc/types.hpp>

#include <pmacc/math/Vector.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include "picongpu/fields/background/templates/TWTS/numComponents.hpp"

namespace picongpu
{
/* Load pre-defined background field */
namespace templates
{
/* Traveling-wave Thomson scattering laser pulse */
namespace twts
{

class EField
{
public:
    typedef float_X float_T;

    enum PolarizationType
    {
        /* The linear polarization of the TWTS laser is defined
         * relative to the plane of the pulse front tilt.
         *
         * Polarisation is normal to the reference plane.
         * Use Ex-fields (and corresponding B-fields) in TWTS laser internal coordinate system.
         */
        LINEAR_X = 1u,
        /* Polarization lies within the reference plane.
         * Use Ey-fields (and corresponding B-fields) in TWTS laser internal coordinate system.
         */
        LINEAR_YZ = 2u,
    };

    /* Center of simulation volume in number of cells */
    PMACC_ALIGN(halfSimSize,DataSpace<simDim>);
    /* y-position of TWTS coordinate origin inside the simulation coordinates [meter]
       The other origin coordinates (x and z) default to globally centered values
       with respect to the simulation volume. */
    PMACC_ALIGN(focus_y_SI, const float_64);
    /* Laser wavelength [meter] */
    PMACC_ALIGN(wavelength_SI, const float_64);
    /* TWTS laser pulse duration [second] */
    PMACC_ALIGN(pulselength_SI, const float_64);
    /* line focus height of TWTS pulse [meter] */
    PMACC_ALIGN(w_x_SI, const float_64);
    /* line focus width of TWTS pulse [meter] */
    PMACC_ALIGN(w_y_SI, const float_64);
    /* interaction angle between TWTS laser propagation vector and the y-axis [rad] */
    PMACC_ALIGN(phi, const float_X);
    /* Takes value 1.0 for phi > 0 and -1.0 for phi < 0. */
    PMACC_ALIGN(phiPositive,float_X);
    /* propagation speed of TWTS laser overlap
    normalized to the speed of light. [Default: beta0=1.0] */
    PMACC_ALIGN(beta_0, const float_X);
    /* If auto_tdelay=FALSE, then a user defined delay is used. [second] */
    PMACC_ALIGN(tdelay_user_SI, const float_64);
    /* Make time step constant accessible to device. */
    PMACC_ALIGN(dt, const float_64);
    /* Make length normalization constant accessible to device. */
    PMACC_ALIGN(unit_length, const float_64);
    /* TWTS laser time delay */
    PMACC_ALIGN(tdelay,float_64);
    /* Should the TWTS laser delay be chosen automatically, such that
     * the laser gradually enters the simulation volume? [Default: TRUE]
     */
    PMACC_ALIGN(auto_tdelay, const bool);
    /* Polarization of TWTS laser */
    PMACC_ALIGN(pol, const PolarizationType);

    /** Electric field of the TWTS laser
     *
     * \param focus_y_SI the distance to the laser focus in y-direction [m]
     * \param wavelength_SI central wavelength [m]
     * \param pulselength_SI sigma of std. gauss for intensity (E^2),
     *  pulselength_SI = FWHM_of_Intensity / 2.35482 [seconds (sigma)]
     * \param w_x beam waist: distance from the axis where the pulse electric field
     *  decreases to its 1/e^2-th part at the focus position of the laser [m]
     * \param w_y \see w_x
     * \param phi interaction angle between TWTS laser propagation vector and
     *  the y-axis [rad, default = 90.*(PI/180.)]
     * \param beta_0 propagation speed of overlap normalized to
     *  the speed of light [c, default = 1.0]
     * \param tdelay_user manual time delay if auto_tdelay is false
     * \param auto_tdelay calculate the time delay such that the TWTS pulse is not
     *  inside the simulation volume at simulation start timestep = 0 [default = true]
     * \param pol dtermines the TWTS laser polarization, which is either normal or parallel
     *  to the laser pulse front tilt plane [ default= LINEAR_X , LINEAR_YZ ]
     */
    HINLINE
    EField( const float_64 focus_y_SI,
            const float_64 wavelength_SI,
            const float_64 pulselength_SI,
            const float_64 w_x_SI,
            const float_64 w_y_SI,
            const float_X phi               = 90.*(PI / 180.),
            const float_X beta_0            = 1.0,
            const float_64 tdelay_user_SI   = 0.0,
            const bool auto_tdelay          = true,
            const PolarizationType pol      = LINEAR_X );

    /** Specify your background field E(r,t) here
     *
     * \param cellIdx The total cell id counted from the start at timestep 0.
     * \param currentStep The current time step
     * \return float3_X with field normalized to amplitude in range [-1.:1.]
     */
    HDINLINE float3_X
    operator()( const DataSpace<simDim>& cellIdx,
                const uint32_t currentStep ) const;

    /** Calculate the Ex(r,t) field here (electric field vector normal to pulse-front-tilt plane)
     *
     * \param pos Spatial position of the target field
     * \param time Absolute time (SI, including all offsets and transformations)
     *  for calculating the field
     * \return Ex-field component of the non-rotated TWTS field in SI units */
    HDINLINE float_T
    calcTWTSEx( const float3_64& pos, const float_64 time ) const;

    /** Calculate the Ey(r,t) field here (electric field vector in pulse-front-tilt plane)
     *
     * \param pos Spatial position of the target field
     * \param time Absolute time (SI, including all offsets and transformations)
     *  for calculating the field
     * \return Ex-field component of the non-rotated TWTS field in SI units */
    HDINLINE float_T
    calcTWTSEy( const float3_64& pos, const float_64 time ) const;

    /** Calculate the E-field vector of the TWTS laser in SI units.
     * \tparam T_dim Specializes for the simulation dimension
     * \param cellIdx The total cell id counted from the start at timestep 0
     * \return Efield vector of the rotated TWTS field in SI units */
    template <unsigned T_dim>
    HDINLINE float3_X
    getTWTSEfield_Normalized(
            const pmacc::math::Vector<floatD_64,detail::numComponents>& eFieldPositions_SI,
            const float_64 time) const;

    /** Calculate the E-field vector of the "in-plane polarized" TWTS laser in SI units.
     * \tparam T_dim Specializes for the simulation dimension
     * \param cellIdx The total cell id counted from the start at timestep 0
     * \return Efield vector of the rotated TWTS field in SI units */
    template <unsigned T_dim>
    HDINLINE float3_X
    getTWTSEfield_Normalized_Ey(
            const pmacc::math::Vector<floatD_64,detail::numComponents>& eFieldPositions_SI,
            const float_64 time) const;

};

} /* namespace twts */
} /* namespace templates */
} /* namespace picongpu */
