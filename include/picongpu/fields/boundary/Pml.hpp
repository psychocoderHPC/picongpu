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

#include "picongpu/fields/boundary/IBoundary.hpp"
#include "picongpu/fields/boundary/impl/Pml.hpp"
#include "picongpu/param/precision.param"

#include <pmacc/verify.hpp>

#include <cstdint>
#include <string>


namespace picongpu::fields::boundary
{
    /** Implementation of FDTD + PML updates of E and B
     *
     * The original paper on this approach is J.A. Roden, S.D. Gedney.
     * Convolution PML (CPML): An efficient FDTD implementation of the
     * CFS - PML for arbitrary media. Microwave and optical technology
     * letters. 27 (5), 334-339 (2000).
     * https://doi.org/10.1002/1098-2760(20001205)27:5%3C334::AID-MOP14%3E3.0.CO;2-A
     * Our implementation is based on a more detailed description in section
     * 7.9 of the book A. Taflove, S.C. Hagness. Computational
     * Electrodynamics. The Finite-Difference Time-Domain Method. Third
     * Edition. Artech house, Boston (2005), referred to as
     * [Taflove, Hagness].
     */
    class Pml : IBoundary
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

        /** Order of polynomial grading for artificial electric conductivity and
         *  stretching coefficient
         *
         * The conductivity (sigma) is polynomially scaling from 0 at the internal
         * border of PML to the maximum value (defined below) at the external
         * border. The stretching coefficient (kappa) scales from 1 to the
         * corresponding maximum value (defined below) with the same polynomial.
         * The grading is given in [Taflove, Hagness], eq. (7.60a, b), with
         * the order denoted 'm'.
         * Must be >= 0. Normally between 3 and 4, not required to be integer.
         * Unitless.
         */
        float_64 m_sigmaKappaGradingOrder;

        //! Muptiplier to express SIGMA_MAX_SI with SIGMA_OPT_SI. Not required, defined for convenience
        float_64 m_sigmaOptMultiplier;

        /** Max value of coordinate stretching coefficient in PML
         *
         * Components correspond to directions: element 0 corresponds to absorption
         * along x direction, 1 = y, 2 = z. Grading is described in comments for
         * SIGMA_KAPPA_GRADING_ORDER.
         * Must be >= 1. For relatively homogeneous domains 1.0 is a reasonable value.
         * Highly elongated domains can have better absorption with values between
         * 7.0 and 20.0, for example, see section 7.11.2 in [Taflove, Hagness].
         * Unitless.
         */
        vec3<float_64> m_kappaMax;

        /** Order of polynomial grading for complex frequency shift
         *
         * The complex frequency shift (alpha) is polynomially downscaling from the
         * maximum value (defined below) at the internal border of PML to 0 at the
         * external border. The grading is given in [Taflove, Hagness], eq. (7.79),
         * with the order denoted 'm_a'.
         * Must be >= 0. Normally values are around 1.0.
         * Unitless.
         */
        float_64 m_alphaGradingOrder;

        /** Complex frequency shift in PML
         *
         * Components correspond to directions: element 0 corresponds to absorption
         * along x direction, 1 = y, 2 = z. Setting it to 0 will make PML behave
         * as uniaxial PML. Setting it to a positive value helps to attenuate
         * evanescent modes, but can degrade absorption of propagating modes, as
         * described in section 7.7 and 7.11.3 in [Taflove, Hagness].
         * Must be >= 0. Normally values are 0 or between 0.15 and 0.3.
         * Unit: siemens / m.
         */
        vec3<float_64> m_alphaMax;

        friend class impl::Pml;

    public:
        Pml(vec3<vec2<uint32_t>> const& numCells = {{12, 12}, {12, 12}, {12, 12}},
            float_64 const sigmaKappaGradingOrder = 4.0,
            float_64 const sigmaOptMultiplier = 1.0,
            vec3<float_64> const& kappaMax = {1.0, 1.0, 1.0},
            float_64 const alphaGradingOrder = 1.0,
            vec3<float_64> const& alphaMax = {0.2, 0.2, 0.2})
            : m_numCells(numCells)
            , m_sigmaKappaGradingOrder(sigmaKappaGradingOrder)
            , m_sigmaOptMultiplier(sigmaOptMultiplier)
            , m_kappaMax(kappaMax)
            , m_alphaGradingOrder(alphaGradingOrder)
            , m_alphaMax(alphaMax)
        {
        }

        impl::Pml getSolver() const
        {
            PMACC_VERIFY_MSG(
                m_sigmaKappaGradingOrder >= 0.0,
                "You_can_not_set_negative_grading_order_for_pml_kappa_and_sigma");
            PMACC_VERIFY_MSG(m_alphaGradingOrder >= 0.0, "You_can_not_set_negative_grading_order_for_pml_alpha");
            for(uint32_t d = 0u; d < simDim; ++d)
                PMACC_VERIFY_MSG(m_alphaMax[d] >= 0.0, "You_can_not_set_negative_pml_alpha_max");

            return {*this};
        }

        vec3<vec2<uint32_t>> getNumCells() override
        {
            return m_numCells;
        }
    };

} // namespace picongpu::fields::boundary
