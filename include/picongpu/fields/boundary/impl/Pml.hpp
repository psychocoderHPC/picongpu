/* Copyright 2013-2023 Sergei Bastrakov, Rene Widera
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

#include "picongpu/fields/boundary/impl/Thickness.hpp"
#include "picongpu/fields/boundary/impl/pml/Parameters.hpp"
#include "picongpu/fields/boundary/impl/pml/Pml.kernel"
#include "picongpu/param/precision.param"

#include <cstdint>


namespace picongpu::fields::boundary
{
    class Pml;

    namespace impl
    {
        class Pml
        {
            pml::Parameters m_param;
            vec3<vec2<uint32_t>> m_numCells;

        public:
            Pml(boundary::Pml const&);

            /** Functor to update electric field by a time step using FDTD with the given curl and PML
             *
             * @tparam T_CurlB curl functor type according to the Curl concept
             *
             * @param currentStep index of the current time iteration
             */
            template<typename T_CurlB>
            pml::UpdateEFunctor<T_CurlB> getUpdateEFunctor(float_X const currentStep)
            {
                return {psiE->getDeviceOuterLayerBox(), getLocalParameters(currentStep)};
            }

            /** Functor to update magnetic field by half a time step using FDTD with the given curl and PML
             *
             * @tparam T_CurlE curl functor type according to the Curl concept
             *
             * @param currentStep index of the current time iteration
             * @param updatePsiB whether convolutional magnetic fields need to be updated, or are
             * up-to-date
             */
            template<typename T_CurlE>
            pml::UpdateBHalfFunctor<T_CurlE> getUpdateBHalfFunctor(float_X const currentStep, bool const updatePsiB)
            {
                return {psiB->getDeviceOuterLayerBox(), getLocalParameters(currentStep), updatePsiB};
            }

            /** Get m_param for the local domain
             *
             * @param currentStep index of the current time iteration
             */
            pml::LocalParameters getLocalParameters(float_X const currentStep) const;

        private:
            /** Get PML thickness for the local domain at the current time step.
             *
             * It depends on the current step because of the moving window.
             */
            Thickness getLocalThickness(float_X const currentStep) const;

            /** Get absorber thickness in number of cells for the global domain
             *
             * This function takes into account which boundaries are periodic and absorbing.
             */
            impl::Thickness getGlobalThickness() const;

            //! Verify that PML fits the local domain
            void checkLocalThickness(Thickness const localThickness) const;

            /* PML convolutional field data, defined as in [Taflove, Hagness],
             * eq. (7.105a,b), and similar for other components
             */
            std::shared_ptr<pml::FieldE> psiE;
            std::shared_ptr<pml::FieldB> psiB;
        };

    } // namespace impl
} // namespace picongpu::fields::boundary
