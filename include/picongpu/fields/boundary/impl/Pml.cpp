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

#include "picongpu/fields/boundary/Pml.hpp"

#include "picongpu/fields/boundary/impl/Pml.hpp"
#include "picongpu/fields/boundary/impl/Thickness.hpp"
#include "picongpu/fields/boundary/impl/pml/Field.hpp"
#include "picongpu/fields/boundary/impl/pml/Parameters.hpp"
#include "picongpu/param/dimension.param"
#include "picongpu/param/precision.param"
#include "picongpu/simulation/cfg/domain.hpp"
#include "picongpu/simulation/control/MovingWindow.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>

#include <cstdint>
#include <string>


namespace picongpu::fields::boundary::impl
{
    inline Pml::Pml(boundary::Pml const& base) : m_numCells(base.m_numCells)
    {
        m_param.sigmaKappaGradingOrder = static_cast<float_X>(base.m_sigmaKappaGradingOrder);
        m_param.kappaMax = precisionCast<float_X>(base.m_kappaMax).shrink<simDim>();
        m_param.alphaGradingOrder = static_cast<float_X>(base.m_alphaGradingOrder);

        // [Taflove, Hagness], eq. (7.66). Not required, defined for convenience
        auto sigmaOpt_si = vec3<float_64>::create(0.8 * (base.m_sigmaKappaGradingOrder + 1.0))
            / (setup(unit::si_).cell * setup(unit::si_).physicalConstant.z0);
        auto sigmaMax_si = sigmaOpt_si * base.m_sigmaOptMultiplier;

        // we can not throw in constructor
        // PMACC_VERIFY_MSG(sigmaMax_si >= 0.0, "You_can_not_set_negative_value_pml_sigma_max");

        auto sigmaMax = precisionCast<float_X>(
            sigmaMax_si / setup(unit::si_).physicalConstant.eps0 * setup(unit::si_).unit.time);
        m_param.normalizedSigmaMax = sigmaMax.shrink<simDim>();

        auto alphaMax = precisionCast<float_X>(
            base.m_alphaMax / setup(unit::si_).physicalConstant.eps0 * setup(unit::si_).unit.time);
        m_param.normalizedAlphaMax = alphaMax.shrink<simDim>();


        DataConnector& dc = Environment<>::get().DataConnector();
        auto domDesc = simulation::cfg::getDomainDescription();
        psiE = std::make_shared<pml::FieldE>(domDesc, getGlobalThickness());
        psiB = std::make_shared<pml::FieldB>(domDesc, getGlobalThickness());
        dc.share(psiE);
        dc.share(psiB);
    }

    pml::LocalParameters Pml::getLocalParameters(float_X const currentStep) const
    {
        impl::Thickness localThickness = getLocalThickness(currentStep);
        checkLocalThickness(localThickness);
        auto domDesc = simulation::cfg::getDomainDescription();
        return pml::LocalParameters(
            m_param,
            localThickness,
            domDesc.getGridSuperCells() * SuperCellSize::toRT(),
            domDesc.getGuardingSuperCells() * SuperCellSize::toRT());
    }

    impl::Thickness Pml::getLocalThickness(float_X const currentStep) const
    {
        /* The logic of the following checks is to disable the absorber
         * at a border we set the corresponding thickness to 0.
         */
        auto& movingWindow = MovingWindow::getInstance();
        auto const numExchanges = NumberOfExchanges<simDim>::value;
        auto const communicationMask = Environment<simDim>::get().GridController().getCommunicationMask();
        impl::Thickness localThickness = getGlobalThickness();
        for(uint32_t exchange = 1u; exchange < numExchanges; ++exchange)
        {
            /* Here we are only interested in the positive and negative
             * directions for x, y, z axes and not the "diagonal" ones.
             * So skip other directions except left, right, top, bottom,
             * back, front
             */
            if(FRONT % exchange != 0)
                continue;

            // Transform exchange into a pair of axis and direction
            uint32_t axis = 0;
            if(exchange >= BOTTOM && exchange <= TOP)
                axis = 1;
            if(exchange >= BACK)
                axis = 2;
            uint32_t direction = exchange % 2;

            // No PML at the borders between two local domains
            bool hasNeighbour = communicationMask.isSet(exchange);
            if(hasNeighbour)
                localThickness(axis, direction) = 0;

            // Disable PML at the far side of the moving window
            if(movingWindow.isSlidingWindowActive(static_cast<uint32_t>(currentStep)) && exchange == BOTTOM)
                localThickness(axis, direction) = 0;
        }
        return localThickness;
    }

    impl::Thickness Pml::getGlobalThickness() const
    {
        impl::Thickness thickness;
        for(uint32_t axis = 0u; axis < 3u; axis++)
            for(uint32_t direction = 0u; direction < 2u; direction++)
                thickness(axis, direction) = m_numCells[axis][direction];
        const DataSpace<DIM3> isPeriodicBoundary
            = Environment<simDim>::get().EnvironmentController().getCommunicator().getPeriodic();
        for(uint32_t axis = 0u; axis < 3u; axis++)
            if(isPeriodicBoundary[axis])
            {
                thickness(axis, 0) = 0u;
                thickness(axis, 1) = 0u;
            }
        return thickness;
    }

    void Pml::checkLocalThickness(impl::Thickness const localThickness) const
    {
        auto const localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
        auto const localPMLSize = localThickness.getNegativeBorder() + localThickness.getPositiveBorder();
        auto pmlFitsDomain = true;
        for(uint32_t dim = 0u; dim < simDim; dim++)
            if(localPMLSize[dim] > localDomain.size[dim])
                pmlFitsDomain = false;
        if(!pmlFitsDomain)
            throw std::out_of_range("Requested PML size exceeds the local domain");
    }
} // namespace picongpu::fields::boundary::impl
