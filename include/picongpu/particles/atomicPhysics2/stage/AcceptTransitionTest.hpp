/* Copyright 2023 Brian Marre
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

//! @file calculateRate and acceptTransition substage of atomicPhysics

#pragma once

#include "picongpu/simulation_defines.hpp"
// need: ProbabilityApproximationFunctor from picongpu/param/atomicPhysics2.param

#include "picongpu/particles/atomicPhysics2/electronDistribution/LocalHistogramField.hpp"
#include "picongpu/particles/atomicPhysics2/kernel/AcceptTransitionTest_AutonomousIonization.kernel"
#include "picongpu/particles/atomicPhysics2/kernel/AcceptTransitionTest_ElectronicDeOrExcitation.kernel"
#include "picongpu/particles/atomicPhysics2/kernel/AcceptTransitionTest_ElectronicIonization.kernel"
#include "picongpu/particles/atomicPhysics2/kernel/AcceptTransitionTest_NoChange.kernel"
#include "picongpu/particles/atomicPhysics2/kernel/AcceptTransitionTest_SpontaneousDeexcitation.kernel"
#include "picongpu/particles/atomicPhysics2/localHelperFields/LocalRateCacheField.hpp"
#include "picongpu/particles/atomicPhysics2/localHelperFields/LocalTimeStepField.hpp"
#include "picongpu/particles/atomicPhysics2/processClass/TransitionOrdering.hpp"

#include <cstdint>

namespace picongpu::particles::atomicPhysics2::stage
{
    namespace procClass = picongpu::particles::atomicPhysics2::processClass;

    /** @class atomic physics sub-stage calculating the rate of the chosen transition
     *  (chosen by chooseTransition sub-stage and transitionCollctionIndex as well as
     *  processClass extracted by ExtractTransitionCollectionIndex sub-stage)
     *  and tests for acceptance.
     *
     * @attention assumes that both the chooseTransition and ExtractTransitionCollectionIndex
     *  stages have been executed previously in the current atomicPhysics time step
     *
     * @tparam T_IonSpecies ion species type
     *
     */
    template<typename T_IonSpecies>
    struct AcceptTransitionTest
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_IonSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        using DistributionFloat = pmacc::random::distributions::Uniform<float_X>;
        using RngFactoryFloat = particles::functor::misc::Rng<DistributionFloat>;

        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc, uint32_t const currentStep, EventTask depEvent1) const
        {
            EventTask stageEvent;

            eventSystem::startTransaction(depEvent1);
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
            pmacc::lockstep::WorkerCfg workerCfg = pmacc::lockstep::makeWorkerCfg(MappingDesc::SuperCellSize{});

            using AtomicDataType = typename picongpu::traits::GetAtomicDataType<IonSpecies>::type;
            auto& atomicData = *dc.get<AtomicDataType>(IonSpecies::FrameType::getName() + "_atomicData");

            auto& ions = *dc.get<IonSpecies>(IonSpecies::FrameType::getName());

            using SpeciesConfigNumberType = typename AtomicDataType::ConfigNumber;

            auto& localElectronHistogramField
                = *dc.get<picongpu::particles::atomicPhysics2::electronDistribution::
                              LocalHistogramField<picongpu::atomicPhysics2::ElectronHistogram, picongpu::MappingDesc>>(
                    "Electron_localHistogramField");

            auto& localTimeStepField = *dc.get<
                picongpu::particles::atomicPhysics2 ::localHelperFields::LocalTimeStepField<picongpu::MappingDesc>>(
                "LocalTimeStepField");

            auto& localRateCacheField = *dc.get<picongpu::particles::atomicPhysics2 ::localHelperFields::
                                                    LocalRateCacheField<picongpu::MappingDesc, IonSpecies>>(
                IonSpecies::FrameType::getName() + "_localRateCacheField");

            RngFactoryFloat rngFactory = RngFactoryFloat{currentStep};

            auto depEvent=eventSystem::endTransaction();

            eventSystem::startTransaction(depEvent);
            // transition no-change
            PMACC_LOCKSTEP_KERNEL(
                picongpu::particles::atomicPhysics2::kernel::AcceptTransitionTestKernel_NoChange<
                    SpeciesConfigNumberType,
                    picongpu::atomicPhysics2::ProbabilityApproximationFunctor>(),
                workerCfg)
            (mapper.getGridDim())(
                mapper,
                rngFactory,
                ions.getDeviceParticlesBox(),
                localTimeStepField.getDeviceDataBox(),
                localRateCacheField.getDeviceDataBox(),
                atomicData.template getChargeStateOrgaDataBox<false>(),
                atomicData.template getAtomicStateDataDataBox<false>());
            stageEvent+=eventSystem::endTransaction();

            // bound-bound up
            //      electronic collisional excitation
            if constexpr(AtomicDataType::switchElectronicExcitation)
            {
                eventSystem::startTransaction(depEvent);
                PMACC_LOCKSTEP_KERNEL(
                    picongpu::particles::atomicPhysics2::kernel ::AcceptTransitionTestKernel_ElectronicDeOrExcitation<
                        picongpu::atomicPhysics2::ElectronHistogram,
                        AtomicDataType::ConfigNumber::numberLevels,
                        picongpu::atomicPhysics2::ProbabilityApproximationFunctor,
                        /* excitation */ true>(),
                    workerCfg)
                (mapper.getGridDim())(
                    mapper,
                    rngFactory,
                    ions.getDeviceParticlesBox(),
                    localTimeStepField.getDeviceDataBox(),
                    localElectronHistogramField.getDeviceDataBox(),
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData.template getBoundBoundTransitionDataBox<
                        false,
                        procClass::TransitionOrdering::byLowerState>());
                stageEvent+=eventSystem::endTransaction();
            }

            // bound-bound down
            //      electronic collisional deexcitation
            if constexpr(AtomicDataType::switchElectronicDeexcitation)
            {
                eventSystem::startTransaction(depEvent);
                PMACC_LOCKSTEP_KERNEL(
                    picongpu::particles::atomicPhysics2::kernel::AcceptTransitionTestKernel_ElectronicDeOrExcitation<
                        picongpu::atomicPhysics2::ElectronHistogram,
                        AtomicDataType::ConfigNumber::numberLevels,
                        picongpu::atomicPhysics2::ProbabilityApproximationFunctor,
                        /* deexcitation */ false>(),
                    workerCfg)
                (mapper.getGridDim())(
                    mapper,
                    rngFactory,
                    ions.getDeviceParticlesBox(),
                    localTimeStepField.getDeviceDataBox(),
                    localElectronHistogramField.getDeviceDataBox(),
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData.template getBoundBoundTransitionDataBox<
                        false,
                        procClass::TransitionOrdering::byUpperState>());
                stageEvent+=eventSystem::endTransaction();
            }
            //      spontaneous radiative deexcitation
            if constexpr(AtomicDataType::switchSpontaneousDeexcitation)
            {
                eventSystem::startTransaction(depEvent);
                PMACC_LOCKSTEP_KERNEL(
                    picongpu::particles::atomicPhysics2::kernel::AcceptTransitionTestKernel_SpontaneousDeexcitation<
                        AtomicDataType::ConfigNumber::numberLevels,
                        picongpu::atomicPhysics2::ProbabilityApproximationFunctor>(),
                    workerCfg)
                (mapper.getGridDim())(
                    mapper,
                    rngFactory,
                    ions.getDeviceParticlesBox(),
                    localTimeStepField.getDeviceDataBox(),
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData.template getBoundBoundTransitionDataBox<
                        false,
                        procClass::TransitionOrdering::byUpperState>());
                stageEvent+=eventSystem::endTransaction();
            }

            // bound-free up
            //      electronic collisional ionization
            if constexpr(AtomicDataType::switchElectronicIonization)
            {
                eventSystem::startTransaction(depEvent);
                PMACC_LOCKSTEP_KERNEL(
                    picongpu::particles::atomicPhysics2::kernel::AcceptTransitionTestKernel_ElectronicIonization<
                        picongpu::atomicPhysics2::ElectronHistogram,
                        AtomicDataType::ConfigNumber::numberLevels,
                        picongpu::atomicPhysics2::ProbabilityApproximationFunctor>(),
                    workerCfg)
                (mapper.getGridDim())(
                    mapper,
                    rngFactory,
                    ions.getDeviceParticlesBox(),
                    localTimeStepField.getDeviceDataBox(),
                    localElectronHistogramField.getDeviceDataBox(),
                    atomicData.template getChargeStateDataDataBox<false>(),
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData
                        .template getBoundFreeTransitionDataBox<false, procClass::TransitionOrdering::byLowerState>());
                stageEvent+=eventSystem::endTransaction();
            }
            //      fieldIonization
            /// @todo implement field ionization, Brian Marre, 2023

            // bound-free down
            /// @todo implement recombination, Brian Marre, 2023

            // autonomous ionization
            if constexpr(AtomicDataType::switchAutonomousIonization)
            {
                eventSystem::startTransaction(depEvent);
                PMACC_LOCKSTEP_KERNEL(
                    picongpu::particles::atomicPhysics2::kernel::AcceptTransitionTestKernel_AutonomousIonization<
                        picongpu::atomicPhysics2::ProbabilityApproximationFunctor>(),
                    workerCfg)
                (mapper.getGridDim())(
                    mapper,
                    rngFactory,
                    ions.getDeviceParticlesBox(),
                    localTimeStepField.getDeviceDataBox(),
                    atomicData.template getAutonomousTransitionDataBox<
                        false,
                        procClass::TransitionOrdering::byUpperState>());
                stageEvent+=eventSystem::endTransaction();
            }
            eventSystem::setTransactionEvent(stageEvent);
        }
    };
} // namespace picongpu::particles::atomicPhysics2::stage
