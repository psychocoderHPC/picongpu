/**
 * Copyright 2014 Rene Widera
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

#include "types.h"
#include "simulation_defines.hpp"
#include <boost/mpl/if.hpp>
#include "traits/HasFlag.hpp"
#include "traits/GetFlagType.hpp"
#include "fields/Fields.def"
#include "math/MapTuple.hpp"
#include <boost/mpl/plus.hpp>
#include <boost/mpl/accumulate.hpp>
#include <boost/mpl/apply.hpp>
#include <boost/mpl/apply_wrap.hpp>
#include "compileTime/conversion/TypeToPointerPair.hpp"
#include "particles/manipulators/manipulators.def"
#include "particles/gasProfiles/IProfile.def"
#include "particles/startPosition/IFunctor.def"
#include "traits/Resolve.hpp"

namespace picongpu
{

namespace particles
{

/** evaluate a compile time functor
 *
 * Helper functor to call a compile time lambda functor
 * bmpl::_1 is substituted with species name
 *
 * @tparam T_SpeciesName name of a species
 * @tparam T_Functor unary lambda functor
 *                   operator() must taken two params
 *                      - first: storage tuple
 *                      - second: current time step
 */
template<typename T_SpeciesName, typename T_Functor = bmpl::_1>
struct CallSpeciesFunctor
{
    typedef T_SpeciesName SpeciesName;
    typedef typename bmpl::apply1<T_Functor, SpeciesName>::type Functor;

    template<typename T_StorageTuple>
    HINLINE void operator()(
                            T_StorageTuple& tuple,
                            const uint32_t currentStep
                            )
    {
        Functor()(tuple, currentStep);
    }
};

/** evaluate a list of functors for a species
 *
 *
 */
template<typename T_Alias, typename T_SpeciesName = bmpl::_1>
struct Evaluate
{
    typedef T_SpeciesName SpeciesName;
    typedef T_Alias Alias;

    template<typename T_StorageTuple>
    HINLINE void operator()(
                            T_StorageTuple& tuple,
                            const uint32_t currentStep
                            )
    {
        typedef typename Resolve<typename GetFlagType<typename SpeciesName::FrameType, Alias >::type>::type VectorOfFunctors;
        /* add species name to helper functor*/
        typedef CallSpeciesFunctor<typename MakeIdentifier<SpeciesName>::type > Functor;

        ForEach<VectorOfFunctors, Functor > callSpeciesFunctor;
        callSpeciesFunctor(tuple, currentStep);

    }
};

/** create gas based on a gas profile and a position profile
 *
 * constructor with current time step of gas and position profile is called
 *
 * @tparam T_GasFunctor unary lambda functor with gas description
 * @tparam T_PositionFunctor unary lambda functor with position description
 * @tparam T_SPeciesName name of the used species
 */
template<typename T_GasFunctor, typename T_PositionFunctor, typename T_SpeciesName = bmpl::_1>
struct CreateGas
{
    typedef T_SpeciesName SpeciesName;

    /* add interface for compile time interface validation*/
    typedef gasProfiles::IProfile<T_GasFunctor> GasFunctor;
    /* add interface for compile time interface validation*/
    typedef startPosition::IFunctor<T_PositionFunctor> PositionFunctor;

    template<typename T_StorageTuple>
    HINLINE void operator()(
                            T_StorageTuple& tuple,
                            const uint32_t currentStep
                            )
    {
        typedef typename SpeciesName::type SpeciesType;
        typedef typename SpeciesType::FrameType FrameType;

        PMACC_AUTO(speciesPtr, tuple[SpeciesName()]);
        GasFunctor gasFunctor(currentStep);
        PositionFunctor positionFunctor(currentStep);
        speciesPtr->initGas(gasFunctor, positionFunctor, currentStep);
    }
};

template<typename T_SrcSpeciesType, typename T_SpeciesName = bmpl::_1>
struct CloneSpecies
{
    typedef T_SpeciesName SpeciesName;
    typedef T_SrcSpeciesType SrcSpeciesType;
    typedef typename MakeIdentifier<SrcSpeciesType>::type SrcSpeciesName;

    template<typename T_StorageTuple>
    HINLINE void operator()(
                            T_StorageTuple& tuple,
                            const uint32_t
                            )
    {
        PMACC_AUTO(speciesPtr, tuple[SpeciesName()]);
        PMACC_AUTO(srcSpeciesPtr, tuple[SrcSpeciesName()]);

        speciesPtr->deviceCloneFrom(*srcSpeciesPtr);
    }
};

/** run a user defined functor for every particle
 *
 * constructor with current time step is called for the functor
 *
 * @tparam T_Functor unary lambda functor
 * @tparam T_SpeciesName name of the used species
 */
template<typename T_Functor, typename T_SpeciesName = bmpl::_1>
struct Manipulate
{
    typedef T_SpeciesName SpeciesName;
    typedef manipulators::IManipulator<T_Functor> Functor;

    template<typename T_StorageTuple>
    HINLINE void operator()(
                            T_StorageTuple& tuple,
                            const uint32_t currentStep
                            )
    {
        PMACC_AUTO(speciesPtr, tuple[SpeciesName()]);
        Functor functor(currentStep);
        speciesPtr->manipulateAllParticles(currentStep, functor);
    }
};

/** call method fill all gaps of a species
 *
 * @tparam T_SpeciesName name of the species
 */
template<typename T_SpeciesName = bmpl::_1>
struct FillAllGaps
{
    typedef T_SpeciesName SpeciesName;

    template<typename T_StorageTuple>
    HINLINE void operator()(
                            T_StorageTuple& tuple,
                            const uint32_t currentStep
                            )
    {
        PMACC_AUTO(speciesPtr, tuple[SpeciesName()]);
        speciesPtr->fillAllGaps();
    }
};

} //namespace particles

} //namespace picongpu
