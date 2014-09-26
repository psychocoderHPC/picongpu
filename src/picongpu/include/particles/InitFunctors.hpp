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
#include "fields/Fields.def"
#include "math/MapTuple.hpp"
#include <boost/mpl/plus.hpp>
#include <boost/mpl/accumulate.hpp>
#include <boost/mpl/apply.hpp

namespace picongpu
{

namespace particles
{

template<typename T_SpeciesName,typename T_Functor=bmpl::_1>
struct CallSpeciesFunctor
{
    typedef T_SpeciesName SpeciesName;
    typedef T_Functor Functor;

    template<typename T_StorageTuple, typename T_Event>
    HINLINE void operator()(
                            T_StorageTuple& tuple,
                            const uint32_t currentStep
                            )
    {
        Functor()(tuple,currentStep);
    }
};


template<typename T_SpeciesName=bmpl::_1>
struct InitFunctorSpecies
{
    typedef T_SpeciesName SpeciesName;

    template<typename T_StorageTuple, typename T_Event>
    HINLINE void operator()(
                            T_StorageTuple& tuple,
                            const uint32_t currentStep
                            )
    {
            typedef typename GetFlagType<typename SpeciesType::FrameType, initMethods<> >::type::ThisType InitMethods;
            /* add species name to helper functor*/
            typedef CallSpeciesFunctor<SpeciesName> Functor;

            ForEach<InitMethods, Functor > callSpeciesFunctor;
            callSpeciesFunctor(tuple,currentStep);

    }
};

template<typename T_GasFunctor,typename T_PositionFunctor,typename T_SpeciesName=bmpl::_1>
struct CreateGas
{
    typedef T_SpeciesName SpeciesName;
    typedef typename bmpl::apply1<T_GasFunctor, SpeciesName>::type  GasFunctor;
    typedef typename bmpl::apply1<T_PositionFunctor, SpeciesName>::type PositionFunctor;
    typedef typename SpeciesName::type SpeciesType;
    typedef typename SpeciesType::FrameType FrameType;

    template<typename T_StorageTuple, typename T_Event>
    HINLINE void operator()(
                            T_StorageTuple& tuple,
                            const uint32_t currentStep
                            )
    {
            PMACC_AUTO(speciesPtr, tuple[SpeciesName()]);
            GasFunctor gasFunctor;
            PositionFunctor positionFunctor;
            speciesPtr->initGas(gasFunctor,positionFunctor,currentStep);
    }
};

template<typename T_SrcSpeciesType,typename T_SpeciesName=bmpl::_1>
struct CloneSpecies
{
    typedef T_SpeciesName SpeciesName;
    typedef T_SrcSpeciesType SrcSpeciesType;
    typedef MakeIdentifier<SrcSpeciesType> SrcSpeciesName;

    typedef typename SpeciesName::type SpeciesType;
    typedef typename SpeciesType::FrameType FrameType;

    template<typename T_StorageTuple, typename T_Event>
    HINLINE void operator()(
                            T_StorageTuple& tuple,
                            const uint32_t currentStep
                            )
    {
            PMACC_AUTO(speciesPtr, tuple[SpeciesName()]);
            PMACC_AUTO(srcSpeciesPtr, tuple[SrcSpeciesName()]);

            speciesPtr->deviceCloneFrom(*srcSpeciesPtr);
    }
};


} //namespace particles

} //namespace picongpu
