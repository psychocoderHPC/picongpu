/**
 * Copyright 2013 Axel Huebl, Felix Schmitt, Rene Widera
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
#include "simulation_types.hpp"
#include "plugins/hdf5/HDF5Writer.def"
#include "traits/PICToSplash.hpp"
#include "traits/GetComponentsType.hpp"
#include "traits/GetNComponents.hpp"


namespace picongpu
{

namespace hdf5
{
using namespace PMacc;

using namespace splash;

/** write attribute of a particle to hdf5 file
 *
 * @tparam T_Identifier identifier of a particle attribute
 */
template< typename T_Identifier>
struct LoadParticleAttributesFromHDF5
{

    /** write attribute to hdf5 file
     *
     * @param params wrapped params with domainwriter, ...
     * @param frame frame with all particles
     * @param prefix a name prefix for hdf5 attribute (is combined to: prefix_nameOfAttribute)
     * @param simOffset offset from window origin of thedomain
     * @param localSize local domain size
     */
    template<typename FrameType>
    HINLINE void operator()(
                            ThreadParams* params,
                            FrameType& frame,
                            const std::string subGroup,
                            const size_t particlesOffset,
                            const size_t elements)
    {

        typedef T_Identifier Identifier;
        typedef typename Identifier::type ValueType;
        const uint32_t components = GetNComponents<ValueType>::value;
        typedef typename GetComponentsType<ValueType>::type ComponentType;
        typedef typename PICToSplash<ComponentType>::type SplashType;

        log<picLog::INPUT_OUTPUT > ("HDF5:  ( begin ) load species attribute: %1%") % Identifier::getName();

        SplashType splashType;
        const std::string name_lookup[] = {"x", "y", "z"};


        const DomainInformation domInfo;

        ComponentType* tmpArray = new ComponentType[elements];

        ParallelDomainCollector* dataCollector = params->dataCollector;
        for (uint32_t d = 0; d < components; d++)
        {
            std::stringstream datasetName;
            datasetName << subGroup << "/" << T_Identifier::getName();
            if (components > 1)
                datasetName << "/" << name_lookup[d];

            ValueType* dataPtr = frame.getIdentifier(Identifier()).getPointer();
            Dimensions sizeRead(0, 0, 0);
            // read one component from file to temporary array
            dataCollector->read(params->currentStep,
                               Dimensions(elements, 1, 1),
                               Dimensions(particlesOffset, 0, 0),
                               datasetName.str().c_str(),
                               sizeRead,
                               tmpArray
                               );
            assert(sizeRead[0] == elements);

            /* copy component from temporary array to array of structs */
            for (size_t i = 0; i < elements; ++i)
            {
                ComponentType& ref = ((ComponentType*) dataPtr)[i * components + d];
                ref = tmpArray[i];
            }
        }
        __deleteArray(tmpArray);

        log<picLog::INPUT_OUTPUT > ("HDF5:  ( end ) load species attribute: %1%") %
            Identifier::getName();
    }

};

} //namspace hdf5

} //namespace picongpu

