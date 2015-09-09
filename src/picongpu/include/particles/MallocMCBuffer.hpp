/**
 * Copyright 2015 Rene Widera, Benjamin Worpitz
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


#include "dataManagement/ISimulationData.hpp"
#include "simulation_defines.hpp"

#include "mallocMC/mallocMC.hpp"
#include <string>

namespace picongpu
{
    using namespace PMacc;

    class MallocMCBuffer : public ISimulationData
    {
        using BufHost = alpaka::mem::buf::Buf<
            AlpakaHostDev,
            char,
            alpaka::dim::DimInt<1u>,
            AlpakaSize>;
        using BufWrapperDev = alpaka::mem::buf::BufPlainPtrWrapper<
            AlpakaAccDev,
            char,
            alpaka::dim::DimInt<1u>,
            AlpakaSize>;

    public:
        MallocMCBuffer();

        virtual ~MallocMCBuffer();

        SimulationDataId getUniqueId()
        {
            return getName();
        }

        static std::string getName()
        {
            return std::string("MallocMCBuffer");
        }

        std::ptrdiff_t getOffset()
        {
            return hostBufferOffset;
        }

        void synchronize();

    private:
        std::unique_ptr<BufHost> upBufHost;
        std::unique_ptr<BufWrapperDev> upBufWrapperDev;
        std::ptrdiff_t hostBufferOffset;
        mallocMC::HeapInfo deviceHeapInfo;

    };


} // namespace picongpu

#include "particles/MallocMCBuffer.tpp"
