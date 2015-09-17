/**
 * Copyright 2015 Rene Widera
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

#include "simulation_defines.hpp"
#include "mpi/SeedPerRank.hpp"

namespace picongpu
{
namespace particles
{
namespace manipulators
{

template< typename T_SpeciesType>
struct RandomPositionImpl
{
    typedef T_SpeciesType SpeciesType;
    typedef typename MakeIdentifier<SpeciesType>::type SpeciesName;

    HINLINE RandomPositionImpl(uint32_t currentStep) : isInitialized(false), gen(0)
    {
        typedef typename SpeciesType::FrameType FrameType;

        GlobalSeed globalSeed;
        mpi::SeedPerRank<simDim> seedPerRank;
        seed = seedPerRank(globalSeed(), FrameType::CommunicationTag);
        seed ^= POSITION_SEED ^ currentStep;

        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
        localCells = subGrid.getLocalDomain().size;
    }

    PMACC_NO_NVCC_HDWARNING
    template<
        typename T_Acc,
        typename T_Particle1,
        typename T_Particle2>
    DINLINE void operator()(
        T_Acc const & acc,
        const DataSpace<simDim>& localCellIdx,
        T_Particle1& particle, T_Particle2&,
        const bool isParticle, const bool)
    {
        typedef typename T_Particle1::FrameType FrameType;

        if (!isInitialized)
        {
            const uint32_t cellIdx = DataSpaceOperations<simDim>::map(
                                                                      localCells,
                                                                      localCellIdx);
            gen = alpaka::rand::generator::createDefault(acc, seed, cellIdx);
            dist = alpaka::rand::distribution::createNormalReal<float_X>(acc);
            isInitialized = true;
        }
        if (isParticle)
        {
            floatD_X tmpPos;

            for (uint32_t d = 0; d < simDim; ++d)
                tmpPos[d] = dist(gen);

            particle[position_] = tmpPos;
        }
    }

private:
    using Gen =
        decltype(
            alpaka::rand::generator::createDefault(
                std::declval<PMacc::AlpakaAcc<alpaka::dim::DimInt<simDim>> const &>(),
                std::declval<uint32_t &>(),
                std::declval<uint32_t &>()));
    PMACC_ALIGN(gen, Gen);
    using Dist =
        decltype(
            alpaka::rand::distribution::createUniformReal<float_X>(
                std::declval<PMacc::AlpakaAcc<alpaka::dim::DimInt<simDim>> const &>()));
    PMACC_ALIGN(dist, Dist);
    bool isInitialized;
    uint32_t seed;
    DataSpace<simDim> localCells;
};

} //namespace manipulators
} //namespace particles
} //namespace picongpu
