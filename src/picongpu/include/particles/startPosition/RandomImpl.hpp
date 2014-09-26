/**
 * Copyright 2013-2014 Axel Huebl, Heiko Burau, Rene Widera
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
#include "particles/startPosition/MakroParticleCfg.hpp"
#include "nvidia/rng/RNG.hpp"
#include "nvidia/rng/methods/Xor.hpp"
#include "nvidia/rng/distributions/Uniform_float.hpp"

namespace picongpu
{
namespace particles
{
namespace startPosition
{

template<typename T_ParamClass, typename T_SpeciesName>
struct RandomImpl
{
    typedef T_ParamClass ParamClass;
    typedef T_SpeciesName SpeciesName;
    typedef typename SpeciesName::FrameType FrameType;

    HINLINE RandomImpl() : isInitialized(false)
    {
        mpi::SeedPerRank<simDim> seedPerRank;
        uint32_t seed = seedPerRank(GlobalSeed()(), FrameType::CommunicationTag);
        seed ^= POSITION_SEED;

        const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);
        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
        gpuGlobalCellOffset = subGrid.getLocalDomain().offset;
        localDomainSize = subGrid.getLocalDomain().size;

        gpuCellOffset.y() += numSlides * localDomainSize.y();

    }

    /** Distributes the initial particles uniformly random within the cell.
     *
     * @param rng a reference to an initialized, UNIFORM random number generator
     * @param totalNumParsPerCell the total number of particles to init for this cell
     * @param curParticle the number of this particle: [0, totalNumParsPerCell-1]
     * @return float3_X with components between [0.0, 1.0)
     */
    DINLINE floatD_X getPosition(const uint32_t curParticle)
    {
        floatD_X result;
        for (uint32_t i = 0; i < simDim; ++i)
            result[i] = rng();

        return result;
    }

    /** If the particles to initialize (numParsPerCell) end up with a
     *  related particle weighting (macroWeighting) below MIN_WEIGHTING,
     *  reduce the number of particles if possible to satisfy this condition.
     *
     * @param numParsPerCell the intendet number of particles for this cell
     * @param realElPerCell  the number of real electrons in this cell
     * @return macroWeighting the intended weighting per macro particle
     */
    template<typename T_CellSizeType>
    DINLINE MakroParticleCfg mapRealToMakroParticle(
                                                    const float_X realElPerCell,
                                                    const DataSpace<simDim>& globalCellOffset,
                                                    const T_CellSizeType&)
    {
        uint32_t numParsPerCell = ParamClass::numParticlesPerCell;

        if (!isInitialized)
        {
            const uint32_t cellIdx = DataSpaceOperations<simDim>::map(
                                                                      localDomainSize,
                                                                      globalCellOffset - gpuGlobalCellOffset);

            rng = nvrng::create(rngMethods::Xor(seed, cellIdx), rngDistributions::Uniform_float());
            isInitialized = true;
        }
        float_X macroWeighting = float_X(0.0);
        if (numParsPerCell > 0)
            macroWeighting = realElPerCell / float_X(numParsPerCell);

        while (macroWeighting < MIN_WEIGHTING &&
               numParsPerCell > 0)
        {
            --numParsPerCell;
            if (numParsPerCell > 0)
                macroWeighting = realElPerCell / float_X(numParsPerCell);
            else
                macroWeighting = float_X(0.0);
        }
        MakroParticleCfg makroParCfg;
        makroParCfg.weighting = macroWeighting;
        makroParCfg.numParticlesPerCell = numParsPerCell;

        return makroParCfg;
    }

protected:
    typedef PMacc::nvidia::rng::RNG<typename ParamClass::RngMethod, typename ParamClass::Distribution> RngType;
    RngType rng;
    bool isInitialized;
    uint32_t seed;
    DataSpace<simDim> gpuGlobalCellOffset;
    DataSpace<simDim> localDomainSize;

};

} //namespace particlesStartPosition
} //namespace particles
} //namespace picongpu
