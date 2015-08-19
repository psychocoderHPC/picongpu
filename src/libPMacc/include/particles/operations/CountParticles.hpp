/**
 * Copyright 2013-2015 Rene Widera, Benjamin Worpitz
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "types.h"
#include "memory/buffers/GridBuffer.hpp"
#include "mappings/kernel/AreaMapping.hpp"

#include "particles/particleFilter/FilterFactory.hpp"
#include "particles/particleFilter/PositionFilter.hpp"

namespace PMacc
{

/* count particles in an area
 * is not optimized, it checks any particle position if its really a particle
 */
struct KernelCountParticles
{
template<
    typename T_Acc,
    typename PBox,
    typename Filter,
    typename Mapping>
ALPAKA_FN_ACC void operator()(
    T_Acc const & acc,
    PBox const & pb,
    uint64_cu* gCounter,
    Filter filter,
    Mapping const & mapper) const
{
    typedef typename PBox::FrameType FRAME;
    const uint32_t Dim = Mapping::Dim;

    DataSpace<Dim> const blockIndex(alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc));
    DataSpace<Dim> const threadIndex(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc));

    auto frame(alpaka::block::shared::allocVar<FRAME *>(acc));
    auto isValid(alpaka::block::shared::allocVar<bool>(acc));
    auto counter(alpaka::block::shared::allocVar<int>(acc));
    auto particlesInSuperCell(alpaka::block::shared::allocVar<lcellId_t>(acc));


    acc.syncBlockThreads(); /*wait that all shared memory is initialized*/

    typedef typename Mapping::SuperCellSize SuperCellSize;

    const int linearThreadIdx = DataSpaceOperations<Dim>::template map<SuperCellSize > (threadIndex);
    const DataSpace<Dim> superCellIdx(mapper.getSuperCellIndex(DataSpace<Dim > (blockIndex)));

    if (linearThreadIdx == 0)
    {
        frame = &(pb.getLastFrame(superCellIdx, isValid));
        particlesInSuperCell = pb.getSuperCell(superCellIdx).getSizeLastFrame();
        counter = 0;
    }
    acc.syncBlockThreads();
    if (!isValid)
        return; //end kernel if we have no frames
    filter.setSuperCellPosition((superCellIdx - mapper.getGuardingSuperCells()) * mapper.getSuperCellSize());
    while (isValid)
    {
        if (linearThreadIdx < particlesInSuperCell)
        {
            if (filter(*frame, linearThreadIdx))
                alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(acc, &counter, 1);
        }
        acc.syncBlockThreads();
        if (linearThreadIdx == 0)
        {
            frame = &(pb.getPreviousFrame(*frame, isValid));
            particlesInSuperCell = math::CT::volume<SuperCellSize>::type::value;
        }
        acc.syncBlockThreads();
    }

    acc.syncBlockThreads();
    if (linearThreadIdx == 0)
    {
        alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(acc, gCounter, (uint64_cu) counter);
    }
}
};

struct CountParticles
{

    /** Get particle count
     *
     * @tparam AREA area were particles are counted (CORE, BORDER, GUARD)
     *
     * @param buffer source particle buffer
     * @param cellDescription instance of MappingDesction
     * @param filter filter instance which must inherit from PositionFilter
     * @return number of particles in defined area
     */
    template<uint32_t AREA, class PBuffer, class Filter, class CellDesc>
    static uint64_cu countOnDevice(PBuffer& buffer, CellDesc cellDescription, Filter filter)
    {
        GridBuffer<uint64_cu, DIM1> counter(DataSpace<DIM1>(1));

        AreaMapping<AREA, CellDesc> mapper(cellDescription);

        KernelCountParticles kernelCountParticles;

        __cudaKernel(
            kernelCountParticles,
            alpaka::dim::DimInt<3u>,
            mapper.getGridDim(),
            CellDesc::SuperCellSize::toRT())(
                buffer.getDeviceParticlesBox(),
                counter.getDeviceBuffer().getBasePointer(),
                filter,
                mapper);

        counter.deviceToHost();
        return *(counter.getHostBuffer().getDataBox());
    }

    /** Get particle count
     *
     * @param buffer source particle buffer
     * @param cellDescription instance of MappingDesction
     * @param filter filter instance which must inherit from PositionFilter
     * @return number of particles in defined area
     */
    template< class PBuffer, class Filter, class CellDesc>
    static uint64_cu countOnDevice(PBuffer& buffer, CellDesc cellDescription, Filter filter)
    {
        return PMacc::CountParticles::countOnDevice < CORE + BORDER + GUARD > (buffer, cellDescription, filter);
    }

    /** Get particle count
     *
     * @tparam AREA area were particles are counted (CORE, BORDER, GUARD)
     *
     * @param buffer source particle buffer
     * @param cellDescription instance of MappingDesction
     * @param origin local cell position (can be negative)
     * @param size local size in cells for checked volume
     * @return number of particles in defined area
     */
    template<uint32_t AREA, class PBuffer, class CellDesc, class Space>
    static uint64_cu countOnDevice(PBuffer& buffer, CellDesc cellDescription, const Space& origin, const Space& size)
    {
        typedef bmpl::vector< typename GetPositionFilter<Space::Dim>::type > usedFilters;
        typedef typename FilterFactory<usedFilters>::FilterType MyParticleFilter;
        MyParticleFilter filter;
        filter.setStatus(true); /*activate filter pipeline*/
        filter.setWindowPosition(origin, size);
        return PMacc::CountParticles::countOnDevice<AREA>(buffer, cellDescription, filter);
    }

    /** Get particle count
     *
     * @param buffer source particle buffer
     * @param cellDescription instance of MappingDesction
     * @param origin local cell position (can be negative)
     * @param size local size in cells for checked volume
     * @return number of particles in defined area
     */
    template< class PBuffer, class Filter, class CellDesc, class Space>
    static uint64_cu countOnDevice(PBuffer& buffer, CellDesc cellDescription, const Space& origin, const Space& size)
    {
        return PMacc::CountParticles::countOnDevice < CORE + BORDER + GUARD > (buffer, cellDescription, origin, size);
    }

};

} //namespace PMacc
