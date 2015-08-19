/**
 * Copyright 2013-2015 Rene Widera, Marco Garten, Benjamin Worpitz
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
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

#include "types.hpp"
#include "math/Vector.hpp"
#include "mappings/threads/ThreadCollective.hpp"
#include "nvidia/functors/Assign.hpp"
#include "memory/boxes/CachedBox.hpp"
#include "memory/dataTypes/Mask.hpp"

namespace gol
{
    //#############################################################################
    //! Functor returning if the cell is alive.
    //#############################################################################
    struct IsCellAlive
    {
        template<
            typename TCellType>
        HDINLINE bool operator()(
            TCellType const & cell) const
        {
            return (cell > 0);
        }
    };

    namespace kernel
    {
        //#############################################################################
        //! The evolution kernel.
        //#############################################################################
        class evolution
        {
        public:
            //-----------------------------------------------------------------------------
            //! \param rule The first 9 bits (0-8) represent the stay-alive rules, the next 9 bits (9-17) the new-born rules.
            //-----------------------------------------------------------------------------
            template<
                typename TAcc,
                class BoxReadOnly,
                class BoxWriteOnly,
                class Mapping>
            ALPAKA_FN_ACC void operator()(
                TAcc const & acc,
                BoxReadOnly const & buffRead,
                BoxWriteOnly const & buffWrite,
                std::uint32_t const & rule,
                Mapping const & mapper) const
            {
                static_assert(
                    alpaka::dim::Dim<TAcc>::value == 2u,
                    "evolution kernel can only be executed two-dimensionally!");

                using CellType = typename BoxReadOnly::ValueType;
                using BlockArea = PMacc::SuperCellDescription<
                    typename Mapping::SuperCellSize,
                    PMacc::math::CT::Int<1, 1>,
                    PMacc::math::CT::Int<1, 1>>;

                //----------
                // Calculate indices.

                // Get the block index.
                PMacc::DataSpace<DIM2> const blockIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc));
                // Get the SuperCell index relative to the whole grid in units of SuperCells. A block corresponds directly to one SuperCell but the mapping can be arbitrary.
                PMacc::DataSpace<DIM2> const gridSuperCellIdxSC(mapper.getSuperCellIndex(blockIdx));
                // Get the SuperCell index relative to the whole grid in unit of cells.
                PMacc::DataSpace<DIM2> const gridSuperCellIdxC(gridSuperCellIdxSC * Mapping::SuperCellSize::toRT());
                // Get the cell index relative to the super cell.
                PMacc::DataSpace<DIM2> const superCellCellIdxC(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc));
                // The cell index relative to the whole grid.
                PMacc::DataSpace<DIM2> const gridCellIdxC(gridSuperCellIdxC + superCellCellIdxC);

                //----------
                // Cache the data of the current block.

                // Create a collective operation (executed by all threads in the block).
                PMacc::ThreadCollective<BlockArea> collective(superCellCellIdxC);
                // The collective operation is a simple assignment ...
                PMacc::nvidia::functors::Assign const assign;
                // ... into a cache with the size of the block.
                auto cache(PMacc::CachedBox::create<0, CellType>(
                    acc,
                    BlockArea()));
                // The input is shifted to the position of the current block inside the input buffer.
                auto shiftedBuffRead(buffRead.shift(gridSuperCellIdxC));
                // Execute the collective operation ...
                collective(
                    assign,
                    cache,
                    shiftedBuffRead);
                // ... resulting in the values from the input buffer corresponding to the current block being cached.
                // Wait for the collective operation to be finished by all threads.
                acc.syncBlockThreads();

                //----------
                // Update the alive state.

                // The functor determining if a cell is alive.
                IsCellAlive const isCellAlive;

                // Count the number of living neighbors.
                std::uint32_t neighbors(0u);
                for(std::uint32_t i(1u); i < 9u; ++i)
                {
                    PMacc::DataSpace<DIM2> offsetIdxC(PMacc::Mask::getRelativeDirections<DIM2>(i));
                    neighbors += static_cast<std::uint32_t>(isCellAlive(cache(superCellCellIdxC + offsetIdxC)));
                }

                bool const isLife(isCellAlive(cache(superCellCellIdxC)));
                // The cell is alive after this step if:
                buffWrite(gridCellIdxC) = static_cast<CellType>(
                    // - it was alive before and the number of neighbors is in the stay-alive rule.
                    ((isLife) && ((1 << (neighbors)) & rule)) ||
                    // - it was dead before and the number of neighbors is in the new-born rule.
                    ((!isLife) && ((1 << (neighbors + 9)) & rule)) );
            }
        };

        //#############################################################################
        //! The random initialization kernel.
        //#############################################################################
        class randomInit
        {
        public:
            //-----------------------------------------------------------------------------
            //!
            //-----------------------------------------------------------------------------
            template<
                typename TAcc,
                class BoxWriteOnly,
                class Mapping>
            ALPAKA_FN_ACC void operator()(
                TAcc const & acc,
                BoxWriteOnly const & buffWrite,
                std::uint32_t const & seed,
                float const & fraction,
                Mapping const & mapper) const
            {
                static_assert(
                    alpaka::dim::Dim<TAcc>::value == 2u,
                    "randomInit kernel can only be executed two-dimensionally!");

                //----------
                // Calculate indices.

                // Get the block index.
                PMacc::DataSpace<DIM2> const blockIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc));
                // Get the SuperCell index relative to the whole grid in units of SuperCells. A block corresponds directly to one SuperCell but the mapping can be arbitrary.
                PMacc::DataSpace<DIM2> const gridSuperCellIdxSC(mapper.getSuperCellIndex(blockIdx));
                // Get the SuperCell index in unit of cells.
                PMacc::DataSpace<DIM2> const gridSuperCellIdxC(gridSuperCellIdxSC * Mapping::SuperCellSize::toRT());
                // Get the cell index relative to the super cell.
                PMacc::DataSpace<DIM2> const superCellCellIdxC(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc));
                // The cell index relative to the whole grid.
                PMacc::DataSpace<DIM2> const gridCellIdxC(gridSuperCellIdxC + superCellCellIdxC);
                // The extent of the whole grid in cells.
                PMacc::DataSpace<DIM2> const gridExtentC(mapper.getGridSuperCells() * Mapping::SuperCellSize::toRT());
                // Map the 2D index to an 1D index.
                uint32_t const gridCellIdxC1d(PMacc::DataSpaceOperations<DIM2>::map(gridExtentC, gridCellIdxC));

                /*std::cout << "blockIdx: " << blockIdx.toString() << std::endl;
                std::cout << "gridSuperCellIdxSC: " << gridSuperCellIdxSC.toString() << std::endl;
                std::cout << "gridSuperCellIdxC: " << gridSuperCellIdxC.toString() << std::endl;
                std::cout << "superCellCellIdxC: " << superCellCellIdxC.toString() << std::endl;
                std::cout << "gridCellIdxC: " << gridCellIdxC.toString() << std::endl;
                std::cout << "gridExtentC: " << gridExtentC.toString() << std::endl;
                std::cout << "gridCellIdxC1d: " << gridCellIdxC1d << std::endl;*/

                //----------
                // Generate random data.

                auto const gen(alpaka::rand::generator::createDefault(acc, seed, gridCellIdxC1d));
                auto const dist(alpaka::rand::distribution::createUniformReal<float>(acc));
                auto ufRng(std::bind(dist, gen));

                // Write 1 (alive) if uniform random number 0 <= rng < 1 is smaller than 'fraction'.
                buffWrite(gridCellIdxC) = (ufRng() <= fraction) ? 1u : 0u;
            }
        };
    }

    template<class MappingDesc>
    struct Evolution
    {
        MappingDesc mapping;
        uint32_t rule;

        Evolution(uint32_t rule) : rule(rule)
        {

        }

        void init(const MappingDesc & desc)
        {
            mapping = desc;
        }

        template<class DBox>
        void initEvolution(const DBox & writeBox, float fraction)
        {

            PMacc::AreaMapping<PMacc::CORE + PMacc::BORDER, MappingDesc> mapper(mapping);
            PMacc::GridController<DIM2> & gc(PMacc::Environment<DIM2>::get().GridController());
            uint32_t const seed(gc.getGlobalSize() + gc.getGlobalRank());

            kernel::randomInit kernel;

            __cudaKernel(
                kernel,
                alpaka::dim::DimInt<2u>,
                mapper.getGridDim(),
                MappingDesc::SuperCellSize::toRT())
            (writeBox,
                seed,
                fraction,
                mapper);
        }
        //-----------------------------------------------------------------------------
        //!
        //-----------------------------------------------------------------------------
        template<
            std::uint32_t TArea,
            class DBox>
        void run(
            DBox const & readBox,
            DBox const & writeBox)
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            kernel::evolution kernel;

            PMacc::AreaMapping<TArea, MappingDesc> mapper(mapping);
            __cudaKernel(
                kernel,
                alpaka::dim::DimInt<2u>,
                mapper.getGridDim(),
                MappingDesc::SuperCellSize::toRT())
            (readBox,
                writeBox,
                rule,
                mapper);
        }
    };
}

