/* Copyright 2013-2022 Felix Schmitt, Heiko Burau, Rene Widera, Sergei Bastrakov
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/dimensions/DataSpaceOperations.hpp"
#include "pmacc/types.hpp"

#include <cstdint>

namespace pmacc
{
    /** Mapping from block indices to supercells in the given area for alpaka kernels
     *
     * Adheres to the MapperConcept.
     *
     * @tparam T_BaseMapper Mapper which should be mapped to a one dimension domain
     */
    template<typename T_BaseMapper>
    class RangeMapping : T_BaseMapper
    {
    public:
        using BaseClass = T_BaseMapper;

        static constexpr uint32_t Dim = BaseClass::Dim;


        using SuperCellSize = typename BaseClass::SuperCellSize;

        HINLINE RangeMapping(BaseClass const& base, uint32_t begin, uint32_t end)
            : BaseClass(base)
            , baseGridDim(base.getGridDim())
            , beginIdx(begin)
            , endIdx(end)
        {
            // clamp range
            setRange(begin, end);
        }

        RangeMapping(RangeMapping const&) = default;

        /** Generate grid dimension information for alpaka kernel calls
         *
         * A kernel using this mapping must use exactly the returned number of blocks.
         * The range will automatically clamped to fit into the N-dimensional block range of the wrapped mapper.
         *
         * @return number of blocks in a grid
         */
        HINLINE int getGridDim()
        {
            /* Update the base grid size in case it is change over time if the mapper instance is used
             * multiple times e.g. StridingMapping
             */
            baseGridDim = BaseClass::getGridDim();
            setRange(beginIdx, endIdx);
            return size();
        }

        HINLINE void setRange(uint32_t begin, uint32_t end)
        {
            /* Update the base grid size in case it is change over time if the mapper instance is used
             * multiple times e.g. StridingMapping
             */
            baseGridDim = BaseClass::getGridDim();
            auto clampedEnd = end == uint32_t(-1) ? static_cast<uint32_t>(baseGridDim.productOfComponents()) : end;
            beginIdx = std::min(beginIdx, clampedEnd);
            endIdx = std::min(endIdx, clampedEnd);
        }

        /** Number of elements described by the range
         *
         * @return  size of the range
         */
        HDINLINE uint32_t size() const
        {
            return endIdx - beginIdx;
        }

        HDINLINE uint32_t begin() const
        {
            return beginIdx;
        }

        HDINLINE uint32_t end() const
        {
            return endIdx;
        }

        HDINLINE uint32_t last() const
        {
            return endIdx == 0u ? 0u : endIdx - 1u;
        }

        HDINLINE DataSpace<Dim> beginND() const
        {
            return DataSpaceOperations<Dim>::map(baseGridDim, beginIdx);
        }

        HDINLINE DataSpace<Dim> endND() const
        {
            return DataSpaceOperations<Dim>::map(baseGridDim, endIdx);
        }

        HDINLINE DataSpace<Dim> lastND() const
        {
            return DataSpaceOperations<Dim>::map(baseGridDim, last());
        }

        /** Return index of a supercell to be processed by the given alpaka block
         *
         * @param blockIdx alpaka block index
         * @return mapped SuperCell index including guards
         */
        HDINLINE DataSpace<Dim> getSuperCellIndex(const DataSpace<Dim>& blockIdx) const
        {
            auto const blockIdxNDim = DataSpaceOperations<Dim>::map(baseGridDim, beginIdx + blockIdx.x());
            return BaseClass::getSuperCellIndex(blockIdxNDim);
        }

    private:
        uint32_t beginIdx = 0u;
        uint32_t endIdx = uint32_t(-1);
        DataSpace<Dim> baseGridDim;
    };

    /** Construct an area mapper instance for the given area and description
     *
     * @tparam T_area area, a value from type::AreaType or a sum of such values
     * @tparam T_MappingDescription mapping description type
     *
     * @param mappingDescription mapping description
     */
    template<typename T_BaseMapper>
    HINLINE auto makeRangeMapper(T_BaseMapper baseMapper, uint32_t begin = 0u, uint32_t end = uint32_t(-1))
    {
        return RangeMapping(baseMapper, begin, end);
    }

} // namespace pmacc
