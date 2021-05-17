/* Copyright 2017-2021 Rene Widera
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

#include "pmacc/types.hpp"


namespace pmacc
{
    namespace memory
    {
        template<typename T_Type, typename T_IdxConfig>
        struct CtxVar;
    } // namespace memory
    namespace mappings
    {
        namespace threads
        {
            //! Hold current index within a lockstep-domain
            struct DomainIdx
            {
                /** Constructor
                 *
                 * @param domElemIndex linear index within the domain
                 * @param workerElemIndex virtual workers linear index of the work item
                 */
                HDINLINE DomainIdx(uint32_t const domElemIndex, uint32_t const workerElemIndex)
                    : domElemIdx(domElemIndex)
                    , workerElemIdx(workerElemIndex)
                {
                }

                /** Get linear index
                 *
                 * @return range [0,domain size)
                 */
                HDINLINE uint32_t lIdx() const
                {
                    return domElemIdx;
                }

                template<typename T_Type, typename T_IdxConfig>
                friend class memory::CtxVar;

            private:
                //! virtual workers linear index of the work item
                uint32_t const workerElemIdx;
                //! linear index within the domain
                uint32_t const domElemIdx;
            };

        } // namespace threads
    } // namespace mappings
} // namespace pmacc
