/* Copyright 2013-2017 Heiko Burau, Rene Widera, Benjamin Worpitz
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

#include "mappings/threads/ForEachIdx.hpp"
#include "mappings/threads/IdxConfig.hpp"
#include "dimensions/SuperCellDescription.hpp"
#include "dimensions/DataSpaceOperations.hpp"
#include "dimensions/DataSpace.hpp"
#include "pmacc_types.hpp"

namespace PMacc
{

template<
    typename T_BlockArea,
    uint32_t T_workerSize = math::CT::volume<
        typename T_BlockArea::SuperCellSize
    >::type::value
>
class ThreadCollective
{
private:
    using SuperCellSize = typename T_BlockArea::SuperCellSize;
    using FullSuperCellSize = typename T_BlockArea::FullSuperCellSize;
    using OffsetOrigin = typename T_BlockArea::OffsetOrigin;

    static constexpr uint32_t workerSize = T_workerSize;
    static constexpr uint32_t dim = T_BlockArea::Dim;

    const PMACC_ALIGN( m_workerIdx, uint32_t );

public:

    DINLINE ThreadCollective( uint32_t const workerIdx ) :
        m_workerIdx( workerIdx )
    {
    }

    DINLINE ThreadCollective( DataSpace< dim > const & workerIdx ) :
        m_workerIdx( DataSpaceOperations< dim >::template map< SuperCellSize >( workerIdx ) )
    {
    }

    template<
        typename T_Functor,
        typename ... T_Args
    >
    DINLINE void operator()(
        T_Functor & functor,
        T_Args && ... args
    )
    {
        mappings::threads::ForEachIdx<
            mappings::threads::IdxConfig<
                math::CT::volume<FullSuperCellSize>::type::value,
                workerSize
            >
        >{ m_workerIdx }(
            [&]( uint32_t const linearIdx, uint32_t const )
            {
                DataSpace< dim > const pos(
                    DataSpaceOperations< dim >::template map< FullSuperCellSize >( linearIdx ) -
                    OffsetOrigin::toRT( )
                );
                functor( args(pos) ... );
            }
        );
    }
};

}//namespace PMacc
