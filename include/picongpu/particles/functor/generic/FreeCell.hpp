/* Copyright 2013-2017 Rene Widera, Axel Huebl
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

#include "picongpu/simulation_defines.hpp"
#include "picongpu/particles/functor/generic/Free.def"

#include <utility>
#include <type_traits>

namespace picongpu
{
namespace particles
{
namespace functor
{
namespace generic
{
namespace acc
{
    /** wrapper for the user functor on the accelerator
     *
     * @tparam T_Functor user defined functor
     */
    template< typename T_Functor >
    struct FreeCell : private T_Functor
    {
        //! type of the user functor
        using Functor = T_Functor;

        //! store user functor instance
        HDINLINE FreeCell(
            Functor const & functor,
            DataSpace< simDim > const & gpuCellOffsetToTotalOrigin
        ) :
            Functor( functor ),
            m_superCellToLocalOriginCellOffset( superCellToLocalOriginCellOffset )
        {
        }

        /** execute the user functor
         *
         * @tparam T_Args type of the arguments passed to the user functor
         *
         * @param args arguments passed to the user functor
         *
         * @{
         */
        template<
            typename T_Acc,
            typename T_Particle,
            typename ... T_Args
        >
        HDINLINE
        void operator( )(
            T_Acc const &,
            T_Particle && particle,
            T_Args && ... args
        )
        {
            DataSpace< simDim > const cellInSuperCell(
                DataSpaceOperations< simdDim >::template map< SuperCellSize > ( particle[ localCellIdx_ ] )
            );
            Functor::operator( )(
                m_superCellToLocalOriginCellOffset + cellInSuperCell,
                particle,
                args ...
            );
        }

        template<
            typename T_Acc,
            typename T_Particle
        >
        HDINLINE
        bool operator( )(
            T_Acc const &,
            T_Particle const & particle
        )
        {
            DataSpace< simDim > const cellInSuperCell(
                DataSpaceOperations< simdDim >::template map< SuperCellSize > ( particle[ localCellIdx_ ] )
            );
            return Functor::operator( )(
                m_superCellToLocalOriginCellOffset + cellInSuperCell
                particle
            );
        }
        //!@}

         DataSpace< simDim > const m_superCellToLocalOriginCellOffset;
    };
} // namespace acc

    template< typename T_Functor >
    struct FreeCell : protected T_Functor
    {

        using Functor = T_Functor;

        template< typename T_SpeciesType >
        struct apply
        {
            using type = FreeCell;
        };

        /** constructor
         *
         * This constructor is only compiled if the user functor has
         * a host side constructor with one (uint32_t) argument.
         *
         * @tparam DeferFunctor is used to defer the functor type evaluation to enable/disable
         *                      the constructor
         * @param currentStep current simulation time step
         * @param is used to enable/disable the constructor (do not pass any value to this parameter)
         */
        template< typename DeferFunctor = Functor >
        HINLINE FreeCell(
            uint32_t currentStep,
            typename std::enable_if<
                std::is_constructible<
                    DeferFunctor,
                    uint32_t
                >::value
            >::type* = 0
        ) : Functor( currentStep )
        {
            hostInit( uint32_t currentStep );
        }

        /** constructor
         *
         * This constructor is only compiled if the user functor has a default constructor.
         *
         * @tparam DeferFunctor is used to defer the functor type evaluation to enable/disable
         *                      the constructor
         * @param current simulation time step
         * @param is used to enable/disable the constructor (do not pass any value to this parameter)
         */
        template< typename DeferFunctor = Functor >
        HINLINE FreeCell(
            uint32_t currentStep,
            typename std::enable_if<
                std::is_constructible< DeferFunctor >::value
            >::type* = 0
        ) : Functor( )
        {
            hostInit( uint32_t currentStep );
        }

        /** create device functor
         *
         * @tparam T_WorkerCfg pmacc::mappings::threads::WorkerCfg, configuration of the worker
         * @tparam T_Acc alpaka accelerator type
         *
         * @param alpaka accelerator
         * @param offset (in supercells, without any guards) to the
         *         origin of the local domain
         * @param configuration of the worker
         */
        template<
            typename T_WorkerCfg,
            typename T_Acc
        >
        HDINLINE acc::FreeCell< Functor >
        operator()(
            T_Acc const & acc,
            DataSpace< simDim > const & localSupercellOffset,
            T_WorkerCfg const &
        )
        {
            uint32_t const superCellToLocalOriginCellOffset(
                localSupercellOffset * SuperCellSize::toRT( )
            );

            return acc::FreeCell< Functor >(
                *static_cast< Functor * >( this ),
                gpuCellOffsetToTotalOrigin + superCellToLocalOriginCellOffset
            );
        }

    private:

        HINLINE
        void
        hostInit( uint32_t currentStep )
        {
            uint32_t const numSlides = MovingWindow::getInstance( ).getSlideCounter( currentStep );
            SubGrid< simDim > const & subGrid = Environment< simDim >::get( ).SubGrid( );
            DataSpace< simDim > localCells = subGrid.getLocalDomain( ).size;
            gpuCellOffsetToTotalOrigin = subGrid.getLocalDomain( ).offset;
            gpuCellOffsetToTotalOrigin.y( ) += numSlides * localCells.y( );
        }

        DataSpace< simDim > gpuCellOffsetToTotalOrigin;
    };

} // namespace generic
} // namespace functor
} // namespace particles
} // namespace picongpu
