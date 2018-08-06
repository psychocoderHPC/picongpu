/* Copyright 2018 Rene Widera
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
#include "picongpu/particles/manipulators/generic/FreeRng.def"
#include "picongpu/particles/functor/misc/Rng.hpp"
#include "picongpu/particles/functor/User.hpp"

#include "pmacc/math/vector/compile-time/Vector.hpp"

#include <utility>
#include <type_traits>
#include <string>


namespace picongpu
{
namespace particles
{
namespace collision
{
namespace generic
{
namespace acc
{

    template<
        typename T_ParBox
    >
    struct MyParIter
    {
        using FramePtr = typename T_ParBox::FramePtr;

        HDINLINE MyParIter(
            T_ParBox const & parBox,
            DataSpace< simDim > const & superCellIdx
        ) :
            m_parBox( parBox ),
            firstFrame( parBox.getFirstFrame( superCellIdx ) ),
            numParPerCell( parBox.getSuperCell( superCellIdx ).getNumParticles() )
        {

        }

        HDINLINE
        typename T_ParBox::FrameType::ParticleType
        operator[]( uint32_t const idx )
        {
            constexpr uint32_t frameSize = pmacc::math::CT::volume< typename T_ParBox::FrameType::SuperCellSize >::type::value;
            auto tmpFrame = firstFrame;
            uint32_t skipFrames = idx / frameSize;
            for(uint32_t i = 0; i < skipFrames; ++i)
                tmpFrame = m_parBox.getNextFrame( tmpFrame );
            return tmpFrame[ idx % frameSize ];
        }

        T_ParBox m_parBox;
        FramePtr firstFrame;
        uint32_t numParPerCell;
    };

    template<
        typename T_Functor,
        typename T_RngType,
        typename T_ParticleIter
    >
    struct FreeRng : private T_Functor
    {

        using Functor = T_Functor;
        using RngType = T_RngType;

        HDINLINE FreeRng(
            Functor const & functor,
            RngType const & rng,
            T_ParticleIter const & parIter
        ) :
            T_Functor( functor ), m_rng( rng ), m_parIter( parIter )
        {
        }

        /** call user functor
         *
         * The random number generator is initialized with the first call.
         *
         * @tparam T_Particle type of the particle to manipulate
         * @tparam T_Args type of the arguments passed to the user functor
         * @tparam T_Acc alpaka accelerator type
         *
         * @param alpaka accelerator
         * @param particle particle which is given to the user functor
         * @return void is used to enable the operator if the user functor except two arguments
         */
        template<
            typename T_Particle,
            typename ... T_Args,
            typename T_Acc
        >
        HDINLINE
        void operator()(
            T_Acc const &,
            T_Particle& particle,
            T_Args && ... args
        )
        {
            namespace nvrng = nvidia::rng;

            Functor::operator()(
                m_rng,
                m_parIter,
                particle,
                args ...
            );
        }

    private:

        RngType m_rng;
        T_ParticleIter m_parIter;
    };
} // namespace acc

    template<
        typename T_Functor,
        typename T_Distribution,
        typename T_PairSpeciesType
    >
    struct FreeRng :
    protected functor::User< T_Functor >,
        private picongpu::particles::functor::misc::Rng<
            T_Distribution
        >
    {
        using RngGenerator = picongpu::particles::functor::misc::Rng<
            T_Distribution
        >;

        using RngType = typename RngGenerator::RandomGen;

        using Functor = functor::User< T_Functor >;
        using Distribution = T_Distribution;

        using PairSpeciesType = pmacc::particles::compileTime::FindByNameOrType_t<
            VectorAllSpecies,
            T_PairSpeciesType
        >;
        using PairFrameType = typename  PairSpeciesType::FrameType;
        using ParBoxType = typename PairSpeciesType::ParticlesBoxType;
        memory::Array< ParBoxType, 1u > parBox;

        /** constructor
         *
         * @param currentStep current simulation time step
         */
        HINLINE FreeRng( uint32_t currentStep ) :
            Functor( currentStep ),
            RngGenerator( currentStep )
        {
            DataConnector &dc = Environment<>::get().DataConnector();
            auto speciesPtr = dc.get< PairSpeciesType >(
                PairFrameType::getName(),
                true
            );
            parBox[0] = speciesPtr->getDeviceParticlesBox();

            dc.releaseData( PairFrameType::getName() );
        }

        /** create functor for the accelerator
         *
         * @tparam T_WorkerCfg pmacc::mappings::threads::WorkerCfg, configuration of the worker
         * @tparam T_Acc alpaka accelerator type
         *
         * @param alpaka accelerator
         * @param localSupercellOffset offset (in superCells, without any guards) relative
         *                        to the origin of the local domain
         * @param workerCfg configuration of the worker
         */
        template<
            typename T_WorkerCfg,
            typename T_Acc
        >
        HDINLINE auto
        operator()(
            T_Acc const & acc,
            DataSpace< simDim > const & localSupercellOffset,
            T_WorkerCfg const & workerCfg
        ) const
        -> acc::FreeRng<
            Functor,
            RngType,
            acc::MyParIter< ParBoxType >
        >
        {
            RngType const rng = ( *static_cast< RngGenerator const * >( this ) )(
                acc,
                localSupercellOffset,
                workerCfg
            );

            auto superCellIdx = localSupercellOffset + GuardSize::toRT();

            return acc::FreeRng<
                Functor,
                RngType,
                acc::MyParIter< ParBoxType >
            >(
                *static_cast< Functor const * >( this ),
                rng,
                acc::MyParIter< ParBoxType >(
                    parBox[0],
                    superCellIdx
                )
            );
        }

        static
        HINLINE std::string
        getName( )
        {
            // we provide the name from the param class
            return Functor::name;
        }
    };

} // namespace generic
} // namespace collision
} // namespace particles
} // namespace picongpu
