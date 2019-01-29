/* Copyright 2013-2018 Rene Widera, Axel Huebl
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
#include "picongpu/particles/manipulators/generic/Free.def"
#include "picongpu/particles/functor/User.hpp"

#include <utility>
#include <type_traits>

namespace picongpu
{
namespace particles
{
namespace pusher
{
namespace acc
{

    template<class PushAlgo, class T_Field2ParticleInterpolation>
    struct PushParticlePerFrame
    {

        template<class T_Particle, class BoxB, class BoxE, typename T_Acc >
        DINLINE void operator()(
            T_Acc const & acc,
            T_Particle& particle,
            BoxB& bBox,
            BoxE& eBox,
            uint32_t const currentStep,
            int& mustShift
        )
        {

            using TVec = SuperCellSize;
            using Field2ParticleInterpolation = T_Field2ParticleInterpolation;

            using BType = typename BoxB::ValueType;
            using EType = typename BoxE::ValueType;

            floatD_X pos = particle[position_];
            const int particleCellIdx = particle[localCellIdx_];

            DataSpace<TVec::dim> localCell(DataSpaceOperations<TVec::dim>::template map<TVec > (particleCellIdx));

            const traits::FieldPosition<typename fields::Solver::NummericalCellType, FieldE> fieldPosE;
            const traits::FieldPosition<typename fields::Solver::NummericalCellType, FieldB> fieldPosB;

            auto functorEfield = CreateInterpolationForPusher<Field2ParticleInterpolation>()( eBox.shift(localCell).toCursor(), fieldPosE() );
            auto functorBfield = CreateInterpolationForPusher<Field2ParticleInterpolation>()( bBox.shift(localCell).toCursor(), fieldPosB() );

            /** @todo this functor should only manipulate the momentum and all changes
             *        in position and cell below need to go into a separate kernel
             */
            PushAlgo push;
            push(
                 functorBfield,
                 functorEfield,
                 particle,
                 pos,
                 currentStep
            );

            DataSpace<simDim> dir;
            for (uint32_t i = 0; i < simDim; ++i)
            {
                /* ATTENTION we must handle float rounding errors
                 * pos in range [-1;2)
                 *
                 * If pos is negative and very near to 0 (e.g. pos < -1e-8)
                 * and we move pos with pos+=1.0 back to normal in cell postion
                 * we get a rounding error and pos is assigned to 1. This breaks
                 * our in cell definition range [0,1)
                 *
                 * if pos negativ moveDir is set to -1
                 * if pos positive and >1 moveDir is set to +1
                 * 0 (zero) if particle stays in cell
                 */
                float_X moveDir = math::floor(pos[i]);
                /* shift pos back to cell range [0;1)*/
                pos[i] -= moveDir;
                /* check for rounding errors and correct them
                 * if position now is 1 we have a rounding error
                 *
                 * We correct moveDir that we not have left the cell
                 */
                const float_X valueCorrector = math::floor(pos[i]);
                /* One has also to correct moveDir for the following reason:
                 * Imagine a new particle moves to -1e-20, leaving the cell to the left,
                 * setting moveDir to -1.
                 * The new in-cell position will be -1e-20 + 1.0,
                 * which can flip to 1.0 (wrong value).
                 * We move the particle back to the old cell at position 0.0 and
                 * moveDir has to be corrected back, too (add +1 again).*/
                moveDir += valueCorrector;
                /* If we have corrected moveDir we must set pos to 0 */
                pos[i] -= valueCorrector;
                dir[i] = precisionCast<int>(moveDir);
            }
            particle[position_] = pos;

            /* new local cell position after particle move
             * can be out of supercell
             */
            localCell += dir;

            /* ATTENTION ATTENTION we cast to unsigned, this means that a negative
             * direction is know a very very big number, than we compare with supercell!
             *
             * if particle is inside of the supercell the **unsigned** representation
             * of dir is always >= size of the supercell
             */
            for (uint32_t i = 0; i < simDim; ++i)
                dir[i] *= precisionCast<uint32_t>(localCell[i]) >= precisionCast<uint32_t>(TVec::toRT()[i]) ? 1 : 0;

            /* if partice is outside of the supercell we use mod to
             * set particle at cell supercellSize to 1
             * and partticle at cell -1 to supercellSize-1
             * % (mod) can't use with negativ numbers, we add one supercellSize to hide this
             *
            localCell.x() = (localCell.x() + TVec::x) % TVec::x;
            localCell.y() = (localCell.y() + TVec::y) % TVec::y;
            localCell.z() = (localCell.z() + TVec::z) % TVec::z;
             */

            /*dir is only +1 or -1 if particle is outside of supercell
             * y=cell-(dir*superCell_size)
             * y=0 if dir==-1
             * y=superCell_size if dir==+1
             * for dir 0 localCel is not changed
             */
            localCell -= (dir * TVec::toRT());
            /*calculate one dimensional cell index*/
            particle[localCellIdx_] = DataSpaceOperations<TVec::dim>::template map<TVec > (localCell);

            /* [ dir + int(dir < 0)*3 ] == [ (dir + 3) %3 = y ]
             * but without modulo
             * y=0 for dir = 0
             * y=1 for dir = 1
             * y=2 for dir = -1
             */
            int direction = 1;
            uint32_t exchangeType = 1; // see inlcude/pmacc/types.h for RIGHT, BOTTOM and BACK
            for (uint32_t i = 0; i < simDim; ++i)
            {
                direction += (dir[i] == -1 ? 2 : dir[i]) * exchangeType;
                exchangeType *= 3; // =3^i (1=RIGHT, 3=BOTTOM; 9=BACK)
            }

            particle[multiMask_] = direction;

            /* set our tuning flag if minimal one particle leave the supercell
             * This flag is needed for later fast shift of particles only if needed
             */
            if (direction >= 2)
            {
                /* if we did not use atomic we would get a WAW error */
                mustShift = 1;
            }
        }
    };

    template<
        typename T_WorkerCfg,
        typename T_DataDomain,
        typename T_CacheB,
        typename T_CacheE,
    >
    struct Pusher
    {
        T_CacheB m_cacheB;
        T_CacheE m_cacheE;

        uint32_t const m_currentStep;
        int m_mustShift = 0;

        //! store user manipulators instance
        HDINLINE Free(
            T_WorkerCfg const &,
            uint32_t const currentStep,
            T_CacheB const & cacheB,
            T_CacheE const & cacheE
        ) :
            m_workerCfg( workerCfg ),
            m_currentStep( currentStep ),
            m_cacheB( cacheB ),
            m_cacheE( cacheE ),
        {
        }

        /** execute the user manipulator functor
         *
         * @tparam T_Args type of the arguments passed to the user manipulator functor
         *
         * @param args arguments passed to the user functor
         */
        template<
            typename T_Acc,
            typename T_Particle >
        HDINLINE
        void operator( )(
            T_Acc const &,
            T_Particle && ... particle
        )
        {
            using FrameType = typename T_Particle::FrameType;
            using PusherAlias = typename GetFlagType<FrameType,particlePusher<> >::type;
            using ParticlePush = typename pmacc::traits::Resolve<PusherAlias>::type;

            using InterpolationScheme = typename pmacc::traits::Resolve<
                typename GetFlagType<
                    FrameType,
                    interpolation< >
                >::type
            >::type;

            using FrameSolver = PushParticlePerFrame<
                ParticlePush,
                MappingDesc::SuperCellSize,
                InterpolationScheme
            >;

        }

        template<
            typename T_Acc,
            typename T_WorkerCfg,
            typename T_ParticlesBox
        >
        void
        finalize(
            T_Acc const &,
            DataSpace< simDim > const & superCellOffset,
            T_WorkerCfg const & workerCfg,
            T_ParticlesBox const &
        ) const
        {
            PMACC_SMEM(
                acc,
                mustShift,
                int
            );
            ForEachIdx<
                IdxConfig<
                    1,
                    numWorkers
                >
            > onlyMaster{ workerIdx };

            onlyMaster(
                [&](
                    uint32_t const,
                    uint32_t const
                )
                {
                    mustShift = 0;
                }
            );
            __syncthreads();
            nvidia::atomicAllExch(acc, &mustShift, 1, ::alpaka::hierarchy::Threads{});
            __syncthreads();
            onlyMaster(
                [&](
                    uint32_t const,
                    uint32_t const
                )
                {
                    /* set in SuperCell the mustShift flag which is an optimization
                     * for shift particles (pmacc::KernelShiftParticles)
                     */
                    if( m_mustShift == 1 )
                    {
                        m_superCellPtr->setMustShift( true );
                    }
                }
            );

        }
    };
} // namespace acc

    template<
        typename T_SpeciesType
    >
    struct Pusher
    {
        typename FieldE::DataBoxType fieldE;
        typename FieldB::DataBoxType fieldB;

        uint32_t const m_currentStep;

        /** constructor
         *
         * @param currentStep current simulation time step
         */
        HINLINE Pusher( uint32_t const currentStep ) : m_currentStep( currentStep )
        {
            DataConnector & dc = Environment< >::get( ).DataConnector( );

            auto fieldEPtr = dc.get< FieldE >(
                FieldE::getName(),
                true
            );
            fieldE = fieldEPtr->getDeviceDataBox();

            auto fieldBPtr = dc.get< FieldB >(
                FieldB::getName(),
                true
            );
            fieldB = fieldBPtr->getDeviceDataBox();

              using SpeciesType = pmacc::particles::compileTime::FindByNameOrType_t<
                VectorAllSpecies,
                T_SpeciesType
            >;
            using FrameType = typename SpeciesType::FrameType;

            DataConnector &dc = Environment<>::get().DataConnector();
            auto speciesPtr = dc.get< SpeciesType >(
                FrameType::getName(),
                true
            );


            dc.releaseData( FieldE::getName() );
            dc.releaseData( FieldB::getName() );
        }

        /** create device manipulator functor
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
            typename T_Acc,
            typename T_WorkerCfg,
            typename T_ParticlesBox
        >
        HDINLINE acc::Pusher< Functor >
        operator()(
            T_Acc const &,
            DataSpace< simDim > const & superCellOffset,
            T_WorkerCfg const & workerCfg,
            T_ParticlesBox const &
        ) const
        {
            // adjust interpolation area in particle pusher to allow sub-sampling pushes
            using LowerMargin = typename GetLowerMarginPusher< T_SpeciesType >::type;
            using UpperMargin = typename GetUpperMarginPusher< T_SpeciesType >::type;

            using BlockArea = SuperCellDescription<
                typename MappingDesc::SuperCellSize,
                LowerMargin,
                UpperMargin
            >;

            auto cachedB = CachedBox::create<
                0,
                typename T_BBox::ValueType
            >(
                acc,
                T_DataDomain( )
            );
            auto cachedE = CachedBox::create<
                1,
                typename T_EBox::ValueType
            >(
                acc,
                T_DataDomain( )
            );

            nvidia::functors::Assign assign;
            ThreadCollective<
                T_DataDomain,
                numWorkers
            > collective{ workerIdx };

            auto fieldBBlock = fieldB.shift( superCellOffset + GuardSize::toRT() );
            collective(
                acc,
                assign,
                cachedB,
                fieldBBlock
            );

            auto fieldEBlock = fieldE.shift( superCellOffset + GuardSize::toRT() );
            collective(
                acc,
                assign,
                cachedE,
                fieldEBlock
            );

            __syncthreads();

            return acc::Pusher<
                workerCfg,
                BlockArea,
                decltype( std::cachedB ),
                decltype( std::cachedE )
            >(
                m_currentStep,
                cachedB,
                cachedE
            );
        }

        //! get the name of the functor
        static
        HINLINE std::string
        getName( )
        {
            // we provide the name from the param class
            return Functor::name;
        }

    };

} // namespace pusher
} // namespace particles
} // namespace picongpu
