/* Copyright 2017-2022 Axel Huebl, Rene Widera, Sergei Bastrakov
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

#include "pmacc/Environment.hpp"
#include "pmacc/attribute/Constexpr.hpp"
#include "pmacc/lockstep.hpp"
#include "pmacc/lockstep/Config.hpp"
#include "pmacc/lockstep/ForEach.hpp"
#include "pmacc/mappings/kernel/AreaMapping.hpp"
#include "pmacc/particles/frame_types.hpp"

#include <cstdint>
#include <type_traits>
#include <utility>


namespace pmacc
{
    namespace particles
    {
        namespace algorithm
        {
            namespace acc
            {
                struct Forward
                {
                    template<typename T_ParBox>
                    static DINLINE auto getFirst(T_ParBox& pb, DataSpace<T_ParBox::Dim> const& superCellIdx)
                    {
                        return pb.getFirstFrame(superCellIdx);
                    }

                    template<typename T_ParBox>
                    static DINLINE auto next(T_ParBox& pb, typename T_ParBox::FramePtr const& framePtr)
                    {
                        return pb.getNextFrame(framePtr);
                    }
                };

                struct Backward
                {
                    template<typename T_ParBox>
                    static DINLINE auto getFirst(T_ParBox& pb, DataSpace<T_ParBox::Dim> const& superCellIdx)
                    {
                        return pb.getLastFrame(superCellIdx);
                    }

                    template<typename T_ParBox>
                    static DINLINE auto next(T_ParBox& pb, typename T_ParBox::FramePtr const& framePtr)
                    {
                        return pb.getPreviousFrame(framePtr);
                    }
                };

                struct CallParticleFunctor
                {
                };

                struct CallFrameFunctor
                {
                };

                template<typename T_Functor>
                struct FrameFunctorInterface
                {
                    T_Functor m_functor;
                    DINLINE FrameFunctorInterface(T_Functor&& functor) : m_functor(std::forward<T_Functor>(functor))
                    {
                    }

                    template<typename T_Acc, typename T_FrameCtx>
                    DINLINE void operator()(T_Acc const& acc, T_FrameCtx& frameIterCtx)
                    {
                        m_functor(acc, frameIterCtx);
                    }
                };

                template<typename T>
                DINLINE auto makeFrameFunctorInterface(T&& t)
                {
                    return FrameFunctorInterface<T>{std::forward<T>(t)};
                }

                template<typename T_Functor>
                struct ParticleFunctorInterface
                {
                    T_Functor m_functor;
                    DINLINE ParticleFunctorInterface(T_Functor&& functor) : m_functor(std::forward<T_Functor>(functor))
                    {
                    }

                    template<typename T_Acc, typename T_FrameType, typename T_ValueTypeSeq>
                    DINLINE void operator()(T_Acc const& acc, Particle<T_FrameType, T_ValueTypeSeq>& particle)
                    {
                        m_functor(acc, particle);
                    }

                    template<typename T_Acc, typename T_FrameType, typename T_ValueTypeSeq>
                    DINLINE void operator()(T_Acc const& acc, Particle<T_FrameType, T_ValueTypeSeq>& particle) const
                    {
                        m_functor(acc, particle);
                    }
                };

                template<typename T>
                DINLINE auto makeParticleFunctorInterface(T&& t)
                {
                    return ParticleFunctorInterface<T>{std::forward<T>(t)};
                }

                /** operate on particles of a superCell
                 *
                 * @tparam T_ParBox type of the particle box
                 * @tparam T_numWorkers number of workers used for execution
                 */
                template<typename T_Accessor, typename T_FrameDirectionIter, typename T_ParBox, uint32_t T_numWorkers>
                struct ForEachParticleForward
                {
                private:
                    static constexpr uint32_t dim = T_ParBox::Dim;


                    using SuperCellSize = typename T_ParBox::FrameType::SuperCellSize;
                    static constexpr uint32_t frameSize = pmacc::math::CT::volume<SuperCellSize>::type::value;
                    static constexpr uint32_t numWorkers = T_numWorkers;
                    /* We work with virtual CUDA blocks if we have more workers than particles.
                     * Each virtual CUDA block is working on a frame, if we have 2 blocks each block processes
                     * every second frame until all frames are processed.
                     */
                    static constexpr uint32_t numVirtualBlocks = (numWorkers + frameSize - 1u) / frameSize;

                    using ForEachParticleInFrame
                        = lockstep::ForEach<lockstep::Config<frameSize * numVirtualBlocks, T_numWorkers, 1>>;

                    using FramePtr = typename T_ParBox::FramePtr;
                    using FramePtrCtx = lockstep::Variable<FramePtr, typename ForEachParticleInFrame::BaseConfig>;
                    mutable FramePtrCtx framePtrCtx;
                    DataSpace<dim> m_superCellIdx;
                    ForEachParticleInFrame const forEachParticleInFrame;
                    T_ParBox m_particlesBox;


                    PMACC_CASSERT_MSG(
                        __this_algorithm_support_foward_frame_iteration_only,
                        std::is_same_v<T_FrameDirectionIter, Forward>);

                public:
                    DINLINE auto lockstepForEach() const
                    {
                        return forEachParticleInFrame;
                    }

                    /** Construct algoritm to operate on particles in a superCell
                     *
                     * @param workerIdx workerIdx index of the worker: range [0;workerSize)
                     * @param particlesBox particles memory
                     *                     It is not allowed concurrently to add or remove particles during the
                     *                     execution of this algorithm.
                     * @param superCellIdx index of the superCell where particles should be processed
                     */
                    DINLINE ForEachParticleForward(
                        uint32_t const workerIdx,
                        T_ParBox const& particlesBox,
                        DataSpace<dim> const& superCellIdx)
                        : forEachParticleInFrame(workerIdx)
                        , m_particlesBox(particlesBox)
                        , m_superCellIdx(superCellIdx)
                    {
                    }

                    /** Execute unary functor for each particle.
                     *
                     * @attention There is no guarantee in which order particles will be executed.
                     *            It is not allowed to assume that workers execute particles frame wise.
                     *
                     * @tparam T_Acc alpaka accelerator type
                     * @tparam T_ParticleFunctor unary particle functor
                     * @param acc alpaka accelerator
                     * @param unaryParticleFunctor Functor executed for each particle with
                     *                             'void operator()(T_Acc const &, ParticleType)'.
                     *                             The caller must ensure that calling the functor in parallel with
                     *                             different workers is data race free.
                     *                             It is not allowed to call a synchronization function within the
                     *                             functor.
                     */
                    template<typename T_Acc, typename T_ParticleFunctor>
                    DINLINE void operator()(T_Acc const& acc, T_ParticleFunctor&& unaryParticleFunctor) const
                    {
                        using FramePtr = typename T_ParBox::FramePtr;
                        using ParticleType = typename T_ParBox::FrameType::ParticleType;

                        using SuperCellSize = typename T_ParBox::FrameType::SuperCellSize;
                        constexpr uint32_t frameSize = pmacc::math::CT::volume<SuperCellSize>::type::value;
                        constexpr uint32_t numWorkers = T_numWorkers;

                        if constexpr(numWorkers > frameSize)
                        {
                            PMACC_CASSERT_MSG(
                                __number_of_workers_must_be_a_multiple_of_framesize,
                                (numWorkers % frameSize) == 0);
                        }

                        auto const& superCell = m_particlesBox.getSuperCell(m_superCellIdx);
                        uint32_t const numPartcilesInSupercell = superCell.getNumParticles();

                        /* We work with virtual CUDA blocks if we have more workers than particles.
                         * Each virtual CUDA block is working on a frame, if we have 2 blocks each block processes
                         * every second frame until all frames are processed.
                         */
                        constexpr uint32_t numVirtualBlocks = (numWorkers + frameSize - 1u) / frameSize;

                        // end kernel if we have no particles
                        if(numPartcilesInSupercell == 0)
                            return;

                        forEachParticleInFrame(
                            [&](lockstep::Idx const idx)
                            {
                                framePtrCtx[idx] = T_FrameDirectionIter::getFirst(m_particlesBox, m_superCellIdx);

                                constexpr uint32_t numberVirtualBlocks = (numWorkers + frameSize - 1u) / frameSize;
                                if constexpr(numberVirtualBlocks > 1)
                                {
                                    uint32_t const virtualBlockId = idx / frameSize;
                                    /* select N-th (N=virtualBlockId) frame from the end of the list */
                                    for(uint32_t i = 1; i <= virtualBlockId && framePtrCtx[idx].isValid(); ++i)
                                    {
                                        framePtrCtx[idx]
                                            = T_FrameDirectionIter::next(m_particlesBox, framePtrCtx[idx]);
                                    }
                                }
                            });

                        uint32_t parOffset = 0u;

                        while(true)
                        {
                            bool isOneFrameValid = parOffset < numPartcilesInSupercell;

                            if(!isOneFrameValid)
                                break;

                            if constexpr(std::is_same_v<T_Accessor, CallParticleFunctor>)
                            {
                                // loop over all particles in the frame
                                forEachParticleInFrame(
                                    [&](lockstep::Idx const linearIdx)
                                    {
                                        uint32_t const parIdxInSupercell = parOffset + linearIdx;
                                        bool const isParticle = parIdxInSupercell < numPartcilesInSupercell;
                                        auto const particleIdx = linearIdx % frameSize;

                                        auto particle = framePtrCtx[linearIdx][particleIdx];
                                        if(!isParticle)
                                            particle.setHandleInvalid();


                                        PMACC_CASSERT_MSG(
                                            __unaryParticleFunctor_must_return_void,
                                            std::is_void_v<decltype(unaryParticleFunctor(acc, particle))>);
                                        if(particle.isHandleValid())
                                        {
                                            auto particleFunctor = makeParticleFunctorInterface(
                                                std::forward<T_ParticleFunctor>(unaryParticleFunctor));
                                            particleFunctor(acc, particle);
                                        }
                                    });
                            }
                            else
                            {
                                auto frameFunctor
                                    = makeFrameFunctorInterface(std::forward<T_ParticleFunctor>(unaryParticleFunctor));
                                frameFunctor(acc, framePtrCtx);
                            }

                            forEachParticleInFrame(
                                [&](lockstep::Idx const idx)
                                {
                                    for(uint32_t i = 0; i < numVirtualBlocks; ++i)
                                        if(framePtrCtx[idx].isValid())
                                            framePtrCtx[idx]
                                                = T_FrameDirectionIter::next(m_particlesBox, framePtrCtx[idx]);
                                });


                            parOffset += frameSize * numVirtualBlocks;
                        }
                    }

                    DINLINE bool hasParticles() const
                    {
                        return numParticles() != 0u;
                    }

                    DINLINE uint32_t numParticles() const
                    {
                        auto const& superCell = m_particlesBox.getSuperCell(m_superCellIdx);
                        return superCell.getNumParticles();
                    }
                };


                /** operate on particles of a superCell
                 *
                 * @tparam T_ParBox type of the particle box
                 * @tparam T_numWorkers number of workers used for execution
                 */
                template<typename T_Accessor, typename T_FrameDirectionIter, typename T_ParBox, uint32_t T_numWorkers>
                struct ForEachParticleBackward
                {
                private:
                    static constexpr uint32_t dim = T_ParBox::Dim;



                    using SuperCellSize = typename T_ParBox::FrameType::SuperCellSize;
                    static constexpr uint32_t frameSize = pmacc::math::CT::volume<SuperCellSize>::type::value;
                    static constexpr uint32_t numWorkers = T_numWorkers;

                    /* We work with virtual CUDA blocks if we have more workers than particles.
                     * Each virtual CUDA block is working on a frame, if we have 2 blocks each block processes
                     * every second frame until all frames are processed.
                     */
                    static constexpr uint32_t numVirtualBlocks = (numWorkers + frameSize - 1u) / frameSize;

                    using ForEachParticleInFrame
                        = lockstep::ForEach<lockstep::Config<frameSize * numVirtualBlocks, T_numWorkers, 1>>;

                    using FramePtr = typename T_ParBox::FramePtr;
                    using FramePtrCtx = lockstep::Variable<FramePtr, typename ForEachParticleInFrame::BaseConfig>;
                    mutable FramePtrCtx framePtrCtx;
                    DataSpace<dim> m_superCellIdx;
                    ForEachParticleInFrame const forEachParticleInFrame;
                    T_ParBox m_particlesBox;


                    PMACC_CASSERT_MSG(
                        __this_algorithm_support_backward_frame_iteration_only,
                        std::is_same_v<T_FrameDirectionIter, Backward>);

                public:
                    DINLINE auto lockstepForEach() const
                    {
                        return forEachParticleInFrame;
                    }

                    /** Construct algoritm to operate on particles in a superCell
                     *
                     * @param workerIdx workerIdx index of the worker: range [0;workerSize)
                     * @param particlesBox particles memory
                     *                     It is not allowed concurrently to add or remove particles during the
                     *                     execution of this algorithm.
                     * @param superCellIdx index of the superCell where particles should be processed
                     */
                    DINLINE ForEachParticleBackward(
                        uint32_t const workerIdx,
                        T_ParBox const& particlesBox,
                        DataSpace<dim> const& superCellIdx)
                        : forEachParticleInFrame(workerIdx)
                        , m_particlesBox(particlesBox)
                        , m_superCellIdx(superCellIdx)
                    {
                    }

                    /** Execute unary functor for each particle.
                     *
                     * @attention There is no guarantee in which order particles will be executed.
                     *            It is not allowed to assume that workers execute particles frame wise.
                     *
                     * @tparam T_Acc alpaka accelerator type
                     * @tparam T_ParticleFunctor unary particle functor
                     * @param acc alpaka accelerator
                     * @param unaryParticleFunctor Functor executed for each particle with
                     *                             'void operator()(T_Acc const &, ParticleType)'.
                     *                             The caller must ensure that calling the functor in parallel with
                     *                             different workers is data race free.
                     *                             It is not allowed to call a synchronization function within the
                     *                             functor.
                     */
                    template<typename T_Acc, typename T_ParticleFunctor>
                    DINLINE void operator()(T_Acc const& acc, T_ParticleFunctor&& unaryParticleFunctor) const
                    {
                        using FramePtr = typename T_ParBox::FramePtr;


                        if constexpr(numWorkers > frameSize)
                        {
                            PMACC_CASSERT_MSG(
                                __number_of_workers_must_be_a_multiple_of_framesize,
                                (numWorkers % frameSize) == 0);
                        }

                        auto particlesInCurentFrameCtx
                            = lockstep::makeVar<lcellId_t>(forEachParticleInFrame, lcellId_t(0u));

                        forEachParticleInFrame(
                            [&](lockstep::Idx const idx)
                            {
                                framePtrCtx[idx] = T_FrameDirectionIter::getFirst(m_particlesBox, m_superCellIdx);

                                if(framePtrCtx[idx].isValid())
                                    particlesInCurentFrameCtx[idx]
                                        = m_particlesBox.getSuperCell(m_superCellIdx).getSizeLastFrame();

                                constexpr uint32_t numberVirtualBlocks = (numWorkers + frameSize - 1u) / frameSize;
                                if constexpr(numberVirtualBlocks > 1)
                                {
                                    uint32_t const virtualBlockId = static_cast<uint32_t>(idx) / frameSize;
                                    /* select N-th (N=virtualBlockId) frame from the end of the list */
                                    for(uint32_t i = 1; i <= virtualBlockId && framePtrCtx[idx].isValid(); ++i)
                                    {
                                        particlesInCurentFrameCtx[idx] = frameSize;
                                        framePtrCtx[idx]
                                            = T_FrameDirectionIter::next(m_particlesBox, framePtrCtx[idx]);
                                    }
                                }
                            });

                        while(true)
                        {
                            bool isOneFrameValid = false;
                            forEachParticleInFrame(
                                [&](lockstep::Idx const idx)
                                { isOneFrameValid = isOneFrameValid || framePtrCtx[idx].isValid(); });

                            if(!isOneFrameValid)
                                break;

                            if constexpr(std::is_same_v<T_Accessor, CallParticleFunctor>)
                            {
                                // loop over all particles in the frame
                                forEachParticleInFrame(
                                    [&](lockstep::Idx const linearIdx)
                                    {
                                        auto const particleIdx = linearIdx % frameSize;
                                        bool const isParticle = particleIdx < particlesInCurentFrameCtx[linearIdx];
                                        auto particle = framePtrCtx[linearIdx][particleIdx];
                                        if(!framePtrCtx[linearIdx].isValid() || !isParticle)
                                            particle.setHandleInvalid();


                                        PMACC_CASSERT_MSG(
                                            __unaryParticleFunctor_must_return_void,
                                            std::is_void_v<decltype(unaryParticleFunctor(acc, particle))>);
                                        if(particle.isHandleValid())
                                        {
                                            unaryParticleFunctor(acc, particle);
                                        }
                                    });
                            }
                            else
                            {
                                auto frameFunctor
                                    = makeFrameFunctorInterface(std::forward<T_ParticleFunctor>(unaryParticleFunctor));
                                frameFunctor(acc, framePtrCtx);
                            }

                            forEachParticleInFrame(
                                [&](lockstep::Idx const idx)
                                {
                                    particlesInCurentFrameCtx[idx] = frameSize;
                                    for(uint32_t i = 0; i < numVirtualBlocks; ++i)
                                        if(framePtrCtx[idx].isValid())
                                            framePtrCtx[idx]
                                                = T_FrameDirectionIter::next(m_particlesBox, framePtrCtx[idx]);
                                });
                        }
                    }

                    DINLINE bool hasParticles() const
                    {
                        return numParticles() != 0u;
                    }

                    DINLINE uint32_t numParticles() const
                    {
                        auto const& superCell = m_particlesBox.getSuperCell(m_superCellIdx);
                        return superCell.getNumParticles();
                    }
                };


                // frame

                template<uint32_t T_numWorkers, typename T_ParBox>
                DINLINE auto makeForEachFrameBackward(
                    uint32_t workerIdx,
                    T_ParBox const& particlesBox,
                    DataSpace<T_ParBox::Dim> const& superCellIdx)
                {
                    return ForEachParticleBackward<CallFrameFunctor, Backward, T_ParBox, T_numWorkers>(
                        workerIdx,
                        particlesBox,
                        superCellIdx);
                }

                template<uint32_t T_numWorkers, typename T_ParBox>
                DINLINE auto makeForEachFrameForward(
                    uint32_t workerIdx,
                    T_ParBox const& particlesBox,
                    DataSpace<T_ParBox::Dim> const& superCellIdx)
                {
                    return ForEachParticleForward<CallFrameFunctor, Forward, T_ParBox, T_numWorkers>(
                        workerIdx,
                        particlesBox,
                        superCellIdx);
                }

                // particle

                template<uint32_t T_numWorkers, typename T_ParBox>
                DINLINE auto makeForEachBackward(
                    uint32_t workerIdx,
                    T_ParBox const& particlesBox,
                    DataSpace<T_ParBox::Dim> const& superCellIdx)
                {
                    return ForEachParticleBackward<CallParticleFunctor, Backward, T_ParBox, T_numWorkers>(
                        workerIdx,
                        particlesBox,
                        superCellIdx);
                }

                template<uint32_t T_numWorkers, typename T_ParBox>
                DINLINE auto makeForEachForward(
                    uint32_t workerIdx,
                    T_ParBox const& particlesBox,
                    DataSpace<T_ParBox::Dim> const& superCellIdx)
                {
                    return ForEachParticleForward<CallParticleFunctor, Forward, T_ParBox, T_numWorkers>(
                        workerIdx,
                        particlesBox,
                        superCellIdx);
                }

                template<uint32_t T_numWorkers, typename T_ParBox>
                DINLINE auto makeForEach(
                    uint32_t workerIdx,
                    T_ParBox const& particlesBox,
                    DataSpace<T_ParBox::Dim> const& superCellIdx)
                {
                    return makeForEachBackward<T_numWorkers>(workerIdx, particlesBox, superCellIdx);
                }


                namespace detail
                {
                    /** operate on particles of a species
                     *
                     * @tparam T_numWorkers number of workers
                     */
                    template<uint32_t T_numWorkers>
                    struct KernelForEachParticle
                    {
                        /** operate on particles
                         *
                         * @tparam T_Acc alpaka accelerator type
                         * @tparam T_Functor type of the functor to operate on a particle
                         * @tparam T_Mapping mapping functor type
                         * @tparam T_ParBox pmacc::ParticlesBox, type of the species box
                         *
                         * @param acc alpaka accelerator
                         * @param functor functor to operate on a particle
                         *                must fulfill the interface pmacc::functor::Interface<F, 1u, void>
                         * @param mapper functor to map a block to a supercell
                         * @param pb particles species box
                         */
                        template<typename T_Acc, typename T_Functor, typename T_Mapping, typename T_ParBox>
                        DINLINE void operator()(
                            T_Acc const& acc,
                            T_Functor functor,
                            T_Mapping const mapper,
                            T_ParBox pb) const
                        {
                            using SuperCellSize = typename T_ParBox::FrameType::SuperCellSize;
                            constexpr uint32_t dim = SuperCellSize::dim;
                            constexpr uint32_t numWorkers = T_numWorkers;

                            uint32_t const workerIdx = cupla::threadIdx(acc).x;

                            DataSpace<dim> const superCellIdx(
                                mapper.getSuperCellIndex(DataSpace<dim>(cupla::blockIdx(acc))));

                            auto forEachParticle = acc::makeForEach<numWorkers>(workerIdx, pb, superCellIdx);

                            // end kernel if we have no particles
                            if(!forEachParticle.hasParticles())
                                return;

                            // offset of the superCell (in cells, without any guards) to the origin of the local
                            // domain
                            DataSpace<dim> const localSuperCellOffset = superCellIdx - mapper.getGuardingSuperCells();

                            auto accFunctor
                                = functor(acc, localSuperCellOffset, lockstep::Worker<numWorkers>{workerIdx});

                            forEachParticle(acc, accFunctor);
                        }
                    };

                } // namespace detail
            } // namespace acc

            /** Run a unary functor for each particle of a species in the given area
             *
             * Has a version for a fixed area, and for a user-provided mapper factory.
             * They differ only in how the area is defined.
             *
             * @warning Does NOT fill gaps automatically! If the
             *          operation deactivates particles or creates "gaps" in any
             *          other way, CallFillAllGaps needs to be called for the
             *          species manually afterwards!
             *
             * @tparam T_Species type of the species
             * @tparam T_Functor unary particle functor type which follows the interface of
             *                   pmacc::functor::Interface<F, 1u, void>
             *
             * @param species species to operate on
             * @param functor operation which is applied to each particle of the species
             *
             * @{
             */

            /** Version for a custom area mapper factory
             *
             * @tparam T_AreaMapperFactory factory type to construct an area mapper that defines the area to
             * process, adheres to the AreaMapperFactory concept
             *
             * @param areaMapperFactory factory to construct an area mapper,
             *                          the area is defined by the constructed mapper object
             */
            template<typename T_Species, typename T_Functor, typename T_AreaMapperFactory>
            HINLINE void forEach(T_Species&& species, T_Functor functor, T_AreaMapperFactory const& areaMapperFactory)
            {
                using MappingDesc = decltype(species.getCellDescription());
                using SuperCellSize = typename MappingDesc::SuperCellSize;
                constexpr uint32_t numWorkers
                    = pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;

                auto const mapper = areaMapperFactory(species.getCellDescription());
                PMACC_KERNEL(acc::detail::KernelForEachParticle<numWorkers>{})
                (mapper.getGridDim(), numWorkers)(std::move(functor), mapper, species.getDeviceParticlesBox());
            }

            /** Version for a fixed area
             *
             * @tparam T_area area to process particles in
             */
            template<uint32_t T_area, typename T_Species, typename T_Functor>
            HINLINE void forEach(T_Species&& species, T_Functor functor)
            {
                auto const areaMapperFactory = AreaMapperFactory<T_area>{};
                forEach(species, functor, areaMapperFactory);
            }

            /** @} */

        } // namespace algorithm
    } // namespace particles
} // namespace pmacc
