/* Copyright 2019-2022 Rene Widera, Pawel Ordyna
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

#include <pmacc/random/distributions/Uniform.hpp>
#include <pmacc/verify.hpp>

#if(BOOST_LANG_CUDA || BOOST_COMP_HIP)
#    include <mallocMC/mallocMC.hpp>
#endif

namespace picongpu
{
    namespace particles
    {
        namespace collision
        {
            namespace detail
            {
                /* Storage for particle IDs.
                 *
                 * Storage for IDs of collision eligible macro particles. It comes with a simple
                 * shuffling algorithm.
                 */
                struct ListEntry
                {
                    //! Pointer to the actual data stored on the device heap.
                    PMACC_ALIGN(ptrToIndicies, uint32_t*);
                    //! Size of the particle list (number of stored IDs).
                    PMACC_ALIGN(size, uint32_t);


                    /* Initialize storage, allocate memory.
                     *
                     * @param acc alpaka accelerator
                     * @param deviceHeapHandle Heap handle used for allocating storage on device.
                     * @param numPar maximal number of IDs to be stored in the list.
                     */
                    template<typename T_Worker, typename T_DeviceHeapHandle>
                    DINLINE void init(T_Worker const& worker, T_DeviceHeapHandle& deviceHeapHandle, uint32_t numPar)
                    {
                        ptrToIndicies = nullptr;
                        if(numPar != 0u)
                        {
                            const int maxTries = 13; // magic number is not performance critical
                            for(int numTries = 0; numTries < maxTries; ++numTries)
                            {
#if(BOOST_LANG_CUDA || BOOST_COMP_HIP) // Allocate memory on a GPU device
                                constexpr uint32_t typicalParticlesPerSupercell
                                    = particles::TYPICAL_PARTICLES_PER_CELL;
                                /* Round-up the number of slots up to an mutiple of the typical amount of particles of
                                 * a cell. This will waist memory but is avoiding that the mallocMC heap is fragmented
                                 * which often results into out of memory crashes even if the amount of particles in
                                 * the simulation is small.
                                 */
                                uint32_t const allocateNumParticles
                                    = ((numPar + typicalParticlesPerSupercell - 1u) / typicalParticlesPerSupercell)
                                    * typicalParticlesPerSupercell;
                                uint32_t const allocationBytes = sizeof(uint32_t) * allocateNumParticles;
                                ptrToIndicies = (uint32_t*) deviceHeapHandle.malloc(worker.getAcc(), allocationBytes);
#else // No cuda or hip means the device heap is the host heap.
                                ptrToIndicies = new uint32_t[numPar];
#endif
                                if(ptrToIndicies != nullptr)
                                {
                                    break;
                                }
                            }
                            PMACC_DEVICE_VERIFY_MSG(
                                ptrToIndicies != nullptr,
                                "Error: Out of device heap memory in %s:%u\n",
                                __FILE__,
                                __LINE__);
#if 0
                            PMACC_DEVICE_VERIFY_MSG(
                                (((uint64_t)(ptrToIndicies)) % 4) == 0,
                                "Error: Ptr align issue %lld %s:%u\n",
                                (uint64_t)(ptrToIndicies),
                                __FILE__,
                                __LINE__);
#endif
                        }
                        // reset counter
                        size = 0u;
                    }

                    /* Release allocated heap memory.
                     *
                     * @param acc alpaka accelerator
                     * @param deviceHeapHandle Heap handle used for allocating storage on device.
                     */
                    template<typename T_Worker, typename T_DeviceHeapHandle>
                    DINLINE void finalize(T_Worker const& worker, T_DeviceHeapHandle& deviceHeapHandle)
                    {
                        if(ptrToIndicies != nullptr)
                        {
#if(BOOST_LANG_CUDA || BOOST_COMP_HIP)
                            deviceHeapHandle.free(worker.getAcc(), (void*) ptrToIndicies);
                            ptrToIndicies = nullptr;
#else
                            delete(ptrToIndicies);
#endif
                        }
                    }


                    /* Shuffle list entries.
                     *
                     * @param worker lockstep worker
                     * @param rngHandle random number generator handle
                     */
                    template<typename T_Worker, typename T_RngHandle>
                    DINLINE void shuffle(T_Worker const& worker, T_RngHandle& rngHandle)
                    {
                        using UniformUint32_t = pmacc::random::distributions::Uniform<uint32_t>;
                        auto rng = rngHandle.template applyDistribution<UniformUint32_t>();
                        // shuffle the particle lookup table
                        for(uint32_t i = size; i > 1; --i)
                        {
                            /* modulo is not perfect but okish,
                             * because of the loop head mod zero is not possible
                             */
                            uint32_t p = rng(worker) % i;
                            if(i - 1 != p)
                                swap(ptrToIndicies[i - 1], ptrToIndicies[p]);
                        }
                    }


                private:
                    /* Swap two list entries.
                     *
                     * @param v0 index of the 1st entry to swap
                     * @param v0 index of the 2nd entry to swap
                     */
                    DINLINE void swap(uint32_t& v0, uint32_t& v1)
                    {
                        uint32_t tmp = v0;
                        v0 = v1;
                        v1 = tmp;
                    }
                };

                // TODO: simplify (maybe crate a class and inject all the required stuff as private references?), Check
                // const, & etc.
                //! Counting particles per grid frame
                template<
                    typename T_Worker,
                    typename T_ForEach,
                    typename T_ParBox,
                    typename T_FramePtr,
                    typename T_Array,
                    typename T_Filter>
                DINLINE void particlesCntHistogram(
                    T_Worker const& worker,
                    T_ForEach forEach,
                    T_ParBox& parBox,
                    T_FramePtr frame,
                    uint32_t const numParticlesInSupercell,
                    T_Array& nppc,
                    T_Filter& filter)
                {
                    using SuperCellSize = typename T_ParBox::FrameType::SuperCellSize;
                    constexpr uint32_t frameSize = pmacc::math::CT::volume<SuperCellSize>::type::value;

                    for(uint32_t i = 0; i < numParticlesInSupercell; i += frameSize)
                    {
                        forEach(
                            [&](uint32_t const linearIdx)
                            {
                                if(i + linearIdx < numParticlesInSupercell)
                                {
                                    auto particle = frame[linearIdx];
                                    if(filter(worker, particle))
                                    {
                                        auto parLocalIndex = particle[localCellIdx_];
                                        cupla::atomicAdd(
                                            worker.getAcc(),
                                            &nppc[parLocalIndex],
                                            1u,
                                            ::alpaka::hierarchy::Threads{});
                                    }
                                }
                            });
                        frame = parBox.getNextFrame(frame);
                    }
                }

                /* Fills parCellList with new particles.
                 * parCellList stores a list of particles for each grid cell and the
                 * index in the supercell for each particle.
                 */
                template<
                    typename T_Worker,
                    typename T_ForEach,
                    typename T_ParBox,
                    typename T_FramePtr,
                    typename T_EntryListArray,
                    typename T_Filter,
                    typename T_Array>
                DINLINE void updateLinkedList(
                    T_Worker const& worker,
                    T_ForEach forEach,
                    T_ParBox& parBox,
                    T_FramePtr frame,
                    uint32_t const numParticlesInSupercell,
                    T_EntryListArray& parCellList,
                    T_Filter& filter,
                    T_Array& nppc,
                    uint32_t superCellIdx)
                {
                    using SuperCellSize = typename T_ParBox::FrameType::SuperCellSize;
                    constexpr uint32_t frameSize = pmacc::math::CT::volume<SuperCellSize>::type::value;
                    for(uint32_t i = 0; i < numParticlesInSupercell; i += frameSize)
                    {
                        forEach(
                            [&](uint32_t const linearIdx)
                            {
                                uint32_t parInSuperCellIdx = i + linearIdx;
                                if(parInSuperCellIdx < numParticlesInSupercell)
                                {
                                    auto particle = frame[linearIdx];
                                    if(filter(worker, particle))
                                    {
                                        auto parLocalIndex = particle[localCellIdx_];
                                        uint32_t parOffset = cupla::atomicAdd(
                                            worker.getAcc(),
                                            &parCellList[parLocalIndex].size,
                                            1u,
                                            ::alpaka::hierarchy::Threads{});
                                        PMACC_DEVICE_VERIFY_MSG(
                                            parInSuperCellIdx < numParticlesInSupercell,
                                            "Error: update links Out of Range %u of %u %s:%u\n",
                                            parInSuperCellIdx,
                                            numParticlesInSupercell,
                                            __FILE__,
                                            __LINE__);
#if 1
                                        auto x = cupla::atomicExch(
                                            worker.getAcc(),
                                            &(parCellList[parLocalIndex].ptrToIndicies[parOffset]),
                                            parOffset * 10000000 + 2000000000 + superCellIdx*1000 + parLocalIndex);
#else
                                        auto x = cupla::atomicExch(
                                            worker.getAcc(),
                                            &(parCellList[parLocalIndex].ptrToIndicies[parOffset]),
                                            superCellIdx + 100000000);
#endif
                                        PMACC_DEVICE_VERIFY_MSG(
                                            x == superCellIdx + 100000000,
                                            "SuperCellIdxCheck: cell=%u %u/%u %u == %u %lld %u %s:%u\n",
                                            parLocalIndex,
                                            parOffset,
                                            nppc[parLocalIndex],
                                            x,
                                            superCellIdx + 100000000,
                                            (uint64_t) parCellList[parLocalIndex].ptrToIndicies,
                                            linearIdx,
                                            __FILE__,
                                            __LINE__);
                                    }
                                }
                            });
                        frame = parBox.getNextFrame(frame);
                    }
                    worker.sync();
#if 1
                    forEach(
                        [&](uint32_t const linearIdx)
                        {
                            uint32_t const numParInCell = parCellList[linearIdx].size;
                            uint32_t* parListStart = parCellList[linearIdx].ptrToIndicies;
                            for(uint32_t ii = 0; ii < numParInCell; ii++)
                            {
                                auto value = (parListStart[ii] - 2000000000) / 10000000;
                                PMACC_DEVICE_VERIFY_MSG(
                                    value < numParticlesInSupercell,
                                    "Validate error: Out of Range %u:%u -> %u of %u (in cell %u) %s:%u\n",
                                    nppc[linearIdx],
                                    ii,
                                    value,
                                    numParticlesInSupercell,
                                    numParInCell,
                                    __FILE__,
                                    __LINE__);
                            }
                        });
#endif
                }

                template<typename T_ParBox, typename T_FramePtr>
                DINLINE auto getParticle(T_ParBox& parBox, T_FramePtr frame, uint32_t particleId) ->
                    typename T_FramePtr::type::ParticleType
                {
                    constexpr uint32_t frameSize
                        = pmacc::math::CT::volume<typename T_FramePtr::type::SuperCellSize>::type::value;
                    uint32_t const skipFrames = particleId / frameSize;
                    for(uint32_t i = 0; i < skipFrames; ++i)
                        frame = parBox.getNextFrame(frame);
                    return frame[particleId % frameSize];
                }

                template<
                    typename T_Worker,
                    typename T_ForEach,
                    typename T_DeviceHeapHandle,
                    typename T_ParBox,
                    typename T_FramePtr,
                    typename T_EntryListArray,
                    typename T_Array,
                    typename T_Filter>
                DINLINE void prepareList(
                    T_Worker const& worker,
                    T_ForEach forEach,
                    T_DeviceHeapHandle deviceHeapHandle,
                    T_ParBox& parBox,
                    T_FramePtr firstFrame,
                    uint32_t const numParticlesInSupercell,
                    T_EntryListArray& parCellList,
                    T_Array& nppc,
                    T_Filter filter,
                    uint32_t superCellIdx)
                {
                    // Initialize nppc with zeros.
                    forEach([&](uint32_t const linearIdx) { nppc[linearIdx] = 0u; });
                    worker.sync();
                    // Count eligible
                    particlesCntHistogram(worker, forEach, parBox, firstFrame, numParticlesInSupercell, nppc, filter);
                    worker.sync();

                    // memory for particle indices
                    forEach([&](uint32_t const linearIdx)
                            { parCellList[linearIdx].init(worker, deviceHeapHandle, nppc[linearIdx]); });

                    worker.sync();

                    forEach(
                        [&](uint32_t const linearIdx)
                        {
                            uint32_t const numParInCell = nppc[linearIdx];
                            uint32_t* parListStart = parCellList[linearIdx].ptrToIndicies;
                            for(uint32_t ii = 0; ii < numParInCell; ii++)
                            {
                                parListStart[ii] = superCellIdx + 100000000;
                            }
                        });
                    worker.sync();

                    detail::updateLinkedList(
                        worker,
                        forEach,
                        parBox,
                        firstFrame,
                        numParticlesInSupercell,
                        parCellList,
                        filter,
                        nppc,
                        superCellIdx);
                    worker.sync();
                }
            } // namespace detail
        } // namespace collision
    } // namespace particles
} // namespace picongpu
