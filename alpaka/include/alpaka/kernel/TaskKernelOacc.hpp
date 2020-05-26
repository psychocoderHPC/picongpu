/* Copyright 2019 Benjamin Worpitz, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED

#if _OPENACC < 201306
    #error If ALPAKA_ACC_ANY_BT_OACC_ENABLED is set, the compiler has to support OpenACC xx or higher!
#endif

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

// Implementation details.
#include <alpaka/ctx/block/CtxBlockOacc.hpp>
#include <alpaka/ctx/thread/CtxThreadOacc.hpp>
#include <alpaka/dev/DevOacc.hpp>
#include <alpaka/idx/MapIdx.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>

#include <alpaka/meta/ApplyTuple.hpp>

#include <functional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <algorithm>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>
#endif

namespace alpaka
{
    namespace kernel
    {
        //#############################################################################
        //! The CPU OpenMP 4.0 accelerator execution task.
        template<
            typename TDim,
            typename TIdx,
            typename TKernelFnObj,
            typename... TArgs>
        class TaskKernelOacc final :
            public workdiv::WorkDivMembers<TDim, TIdx>
        {
        public:
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST TaskKernelOacc(
                TWorkDiv && workDiv,
                TKernelFnObj const & kernelFnObj,
                TArgs && ... args) :
                    workdiv::WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv)),
                    m_kernelFnObj(kernelFnObj),
                    m_args(std::forward<TArgs>(args)...)
            {
                static_assert(
                    dim::Dim<std::decay_t<TWorkDiv>>::value == TDim::value,
                    "The work division and the execution task have to be of the same dimensionality!");
            }
            //-----------------------------------------------------------------------------
            TaskKernelOacc(TaskKernelOacc const & other) = default;
            //-----------------------------------------------------------------------------
            TaskKernelOacc(TaskKernelOacc && other) = default;
            //-----------------------------------------------------------------------------
            auto operator=(TaskKernelOacc const &) -> TaskKernelOacc & = default;
            //-----------------------------------------------------------------------------
            auto operator=(TaskKernelOacc &&) -> TaskKernelOacc & = default;
            //-----------------------------------------------------------------------------
            ~TaskKernelOacc() = default;

            //-----------------------------------------------------------------------------
            //! Executes the kernel function object.
            ALPAKA_FN_HOST auto operator()(
                    const
                    dev::DevOacc& dev
                ) const
            -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                auto const gridBlockExtent(
                    workdiv::getWorkDiv<Grid, Blocks>(*this));
                auto const blockThreadExtent(
                    workdiv::getWorkDiv<Block, Threads>(*this));
                auto const threadElemExtent(
                    workdiv::getWorkDiv<Thread, Elems>(*this));

                std::cout << "m_gridBlockExtent=" << this->m_gridBlockExtent << "\tgridBlockExtent=" << gridBlockExtent << std::endl;
                std::cout << "m_blockThreadExtent=" << this->m_blockThreadExtent << "\tblockThreadExtent=" << blockThreadExtent << std::endl;
                std::cout << "m_threadElemExtent=" << this->m_threadElemExtent << "\tthreadElemExtent=" << threadElemExtent << std::endl;

                // Get the size of the block shared dynamic memory.
                auto const blockSharedMemDynSizeBytes(
                    meta::apply(
                        [&](std::decay_t<TArgs> const & ... args)
                        {
                            return
                                kernel::getBlockSharedMemDynSizeBytes<
                                    acc::AccOacc<TDim, TIdx>>(
                                        m_kernelFnObj,
                                        blockThreadExtent,
                                        threadElemExtent,
                                        args...);
                        },
                        m_args));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__
                    << " blockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B" << std::endl;
#endif
                // The number of blocks in the grid.
                TIdx const gridBlockCount(gridBlockExtent.prod());
                // The number of threads in a block.
                TIdx const blockThreadCount(blockThreadExtent.prod());

                // make sure there is at least on team
                TIdx const teamCount(gridBlockCount);
                std::cout << "threadElemCount=" << threadElemExtent[0u] << std::endl;
                std::cout << "teamCount=" << teamCount << "\tgridBlockCount=" << gridBlockCount << std::endl;

                // `When an if(scalar-expression) evaluates to false, the structured block is executed on the host.`
                auto argsD = m_args;
                auto kernelFnObj = m_kernelFnObj;
                dev.makeCurrent();
                #pragma acc parallel num_gangs(teamCount) num_workers(blockThreadCount)
                {
                    {
                        printf("threadElemCount_dev %d\n", int(threadElemExtent[0u]));
                        // iterate over groups of teams to stay withing thread limit
                        for(TIdx t = 0u; t < gridBlockCount; t+=teamCount)
                        {
                            // printf("acc->threadElemCount %d\n"
                            //         , int(acc.m_threadElemExtent[0]));

                            const TIdx bsup = std::min(static_cast<TIdx>(t + teamCount), gridBlockCount);
                            #pragma acc loop gang
                            for(TIdx b = t; b<bsup; ++b)
                            {
                                // vec::Vec<dim::DimInt<1u>, TIdx> const gridBlockIdx(b);
                                // When this is not repeated here:
                                // error: gridBlockExtent referenced in target region does not have a mappable type
                                // auto const gridBlockExtent2(
                                //     workdiv::getWorkDiv<Grid, Blocks>(*static_cast<workdiv::WorkDivMembers<TDim, TIdx> const *>(this)));
                                ctx::CtxBlockOacc<TDim, TIdx> blockShared(
                                    gridBlockExtent,
                                    blockThreadExtent,
                                    threadElemExtent,
                                    b,
                                    blockSharedMemDynSizeBytes);

                                // Execute the threads in parallel.

                                // Parallel execution of the threads in a block is required because when syncBlockThreads is called all of them have to be done with their work up to this line.
                                // So we have to spawn one OS thread per thread in a block.
                                // 'omp for' is not useful because it is meant for cases where multiple iterations are executed by one thread but in our case a 1:1 mapping is required.
                                // Therefore we use 'omp parallel' with the specified number of threads in a block.
                                // vec::Vec<dim::DimInt<1u>, TIdx> blockThreadIdx(static_cast<TIdx>(0u));
                                #pragma acc loop worker
                                for(TIdx w = 0; w < blockThreadCount; ++w)
                                {
                                    // blockThreadIdx[0] = w;
                                    ctx::CtxThreadOacc<TDim, TIdx> acc(
                                        w,
                                        blockShared);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL && 0
                                    // The first thread does some checks in the first block executed.
                                    if((::omp_get_thread_num() == 0) && (b == 0))
                                    {
                                        int const numThreads(::omp_get_num_threads());
                                        printf("%s omp_get_num_threads: %d\n", __func__, numThreads);
                                        if(numThreads != static_cast<int>(blockThreadCount))
                                        {
                                            printf("ERROR: The OpenMP runtime did not use the number of threads that had been requested!\n");
                                        }
                                    }
#endif
                                    meta::apply(
                                        [kernelFnObj, &acc](typename std::decay<TArgs>::type const & ... args)
                                        {
                                            kernelFnObj(
                                                    acc,
                                                    args...);
                                        },
                                        argsD);

                                    // Wait for all threads to finish before deleting the shared memory.
                                    // This is done by default if the omp 'nowait' clause is missing
                                    //block::sync::syncBlockThreads(acc);
                                }
                                block::shared::st::freeMem(blockShared);
                            }
                        }
                    }
                }
            }

        private:
            TKernelFnObj m_kernelFnObj;
            std::tuple<std::decay_t<TArgs>...> m_args;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 execution task accelerator type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct AccType<
                kernel::TaskKernelOacc<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = acc::AccOacc<TDim, TIdx>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 execution task device type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevType<
                kernel::TaskKernelOacc<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = dev::DevOacc;
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 execution task dimension getter trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct DimType<
                kernel::TaskKernelOacc<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = TDim;
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 execution task platform type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct PltfType<
                kernel::TaskKernelOacc<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = pltf::PltfOacc;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 execution task idx type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct IdxType<
                kernel::TaskKernelOacc<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = TIdx;
            };
        }
    }

    namespace queue
    {
        namespace traits
        {
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct Enqueue<
                queue::QueueOaccBlocking,
                kernel::TaskKernelOacc<TDim, TIdx, TKernelFnObj, TArgs...> >
            {
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueOaccBlocking& queue,
                    kernel::TaskKernelOacc<TDim, TIdx, TKernelFnObj, TArgs...> const & task)
                -> void
                {
                    std::lock_guard<std::mutex> lk(queue.m_spQueueImpl->m_mutex);

                    queue.m_spQueueImpl->m_bCurrentlyExecutingTask = true;

                    task(
                            queue.m_spQueueImpl->m_dev
                        );

                    queue.m_spQueueImpl->m_bCurrentlyExecutingTask = false;
                }
            };

            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct Enqueue<
                queue::QueueOaccNonBlocking,
                kernel::TaskKernelOacc<TDim, TIdx, TKernelFnObj, TArgs...> >
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueOaccNonBlocking& queue,
                    kernel::TaskKernelOacc<TDim, TIdx, TKernelFnObj, TArgs...> const & task)
                -> void
                {
                    queue.m_spQueueImpl->m_workerThread.enqueueTask(
                        [&queue, task]()
                        {
                            task(
                                    queue.m_spQueueImpl->m_dev
                                );
                        });
                }
            };
        }
    }
}

#endif
