/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/dev/DevCudaRt.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/stream/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <alpaka/core/Cuda.hpp>

#include <stdexcept>
#include <memory>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>

namespace alpaka
{
    namespace event
    {
        class EventCudaRt;
    }
}

namespace alpaka
{
    namespace stream
    {
        namespace cuda
        {
            namespace detail
            {
                //#############################################################################
                //! The CUDA RT sync stream implementation.
                class StreamCudaRtSyncImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST StreamCudaRtSyncImpl(
                        dev::DevCudaRt const & dev) :
                            m_dev(dev),
                            m_CudaStream()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                m_dev.m_iDevice));
                        // - cudaStreamDefault: Default stream creation flag.
                        // - cudaStreamNonBlocking: Specifies that work running in the created stream may run concurrently with work in stream 0 (the NULL stream),
                        //   and that the created stream should perform no implicit synchronization with stream 0.
                        // Create the stream on the current device.
                        // NOTE: cudaStreamNonBlocking is required to match the semantic implemented in the alpaka CPU stream.
                        // It would be too much work to implement implicit default stream synchronization on CPU.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaStreamCreateWithFlags(
                                &m_CudaStream,
                                cudaStreamDefault));
                    }
                    //-----------------------------------------------------------------------------
                    StreamCudaRtSyncImpl(StreamCudaRtSyncImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    StreamCudaRtSyncImpl(StreamCudaRtSyncImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    auto operator=(StreamCudaRtSyncImpl const &) -> StreamCudaRtSyncImpl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(StreamCudaRtSyncImpl &&) -> StreamCudaRtSyncImpl & = default;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ~StreamCudaRtSyncImpl()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device. \TODO: Is setting the current device before cudaStreamDestroy required?
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                m_dev.m_iDevice));
                        // In case the device is still doing work in the stream when cudaStreamDestroy() is called, the function will return immediately
                        // and the resources associated with stream will be released automatically once the device has completed all work in stream.
                        // -> No need to synchronize here.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaStreamDestroy(
                                m_CudaStream));
                    }

                public:
                    dev::DevCudaRt const m_dev;   //!< The device this stream is bound to.
                    cudaStream_t m_CudaStream;
                };
            }
        }

        //#############################################################################
        //! The CUDA RT sync stream.
        class StreamCudaRtSync final
        {
        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST StreamCudaRtSync(
                dev::DevCudaRt const & dev) :
                m_spStreamImpl(std::make_shared<cuda::detail::StreamCudaRtSyncImpl>(dev))
            {}
            //-----------------------------------------------------------------------------
            StreamCudaRtSync(StreamCudaRtSync const &) = default;
            //-----------------------------------------------------------------------------
            StreamCudaRtSync(StreamCudaRtSync &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(StreamCudaRtSync const &) -> StreamCudaRtSync & = default;
            //-----------------------------------------------------------------------------
            auto operator=(StreamCudaRtSync &&) -> StreamCudaRtSync & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(StreamCudaRtSync const & rhs) const
            -> bool
            {
                return (m_spStreamImpl == rhs.m_spStreamImpl);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(StreamCudaRtSync const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~StreamCudaRtSync() = default;

        public:
            std::shared_ptr<cuda::detail::StreamCudaRtSyncImpl> m_spStreamImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT sync stream device type trait specialization.
            template<>
            struct DevType<
                stream::StreamCudaRtSync>
            {
                using type = dev::DevCudaRt;
            };
            //#############################################################################
            //! The CUDA RT sync stream device get trait specialization.
            template<>
            struct GetDev<
                stream::StreamCudaRtSync>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    stream::StreamCudaRtSync const & stream)
                -> dev::DevCudaRt
                {
                    return stream.m_spStreamImpl->m_dev;
                }
            };
        }
    }
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT sync stream event type trait specialization.
            template<>
            struct EventType<
                stream::StreamCudaRtSync>
            {
                using type = event::EventCudaRt;
            };
        }
    }
    namespace stream
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT sync stream enqueue trait specialization.
            template<
                typename TTask>
            struct Enqueue<
                stream::StreamCudaRtSync,
                TTask>
            {
                //#############################################################################
                enum class CallbackState
                {
                    enqueued,
                    notified,
                    finished,
                };

                //#############################################################################
                struct CallbackSynchronizationData : public std::enable_shared_from_this<CallbackSynchronizationData>
                {
                    std::mutex m_mutex;
                    std::condition_variable m_event;
                    bool notified = false;
                    CallbackState state = CallbackState::enqueued;
                };

                //-----------------------------------------------------------------------------
                static void CUDART_CB cudaRtCallback(cudaStream_t /*stream*/, cudaError_t /*status*/, void *arg)
                {
                    // explicitly copy the shared_ptr so that this method holds the state even when the executing thread has already finished.
                    const auto pCallbackSynchronizationData = reinterpret_cast<CallbackSynchronizationData*>(arg)->shared_from_this();

                    // Notify the executing thread.
                    {
                        std::unique_lock<std::mutex> lock(pCallbackSynchronizationData->m_mutex);
                        pCallbackSynchronizationData->state = CallbackState::notified;
                    }

                    // Wait for the executing thread to finish the task if it has not already finished.
                    std::unique_lock<std::mutex> lock(pCallbackSynchronizationData->m_mutex);
                    if(pCallbackSynchronizationData->state != CallbackState::finished)
                    {
                        pCallbackSynchronizationData->m_event.wait(
                            lock,
                            [pCallbackSynchronizationData](){
                                return pCallbackSynchronizationData->state == CallbackState::finished;
                            }
                        );
                    }
                }

                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCudaRtSync & stream,
                    TTask const & task)
                -> void
                {
                    auto pCallbackSynchronizationData = std::make_shared<CallbackSynchronizationData>();

                    ALPAKA_CUDA_RT_CHECK(cudaStreamAddCallback(
                        stream.m_spStreamImpl->m_CudaStream,
                        cudaRtCallback,
                        pCallbackSynchronizationData.get(),
                        0u));

                    // We start a new std::thread which stores the task to be executed.
                    // This circumvents the limitation that it is not possible to call CUDA methods within the CUDA callback thread.
                    // The CUDA thread signals the std::thread when it is ready to execute the task.
                    // The CUDA thread is waiting for the std::thread to signal that it is finished executing the task
                    // before it executes the next task in the queue (CUDA stream).
                    std::thread t(
                        [pCallbackSynchronizationData, task](){

                            // If the callback has not yet been called, we wait for it.
                            {
                                std::unique_lock<std::mutex> lock(pCallbackSynchronizationData->m_mutex);
                                if(pCallbackSynchronizationData->state != CallbackState::notified)
                                {
                                    pCallbackSynchronizationData->m_event.wait(
                                        lock,
                                        [pCallbackSynchronizationData](){
                                            return pCallbackSynchronizationData->state == CallbackState::notified;
                                        }
                                    );
                                }

                                task();

                                // Notify the waiting CUDA thread.
                                pCallbackSynchronizationData->state = CallbackState::finished;
                            }
                            pCallbackSynchronizationData->m_event.notify_one();
                        }
                    );
                    t.join();
                }
            };
            //#############################################################################
            //! The CUDA RT sync stream test trait specialization.
            template<>
            struct Empty<
                stream::StreamCudaRtSync>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto empty(
                    stream::StreamCudaRtSync const & stream)
                -> bool
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Query is allowed even for streams on non current device.
                    cudaError_t ret = cudaSuccess;
                    ALPAKA_CUDA_RT_CHECK_IGNORE(
                        ret = cudaStreamQuery(
                            stream.m_spStreamImpl->m_CudaStream),
                        cudaErrorNotReady);
                    return (ret == cudaSuccess);
                }
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT sync stream thread wait trait specialization.
            //!
            //! Blocks execution of the calling thread until the stream has finished processing all previously requested tasks (kernels, data copies, ...)
            template<>
            struct CurrentThreadWaitFor<
                stream::StreamCudaRtSync>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    stream::StreamCudaRtSync const & stream)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Sync is allowed even for streams on non current device.
                    ALPAKA_CUDA_RT_CHECK(cudaStreamSynchronize(
                        stream.m_spStreamImpl->m_CudaStream));
                }
            };
        }
    }
}

#endif
