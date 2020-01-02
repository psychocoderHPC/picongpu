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

#include <alpaka/math/atan2/Traits.hpp>

//#include <boost/core/ignore_unused.hpp>

#include <cuda_runtime.h>
#include <type_traits>


namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The CUDA built in atan2.
        class Atan2CudaBuiltIn
        {
        public:
            using Atan2Base = Atan2CudaBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The CUDA atan2 trait specialization.
            template<
                typename Ty,
                typename Tx>
            struct Atan2<
                Atan2CudaBuiltIn,
                Ty,
                Tx,
                typename std::enable_if<
                    std::is_floating_point<Ty>::value
                    && std::is_floating_point<Tx>::value>::type>
            {
                ALPAKA_FN_ACC_CUDA_ONLY static auto atan2(
                    Atan2CudaBuiltIn const & /*abs*/,
                    Ty const & y,
                    Tx const & x)
                -> decltype(::atan2(y, x))
                {
                    //boost::ignore_unused(abs);
                    return ::atan2(y, x);
                }
            };

            template<>
            struct Atan2<
                Atan2CudaBuiltIn,
                float,
                float>
            {
                __device__ static auto atan2(
                    Atan2CudaBuiltIn const & atan2_ctx,
                    float const & y,
                    float const & x)
                -> float
                {
                    alpaka::ignore_unused(atan2_ctx);
                    return ::atan2f(y, x);
                }
            };
        }
    }
}

#endif
