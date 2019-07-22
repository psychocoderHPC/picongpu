/* Copyright 2013-2019 Rene Widera, Axel Huebl
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
#include "picongpu/particles/collision/binary/MomentumSwap.def"

#include <pmacc/random/distributions/Uniform.hpp>

#include <utility>
#include <type_traits>

namespace picongpu
{
namespace particles
{
namespace collision
{
namespace binary
{
namespace acc
{
    struct MomentumSwap
    {
        //! store user manipulators instance
        HDINLINE MomentumSwap()  = default;

        /** execute the user manipulator functor
         *
         * @tparam T_Args type of the arguments passed to the user manipulator functor
         *
         * @param args arguments passed to the user functor
         */
        template<
            typename T_Context,
            typename T_Par0,
            typename T_Par1,
            typename ... T_Args >
        HDINLINE
        void operator( )(
            T_Context const & ctx,
            T_Par0 & par0,
            T_Par1 & par1
        )
        {
            auto const & acc = *ctx.m_acc;
            auto & rngHandle = *ctx.m_hRng;
            using UniformFloat = pmacc::random::distributions::Uniform<float_X>;
            auto rng = rngHandle.template applyDistribution< UniformFloat >();
            float_X rngValue = rng(acc);
            auto par0Mom = par0[ momentum_ ];
            auto par1Mom = par1[ momentum_ ];

            par0[ momentum_ ] = par0Mom * rngValue + par1Mom * ( 1.0_X * rngValue );
            par1[ momentum_ ] = par1Mom * rngValue + par0Mom * ( 1.0_X * rngValue );
        }
    };
} // namespace acc

    struct MomentumSwap
    {
        template< typename T_Species0, typename T_Species1 >
        struct apply
        {
            using type = MomentumSwap;
        };

        HINLINE MomentumSwap( ) = default;

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
            typename T_WorkerCfg,
            typename T_Acc
        >
        HDINLINE acc::MomentumSwap
        operator()(
            T_Acc const &,
            DataSpace< simDim > const &,
            T_WorkerCfg const &
        ) const
        {
            return acc::MomentumSwap( );
        }

        //! get the name of the functor
        static
        HINLINE std::string
        getName( )
        {
            return "MomentumSwap";
        }

    };

} // namespace binary
} // namespace collision
} // namespace particles
} // namespace picongpu
