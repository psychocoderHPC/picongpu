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


namespace pmacc::particles::algorithm::acc
{
    namespace detail
    {
        //! Tag
        struct CallParticleFunctor;

        struct CallFrameFunctor;

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

    } // namespace detail
} // namespace pmacc::particles::algorithm::acc
