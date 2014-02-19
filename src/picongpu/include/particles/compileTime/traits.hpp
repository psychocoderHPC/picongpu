/**
 * Copyright 2014 Rene Widera
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

#include "simulation_defines.hpp"
#include "traits/GetMargin.hpp"

namespace picongpu
{

template<typename T_Species>
struct GetCurrentSolver
{
    typedef typename GetFlagType<typename T_Species::type::FrameType, current<> >::type::ThisType type;
};

template<typename T_Species>
struct GetPusher
{
    typedef typename GetFlagType<typename T_Species::type::FrameType, pusher<> >::type::ThisType type;
};

template<typename T_Species>
struct GetShape
{
    typedef typename GetFlagType<typename T_Species::type::FrameType, shape<> >::type::ThisType type;
};

template<typename T_Species>
struct GetInterpolation
{
    typedef typename GetFlagType<typename T_Species::type::FrameType, interpolation<> >::type::ThisType type;
};

template<typename T_Type>
struct GetLowerMarging
{
    typedef typename traits::GetMargin<T_Type>::LowerMargin type;
};

template<typename T_Type>
struct GetUpperMarging
{
    typedef typename traits::GetMargin<T_Type>::UpperMargin type;
};

}// namespace picongpu
