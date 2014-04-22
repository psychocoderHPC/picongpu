/**
 * Copyright 2014 Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "types.h"
#include "compileTime/conversion/SeqToMap.hpp"
#include "compileTime/conversion/TypeToAliasPair.hpp"
#include "compileTime/conversion/TypeToPair.hpp"
#include "compileTime/conversion/MakeSeqFromNestedSeq.hpp"
#include <boost/mpl/at.hpp>
#include <boost/mpl/copy.hpp>

#include <boost/mpl/assert.hpp>
#include <boost/mpl/pop_back.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>

#include "math/vector/compile-time/Vector.hpp"

namespace PMacc
{
namespace bmpl = boost::mpl;


template<typename T_MathVector, typename T_Value, uint32_t pos>
struct AssignToDim;

template<template<typename, typename, typename> class T_MathVector, typename T_Value,
typename T_X, typename T_Y, typename T_Z>
struct AssignToDim<T_MathVector<T_X, T_Y, T_Z>, T_Value, 0>
{
    typedef T_MathVector<T_Value, T_Y, T_Z> type;
};

template<template<typename, typename, typename> class T_MathVector, typename T_Value,
typename T_X, typename T_Y, typename T_Z>
struct AssignToDim<T_MathVector<T_X, T_Y, T_Z>, T_Value, 1>
{
    typedef T_MathVector<T_X, T_Value, T_Z> type;
};

template<template<typename, typename, typename> class T_MathVector, typename T_Value,
typename T_X, typename T_Y, typename T_Z>
struct AssignToDim<T_MathVector<T_X, T_Y, T_Z>, T_Value, 2>
{
    typedef T_MathVector<T_X, T_Y, T_Value> type;
};

template<typename T_InVector,
typename T_Element,
uint32_t T_pos
>
struct AssignToAny
{
    typedef T_InVector InVector;
    typedef T_Element Element;
    static const uint32_t pos = T_pos;
    
    template<typename T_Value>
    struct AssignToDimFake
    {
        typedef typename AssignToDim<T_Value, Element, pos >::type type;
    };

    typedef typename bmpl::transform< InVector,  AssignToDimFake<bmpl::_1> >::type type;
};

template<typename T_Ranges,
uint32_t T_pos
>
struct RangesToMathVectorSeq
{
    typedef math::CT::Vector<> EmptyVector;

    template<typename T_Value>
    struct AssignToDimFake
    {
        typedef typename AssignToDim<EmptyVector, T_Value, T_pos >::type type;
    };

    typedef typename bmpl::transform< T_Ranges, AssignToDimFake<bmpl::_1> >::type type;
};

template<typename T_RangeVector,
typename T_TmpResult = bmpl::vector0<>,
bool isEmpty = bmpl::empty<T_RangeVector>::value
>
struct AllCombinations;

template<typename T_RangeVector,typename T_TmpResult>
struct AllCombinations<T_RangeVector, T_TmpResult, false >
{
    typedef T_RangeVector RangeVector;
    typedef T_TmpResult TmpResult;

    static const int rangeVectorSize = bmpl::size<RangeVector>::value;
    typedef typename bmpl::at<RangeVector, bmpl::int_<rangeVectorSize - 1> > ::type LastElement;
    typedef typename bmpl::pop_back<RangeVector>::type ShrinkedRangeVector;
        
    typedef typename bmpl::copy<LastElement, bmpl::back_inserter< bmpl::vector0<> > >::type TmpVector;

    template<typename T_Value>
    struct AssignToAnyFake
    {
        typedef typename AssignToAny<TmpResult, T_Value, rangeVectorSize -1  >::type type;
    };
    
    typedef typename bmpl::transform< TmpVector, AssignToAnyFake<bmpl::_1> >::type NestedSeq;
    typedef typename MakeSeqFromNestedSeq<NestedSeq>::type OneSeq;
    
    typedef typename AllCombinations<ShrinkedRangeVector,OneSeq>::type type;
};
 

template<typename T_RangeVector,typename T_TmpResult>
struct AllCombinations<T_RangeVector, T_TmpResult, true >
{
    typedef T_TmpResult type;
};

template<typename T_RangeVector>
struct AllCombinations<T_RangeVector, bmpl::vector0<>, false >
{
    typedef T_RangeVector RangeVector;

    static const int rangeVectorSize = bmpl::size<RangeVector>::value;
    typedef typename bmpl::at<RangeVector, bmpl::int_<rangeVectorSize - 1 > > ::type LastElement;
    typedef typename bmpl::pop_back<RangeVector>::type ShrinkedRangeVector;

    typedef typename bmpl::copy<LastElement, bmpl::back_inserter< bmpl::vector0<> > >::type TmpVector;
    typedef typename RangesToMathVectorSeq<TmpVector, rangeVectorSize - 1 > ::type FirstList;
    
    typedef typename AllCombinations<ShrinkedRangeVector,FirstList>::type type;
};

}//namespace PMacc
