/**
 * Copyright 2013 Heiko Burau, Rene Widera
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

#include <types.h>
#include <boost/mpl/map.hpp>
#include <boost/mpl/erase_key.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/empty.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/back.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/mpl/advance.hpp>
#include <boost/mpl/begin.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_shifted_params.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/static_assert.hpp>

#include <particles/boostExtension/InheritLinearly.hpp>

namespace PMacc
{
namespace math
{



namespace mpl = boost::mpl;

template<typename T_Pair>
struct AlignedData
{
    typedef typename T_Pair::first Key;
    typedef typename T_Pair::second ValueType;

    PMACC_ALIGN(value, ValueType);

    HDINLINE AlignedData()
    {
    }

    HDINLINE AlignedData(const ValueType& value) : value(value)
    {
    }

    HDINLINE ValueType& operator[](const Key)
    {
        return value;
    }
};

template<typename T_Pair>
struct NativeData
{
    typedef typename T_Pair::first Key;
    typedef typename T_Pair::second ValueType;

    ValueType value;

    HDINLINE NativeData()
    {
    }

    HDINLINE NativeData(const ValueType& value) : value(value)
    {
    }

    HDINLINE ValueType& operator[](const Key)
    {
        return value;
    }

};

template<typename Map_, template<typename> class PODType = NativeData>
struct MapTuple : public InheritLinearly2<Map_,PODType>
{
    typedef Map_ Map;
    static const int dim = mpl::size<Map>::type::value;
    typedef MapTuple<Map,PODType> ThisType;

    template<class> struct result;

    template<class F, class TKey>
    struct result<F(TKey)>
    {
        typedef typename mpl::at<Map, TKey>::type& type;

    };

    template<class F, class TKey>
    struct result<const F(TKey)>
    {

        typedef const typename mpl::at<Map, TKey>::type& type;
    };

    template<typename T_Key>
    HDINLINE typename boost::result_of<ThisType(T_Key)>::type operator[](const T_Key key)
    {
        return (*(static_cast<
            PODType<bmpl::pair<T_Key,typename mpl::at<Map, T_Key>::type> >
            *>(this)))[key];
    }
};

} // math
} // PMacc
