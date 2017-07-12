/* Copyright 2013-2017 Rene Widera
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
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

#include "particles/boostExtension/InheritGenerators.hpp"
#include "compileTime/conversion/MakeSeq.hpp"
#include "particles/particleFilter/system/TrueFilter.hpp"
#include "particles/particleFilter/system/DefaultFilter.hpp"

#include "particles/memory/frames/NullFrame.hpp"

#include <boost/mpl/list.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/copy.hpp>
#include <boost/mpl/back_inserter.hpp>
#include <boost/mpl/front_inserter.hpp>

namespace PMacc
{



template<typename UserTypeList = bmpl::vector<NullFrame> >
    class FilterFactory
{
public:

    typedef
    typename LinearInherit
    <
        typename MakeSeq<
           DefaultFilter<> ,
           UserTypeList,
           TrueFilter
        >::type
    >::type FilterType;

};

}//namespace PMacc



