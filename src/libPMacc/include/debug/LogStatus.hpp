/**
 * Copyright 2015 Rene Widera
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


#include <sstream>
#include <typeinfo>
#include <string>
#include <boost/algorithm/string/replace.hpp>
#include <boost/type_traits.hpp>
#include <cxxabi.h>

namespace PMacc
{
namespace debug
{


namespace detail
{

std::string demangle(const std::string name)
{
    int status = -3;

    char* demangledName = abi::__cxa_demangle(name.c_str(),
                                              NULL,
                                              NULL,
                                              &status
                                              );

    if (demangledName != NULL)
    {
        std::stringstream result;
        result << demangledName;

        free(demangledName);
        if (status == 0)
            return result.str();
    }

    return name;
}

template<typename T_Type, bool>
struct ToString;

template<typename T_Type>
struct ToString<T_Type, true>
{

    std::string operator()(const T_Type& object)
    {
        std::stringstream stream;

        stream << object;
        return stream.str();
    }

};

template<typename T_Type>
struct ToString<T_Type, false>
{

    std::string operator()(const T_Type& object)
    {
        return std::string("UNKNOWN");
    }

};

} //namespace detail

template<typename T_Type>
struct LogStatus
{
    typedef T_Type type;

    std::string operator()(const T_Type& object)
    {
        return std::string();
    }

};

template<typename T_Type>
struct ToString
{
    typedef T_Type type;
    static const bool isFundamental = boost::is_fundamental<type>::value;

    std::string operator()(const type& object)
    {
        return detail::ToString<type, isFundamental>()(object);
    }

};

template<typename T_Type>
std::string logStatus(const T_Type& object, const std::string& name = std::string(""))
{


    std::stringstream stream;


    if (!(name.empty()))
    {
        stream << name << " : ";
    }

    stream << detail::demangle(typeid (object).name());

    std::string tmp(LogStatus<T_Type>()(object));

    if (tmp.empty())
    {
        stream << std::string(" = " + ToString<T_Type>()(object));
    }
    tmp = std::string("\n") + tmp;
    boost::replace_all(tmp, "\n", "\n  ");

    return stream.str() + tmp;
}

} //namespace debug
} //namespace PMacc
