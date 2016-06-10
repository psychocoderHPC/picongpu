/* This file is part of ISAAC.
 *
 * ISAAC is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * ISAAC is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with ISAAC.  If not, see <www.gnu.org/licenses/>. */

#pragma once

#include "isaac_defines.hpp"

namespace isaac
{

template <
	typename Type,
	size_t Size
>
struct IsaacVector
{
	Type value[Size];
	Type& operator[] (size_t const& index)
	{
		return value[index];
	}
	IsaacVector<Type,Size>& operator= (IsaacVector<Type,Size> const& rhs)
	{
		if (this != &rhs)
		{
			ISAAC_ELEM_ITERATE( index )
				value[index] = rhs[index];
		}
		return *this;
	}
	const IsaacVector<Type,Size> operator+ (IsaacVector<Type,Size> const& lhs,IsaacVector<Type,Size> const& rhs)
	{
		IsaacVector<Type,Size> result;
		ISAAC_ELEM_ITERATE( index )
			result[index] = lhs[index] + rhs[index];
		return result;
	}
};




} //namespace isaac;
