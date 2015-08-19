/**
 * Copyright 2013 Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
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

#ifndef ALGORITHM_HOST_FOREACH_HPP
#define ALGORITHM_HOST_FOREACH_HPP

#include "math/vector/Size_t.hpp"
#include "math/vector/Int.hpp"
#include "lambda/make_Functor.hpp"
#include <forward.hpp>

namespace PMacc
{
namespace algorithm
{
namespace host
{
namespace detail
{
    /** Return pseudo 3D-range of the zone as math::Int<dim> */
    template< uint32_t dim >
    struct GetRange;

    template<>
    struct GetRange<3u>
    {
        template<typename Zone>
        const math::Int<3u> operator()(const Zone p_zone) const
        {
            return math::Int<3u>(p_zone.size.x(), p_zone.size.y(), p_zone.size.z());
        }
    };
    template<>
    struct GetRange<2u>
    {
        template<typename Zone>
        const math::Int<3u> operator()(const Zone p_zone) const
        {
            return math::Int<3u>(p_zone.size.x(), p_zone.size.y(), 1);
        }
    };
    template<>
    struct GetRange<1u>
    {
        template<typename Zone>
        const math::Int<3u> operator()(const Zone p_zone) const
        {
            return math::Int<3u>(p_zone.size.x(), 1, 1);
        }
    };
} // namespace detail

/** Foreach algorithm (restricted to 3D)
 */
struct Foreach
{
    /* operator()(zone, cursor0, cursor1, ..., cursorN-1, functor or lambdaFun)
     *
     * \param zone Accepts currently only a zone::SphericZone object (e.g. containerObj.zone())
     * \param cursorN cursor for the N-th data source (e.g. containerObj.origin())
     * \param functor or lambdaFun either a functor with N arguments or a N-ary lambda function (e.g. _1 = _2)
     *
     * The functor or lambdaFun is called for each cell within the zone.
     * It is called like functor(*cursor0(cellId), ..., *cursorN(cellId))
     *
     */
    template<
        typename Zone,
        typename Functor,
        typename... TCs>
    void operator()(
        Zone const & p_zone,
        Functor const & functor,
        TCs && ... cs)
    {
        forEachShifted(
            p_zone,
            functor,
            cs(_zone.offset)...);
    }

private:
    /*
     *
     */
    template<
        typename Zone,
        typename Functor,
        typename... TShiftedCs>
    void forEachShifted(
        Zone const & p_zone,
        Functor const & functor,
        TShiftedCs && ... shiftedCs)
    {
        typename lambda::result_of::make_Functor<Functor>::type fun =
            lambda::make_Functor(functor);

        detail::GetRange<Zone::dim> getRange;
        for(int z = 0; z < getRange(p_zone).z(); z++)
        {
            for(int y = 0; y < getRange(p_zone).y(); y++)
            {
                for(int x = 0; x < getRange(p_zone).x(); x++)
                {
                    math::Int<Zone::dim> cellIndex =
                        math::Int<3u>(x, y, z).shrink<Zone::dim>();
                    fun(std::forward<TShiftedCs>(shiftedCs)...);
                }
            }
        }
    }
};

} // host
} // algorithm
} // PMacc

#endif // ALGORITHM_HOST_FOREACH_HPP
