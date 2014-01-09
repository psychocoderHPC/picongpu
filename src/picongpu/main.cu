/**
 * Copyright 2013 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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

/**
 * @mainpage PIConGPU-Frame
 *
 * Project with HZDR for porting their PiC-code to a GPU cluster.
 *
 * \image html picongpu.jpg
 *
 * @author Heiko Burau, Rene Widera, Wolfgang Hoenig, Felix Schmitt, Axel Huebl, Michael Bussmann, Guido Juckeland
 */
#define BOOST_MPL_AUX_HAS_TAG_HPP_INCLUDED

#include <boost/mpl/bool.hpp>
#include <boost/mpl/aux_/type_wrapper.hpp>
#include <boost/mpl/aux_/yes_no.hpp>

namespace boost { namespace mpl { namespace aux {
template< typename T, typename fallback_ = boost::mpl::bool_<false> >
        struct has_tag {
                struct gcc_3_2_wknd
                {
                        template< typename U >
                        static boost::mpl::aux::yes_tag test(   boost::mpl::aux::type_wrapper<U> const volatile* , boost::mpl::aux::type_wrapper<typename U::tag>* = 0 );
                        static boost::mpl::aux::no_tag test(...);
                };
                typedef boost::mpl::aux::type_wrapper<T> t_;
                static const bool value = sizeof(gcc_3_2_wknd::test(static_cast<t_*>(0))) == sizeof(boost::mpl::aux::yes_tag);
                typedef boost::mpl::bool_<value> type; };
}}}

#include <simulation_defines.hpp>
#include <mpi.h>
#include "communication/manager_common.h"

using namespace PMacc;
using namespace picongpu;

/*! start of PIConGPU
 *
 * @param argc count of arguments in argv
 * @param argv arguments of program start
 */
int main(int argc, char **argv)
{
    MPI_CHECK(MPI_Init(&argc, &argv));

    picongpu::simulation_starter::SimStarter sim;
    if (!sim.parseConfigs(argc, argv))
    {
        MPI_CHECK(MPI_Finalize());
        return 1;
    }

    sim.load();
    sim.start();
    sim.unload();

    MPI_CHECK(MPI_Finalize());

    return 0;
}
