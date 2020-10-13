/* Copyright 2013-2020 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Sergei Bastrakov
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

#include "picongpu/boost_workaround.hpp"

#include "picongpu/ArgsParser.hpp"
#include <pmacc/Environment.hpp>
#include <pmacc/types.hpp>
#include <pmacc/simulationControl/PerfData.hpp>

#include <picongpu/simulation_defines.hpp>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>


namespace
{

    /** Run a PIConGPU simulation
     *
     * @param argc count of arguments in argv (same as for main() )
     * @param argv arguments of program start (same as for main() )
     */
    int runSimulation( int argc, char **argv )
    {
        double const startProgram = pmacc::PerfData::inst().getTime();

        using namespace picongpu;

        simulation_starter::SimStarter sim;
        auto const parserStatus = sim.parseConfigs( argc, argv );
        int errorCode = EXIT_FAILURE;

        switch( parserStatus )
        {
            case ArgsParser::Status::error:
                errorCode = EXIT_FAILURE;
                break;
            case ArgsParser::Status::success:
            {
                double const startSimLoad = pmacc::PerfData::inst().getTime();
                sim.load( );
                double const endSimLoad = pmacc::PerfData::inst().getTime();
                sim.start( );
                double const endSimStart = pmacc::PerfData::inst().getTime();
                sim.unload( );
                double const endSimUnload = pmacc::PerfData::inst().getTime();
                pmacc::PerfData::inst().pushRegions( "main-simulation-load", endSimLoad - startSimLoad);
                pmacc::PerfData::inst().pushRegions( "main-simulation-start", endSimStart - endSimLoad);
                pmacc::PerfData::inst().pushRegions( "main-simulation-unload", endSimUnload - endSimStart);
            }
                PMACC_FALLTHROUGH;
            case ArgsParser::Status::successExit:
                errorCode = 0;
                break;
        };

        Environment<>::get().globalSync();
        double const startMpiFinalize = pmacc::PerfData::inst().getTime();
        // finalize the pmacc context */
        pmacc::Environment<>::get( ).finalize( );

        double const endProgram = PerfData::inst().getTime();
        pmacc::PerfData::inst().pushRegions( "mpi-finalize", endProgram - startMpiFinalize);
        pmacc::PerfData::inst().pushRegions( "main-program", endProgram - startProgram);

        return errorCode;
    }

} // anonymous namespace

/** Start of PIConGPU
 *
 * @param argc count of arguments in argv
 * @param argv arguments of program start
 */
int main( int argc, char **argv )
{
    try
    {
        return runSimulation( argc, argv );
    }
    // A last-ditch effort to report exceptions to a user
    catch ( const std::exception & ex )
    {
        auto const typeName = std::string( typeid( ex ).name( ) );
        std::cerr << "Unhandled exception of type '" + typeName +
            "' with message '" + ex.what() + "', terminating\n";
        return EXIT_FAILURE;
    }
    catch ( ... )
    {
        std::cerr << "Unhandled exception of unknown type, terminating\n";
        return EXIT_FAILURE;
    }
}
