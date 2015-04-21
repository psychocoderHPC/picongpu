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

#include "communication/manager_common.h"
#include "types.h"

#include <mpi.h>
#include <string>
#include <sstream>
#include <fstream>



namespace PMacc
{
namespace debug
{

class CrashDump
{
public:

    void dumpToFile(const std::string& text, const std::string& filePrefix)
    {
        std::stringstream stringRank;
        stringRank << rank;

        std::string filename(filePrefix +
                             std::string(".") +
                             stringRank.str() +
                             std::string(".crashDump"));

        std::ofstream outFile;

        outFile.open(filename.c_str(), std::ofstream::out | std::ostream::trunc);
        if (!outFile)
        {
            std::cerr << "[CrashDump] Can't open file '" << std::endl;
        }
        else
        {
            outFile << text;
            outFile.flush();
            outFile << std::endl;
            outFile.close();
        }
    }

    void init()
    {
        MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    }

        static CrashDump& getInstance()
    {
        static CrashDump instance;
        return instance;
    }

private:

    CrashDump() : rank(-1)
    {

    }

    CrashDump(const CrashDump& cc);

    int rank;
};

} //namespace debug
} //namespace PMacc
