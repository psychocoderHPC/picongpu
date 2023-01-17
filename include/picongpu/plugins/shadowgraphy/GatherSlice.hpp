/* Copyright 2023 Rene Widera
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

#include "picongpu/simulation_defines.hpp"

#include <mpi.h>


namespace picongpu
{
    namespace plugins
    {
        namespace shadowgraphy
        {
            //! Gather data of a 2D Cartesian host buffer into a single MPI rank's host memory.
            class GatherSlice
            {
            private:
                MPI_Comm gatherComm = MPI_COMM_NULL;
                // gather rank zero will hold final data
                int gatherRank = -1;
                // number of ranks participating into the gather operation
                int numRanksInPlane = 0;

            public:
                GatherSlice()
                {
                }

                virtual ~GatherSlice()
                {
                    if(gatherComm != MPI_COMM_NULL)
                    {
                        auto err = MPI_Comm_free(&gatherComm);
                        if(err != MPI_SUCCESS)
                            std::cerr << __FILE__ << ":" << __LINE__ << "MPI_Comm_free failed." << std::endl;
                        gatherComm = MPI_COMM_NULL;
                    }
                    gatherRank = -1;
                    numRanksInPlane = 0;
                }

                /** Check if MPI rank is the gather master rank.
                 *
                 * The master will return the data when calling gatherSlice().
                 *
                 * @return True if MPI ranks is returning the gathered data in gatherSlice().
                 */
                bool isMaster() const
                {
                    return gatherRank == 0;
                }

                /** Check if MPI has the gathered data.
                 *
                 * @return True if MPI ranks is returning the gathered data in gatherSlice().
                 */
                bool hasResult() const
                {
                    return isMaster();
                }

                /** Query if MPI rank is part of the gather group.
                 *
                 * @return True if MPI rank is taking part on the gather operation, else false.
                 */
                bool isParticipating() const
                {
                    return gatherRank != -1;
                }

                /** Announce participation of the MPI rank in the gather operation
                 *
                 * @attention Must be called from all MPI ranks even if they to not participate.
                 *
                 * @param isActive True if MPI rank has data to gather, else false.
                 */
                void participate(bool isActive)
                {
                    int countRanks;
                    int globalMpiRank;
                    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &countRanks));
                    std::vector<int> allRank(countRanks);
                    std::vector<int> groupRanks(countRanks);
                    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &globalMpiRank));

                    if(!isActive)
                        globalMpiRank = -1;

                    // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                    __getTransactionEvent().waitForFinished();
                    MPI_CHECK(MPI_Allgather(&globalMpiRank, 1, MPI_INT, allRank.data(), 1, MPI_INT, MPI_COMM_WORLD));

                    int numRanks = 0;
                    for(int i = 0; i < countRanks; ++i)
                    {
                        if(allRank[i] != -1)
                        {
                            std::cout << "allRank[i] i=" << i << " value=" << allRank[i] << std::endl;
                            groupRanks[numRanks] = allRank[i];
                            numRanks++;
                        }
                    }
                    numRanksInPlane = numRanks;

                    MPI_Group group = MPI_GROUP_NULL;
                    MPI_Group newgroup = MPI_GROUP_NULL;
                    MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &group));
                    MPI_CHECK(MPI_Group_incl(group, numRanks, groupRanks.data(), &newgroup));

                    MPI_CHECK(MPI_Comm_create(MPI_COMM_WORLD, newgroup, &gatherComm));

                    if(globalMpiRank != -1)
                    {
                        MPI_CHECK(MPI_Comm_rank(gatherComm, &gatherRank));
                        std::cout << "gather rank=" << gatherRank << std::endl;
                    }
                    MPI_CHECK(MPI_Group_free(&group));
                    MPI_CHECK(MPI_Group_free(&newgroup));
                }


                /**
                 *
                 * Must be called by all participating MPI ranks.
                 * If a non-participating MPI rank is calling the method the returned buffer will be empty.
                 *
                 * @tparam T_DataType slice buffer data type
                 * @param localInputSlice buffer with local slice data
                 * @param globalSliceExtent extent in elements of the global slice
                 * @param localSliceOffset local offset in elements relative to the global slice origin
                 * @return buffer with gathered slice data (only master has valid data)
                 */
                template<typename T_DataType>
                auto gatherSlice(
                    std::shared_ptr<HostBufferIntern<T_DataType, DIM2>> localInputSlice,
                    DataSpace<DIM2> globalSliceExtent,
                    DataSpace<DIM2> localSliceOffset) const
                {
                    using ValueType = T_DataType;
                    // guard against wrong usage, only ranks which are participating into the gather are allowed
                    if(!isParticipating())
                        return std::shared_ptr<HostBufferIntern<ValueType, DIM2>>{};

                    pmacc::GridController<simDim>& con = pmacc::Environment<simDim>::get().GridController();
                    auto numDevices = con.getGpuNodes();

                    std::cout << "[" << gatherRank << "]"
                              << " num devices in slice plane =" << numRanksInPlane << std::endl;

                    // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                    __getTransactionEvent().waitForFinished();
                    // get number of elements per participating mpi rank
                    auto extentPerDevice = std::vector<DataSpace<DIM2>>(numRanksInPlane);

                    auto localSliceSize = localInputSlice->getDataSpace();

                    std::cout << "[" << gatherRank << "]"
                              << " start gather extents " << localSliceSize.toString() << std::endl;
                    // gather extents
                    MPI_CHECK(MPI_Gather(
                        reinterpret_cast<int*>(&localSliceSize),
                        2,
                        MPI_INT,
                        reinterpret_cast<int*>(extentPerDevice.data()),
                        2,
                        MPI_INT,
                        0,
                        gatherComm));

                    if(isMaster())
                    {
                        for(int i = 0; i < numRanksInPlane; ++i)
                        {
                            std::cout << "[" << gatherRank << "]"
                                      << " extent recive=" << extentPerDevice[i].toString() << std::endl;
                        }
                    }

                    std::cout << "[" << gatherRank << "]"
                              << " end gather extents" << std::endl;

                    auto offsetPerDevice = std::vector<DataSpace<DIM2>>(numRanksInPlane);

                    std::cout << "[" << gatherRank << "]"
                              << " start gather offsets " << localSliceOffset.toString() << std::endl;

                    // gather offsets
                    MPI_CHECK(MPI_Gather(
                        reinterpret_cast<int*>(&localSliceOffset),
                        2,
                        MPI_INT,
                        reinterpret_cast<int*>(offsetPerDevice.data()),
                        2,
                        MPI_INT,
                        0,
                        gatherComm));

                    if(isMaster())
                    {
                        for(int i = 0; i < numRanksInPlane; ++i)
                        {
                            std::cout << "[" << gatherRank << "]"
                                      << " offset recive=" << offsetPerDevice[i].toString() << std::endl;
                        }
                    }

                    std::cout << "[" << gatherRank << "]"
                              << " end gather offsets" << std::endl;

                    std::vector<int> displs(numRanksInPlane);
                    std::vector<int> count(numRanksInPlane);
                    // @todo replace by std::scan
                    int offset = 0;
                    int globalNumElements = 0u;

                    if(isMaster())
                    {
                        for(int i = 0; i < numRanksInPlane; ++i)
                        {
                            std::cout << "[" << gatherRank << "]"
                                      << " offset=" << offset << std::endl;

                            displs[i] = offset * sizeof(ValueType);
                            count[i] = extentPerDevice[i].productOfComponents() * sizeof(ValueType);
                            offset += extentPerDevice[i].productOfComponents();
                            globalNumElements += extentPerDevice[i].productOfComponents();

                            std::cout << "[" << gatherRank << "]"
                                      << " extentPerDevice[" << i << "]=" << extentPerDevice[i] << std::endl;
                            std::cout << "[" << gatherRank << "]"
                                      << " displs[" << i << "]=" << displs[i] << std::endl;
                            std::cout << "[" << gatherRank << "]"
                                      << " count[" << i << "]=" << count[i] << std::endl;
                        }
                    }
                    std::cout << "[" << gatherRank << "]"
                              << " globalNumElements=" << globalNumElements << std::endl;

                    // gather all data from other ranks
                    auto allData = std::vector<ValueType>(globalNumElements);
                    int localNumElements = localSliceSize.productOfComponents();

                    MPI_CHECK(MPI_Gatherv(
                        reinterpret_cast<char*>(localInputSlice->getDataBox().getPointer()),
                        localNumElements * sizeof(ValueType),
                        MPI_CHAR,
                        reinterpret_cast<char*>(allData.data()),
                        count.data(),
                        displs.data(),
                        MPI_CHAR,
                        0,
                        gatherComm));

                    std::cout << "[" << gatherRank << "]"
                              << " finish MPI_Gatherv" << std::endl;

                    std::shared_ptr<HostBufferIntern<ValueType, DIM2>> globalField;
                    if(isMaster())
                    {
                        // globalNumElements is only on the master rank valid
                        PMACC_VERIFY_MSG(
                            globalSliceExtent.productOfComponents() == globalNumElements,
                            "Expected and gathered number of elements differ.");

                        globalField = std::make_shared<HostBufferIntern<ValueType, DIM2>>(globalSliceExtent);
                        auto globalFieldBox = globalField->getDataBox();

                        // aggregate data of all MPI ranks into a single 2D buffer
                        for(int dataSetNumber = 0; dataSetNumber < numRanksInPlane; ++dataSetNumber)
                        {
                            for(int y = 0; y < extentPerDevice[dataSetNumber].y(); ++y)
                                for(int x = 0; x < extentPerDevice[dataSetNumber].x(); ++x)
                                {
                                    globalFieldBox(DataSpace<DIM2>(x, y) + offsetPerDevice[dataSetNumber]) = allData
                                        [displs[dataSetNumber] / sizeof(ValueType)
                                         + y * extentPerDevice[dataSetNumber].x() + x];
                                }
                        }
                    }
                    return globalField;
                }
            };

        } // namespace shadowgraphy
    } // namespace plugins
} // namespace picongpu