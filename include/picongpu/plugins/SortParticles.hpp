/* Copyright 2013-2022 Axel Huebl, Felix Schmitt, Heiko Burau,
 *                     Rene Widera, Richard Pausch, Benjamin Worpitz
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

#include "picongpu/particles/collision/detail/ListEntry.hpp"
#include "picongpu/particles/filter/All.hpp"
#include "picongpu/plugins/multi/multi.hpp"
#include "pmacc/particles/operations/Assign.hpp"
#include "pmacc/particles/operations/Deselect.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/math/operation.hpp>
#include <pmacc/memory/shared/Allocate.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/algorithm/ForEach.hpp>
#include <pmacc/particles/memory/buffers/MallocMCBuffer.hpp>

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>


namespace picongpu
{
    /** Sort particles per cell.
     */
    struct SortPar
    {
        /** accumulate particle energies
         *
         * @tparam T_ParBox pmacc::ParticlesBox, particle box type
         * @tparam T_Mapping mapper functor type
         *
         * @param pb particle memory
         * @param mapper functor to map a block to a supercell
         */
        template<typename T_ParBox, typename T_DeviceHeapHandle, typename T_Mapping, typename T_Worker>
        DINLINE void operator()(
            T_Worker const& worker,
            T_ParBox pb,
            T_DeviceHeapHandle deviceHeapHandle,
            T_Mapping mapper) const
        {
            using namespace pmacc::particles::operations;

            using SuperCellSize = typename T_ParBox::FrameType::SuperCellSize;
            constexpr uint32_t frameSize = pmacc::math::CT::volume<SuperCellSize>::type::value;

            using FramePtr = typename T_ParBox::FramePtr;

            PMACC_SMEM(worker, nppc, memory::Array<uint32_t, frameSize>);
            PMACC_SMEM(worker, parCellList, memory::Array<particles::collision::detail::ListEntry, frameSize>);
            PMACC_SMEM(worker, firstDestFrame, FramePtr);

            DataSpace<simDim> const superCellIdx
                = mapper.getSuperCellIndex(DataSpace<simDim>(cupla::blockIdx(worker.getAcc())));


            auto& superCell = pb.getSuperCell(superCellIdx);
            uint32_t numParticlesInSupercell = superCell.getNumParticles();

            if(numParticlesInSupercell == 0)
                return;

            auto accFilter = particles::filter::acc::All{};

            /* loop over all particles in the frame */
            auto forEachFrameElem = lockstep::makeForEach<frameSize>(worker);
            FramePtr firstFrame = pb.getFirstFrame(superCellIdx);

            prepareList(
                worker,
                forEachFrameElem,
                deviceHeapHandle,
                pb,
                firstFrame,
                numParticlesInSupercell,
                parCellList,
                nppc,
                accFilter);

            auto onlyMaster = lockstep::makeMaster(worker);
            onlyMaster(
                [&]()
                {
                    auto tmpFrame = FramePtr{};
                    for(int p = 0; p < numParticlesInSupercell; p += frameSize)
                    {
                        tmpFrame = pb.getEmptyFrame(worker);
                        pb.setAsFirstFrame(worker, tmpFrame, superCellIdx);
                    }
                    // last appended frame to front is first frame in the supercell
                    firstDestFrame = tmpFrame;
                });
            worker.sync();

            auto entryOffsetCtx = forEachFrameElem(
                [&](uint32_t const linearIdx)
                {
                    // exclusive scan implementation
                    int offset = 0;
                    for(int i = 0; i < linearIdx; i++)
                        offset += nppc[i];
#if 0
                    printf("%i offset %i\n", linearIdx, offset);
#endif
                    return offset;
                });

            forEachFrameElem(
                [&](uint32_t const linearIdx, int entryParOffset)
                {
                    uint32_t const numParInCell = parCellList[linearIdx].size;
                    uint32_t* parListStart = parCellList[linearIdx].ptrToIndicies;

                    for(uint32_t ii = 0; ii < numParInCell; ii++)
                    {
                        auto srcPar = particles::collision::detail::getParticle(pb, firstFrame, parListStart[ii]);
                        srcPar[multiMask_] = 0;

                        auto destPar = particles::collision::detail::getParticle(pb, firstDestFrame, entryParOffset);
                        destPar[multiMask_] = 1;
                        destPar[localCellIdx_] = linearIdx;

                        auto dstFilteredParticle
                            = pmacc::particles::operations::deselect<MakeSeq_t<multiMask, localCellIdx>>(destPar);
                        pmacc::particles::operations::assign(dstFilteredParticle, srcPar);
                        entryParOffset++;
                    }
#if 0
                    printf("created %i %i\n", linearIdx, entryParOffset);
#endif
                },
                entryOffsetCtx);

            worker.sync();

            onlyMaster(
                [&]()
                {
                    for(int p = 0; p < numParticlesInSupercell; p += frameSize)
                    {
                        pb.removeLastFrame(worker, superCellIdx);
                    }
                });
#if 0
            onlyMaster(
                [&]()
                {
                    FramePtr frame = pb.getFirstFrame(superCellIdx);
                    int i = 0;
                    int par = 0;
                    while(frame.isValid())
                    {
                        for(int p = 0; p < frameSize; p++)
                            if(frame[p][multiMask_] == 1)
                                par++;
                        i++;
                        frame = pb.getNextFrame(frame);
                    }
                    printf("frames left %i %i\n", i, par);
                });
#endif
            forEachFrameElem([&](uint32_t const linearIdx)
                             { parCellList[linearIdx].finalize(worker, deviceHeapHandle); });
        }
    };

    template<typename ParticlesType>
    class SortParticles : public plugins::multi::IInstance
    {
    public:
        struct Help : public plugins::multi::IHelp
        {
            /** creates an instance
             *
             * @param help plugin defined help
             * @param id index of the plugin, range: [0;help->getNumPlugins())
             */
            std::shared_ptr<IInstance> create(
                std::shared_ptr<IHelp>& help,
                size_t const id,
                MappingDesc* cellDescription) override
            {
                return std::shared_ptr<IInstance>(new SortParticles<ParticlesType>(help, id, cellDescription));
            }

            //! periodicity of computing the particle energy
            plugins::multi::Option<std::string> notifyPeriod
                = {"period",
                   "compute kinetic and total energy [for each n-th step] enable plugin by setting a non-zero value"};


            ///! method used by plugin controller to get --help description
            void registerHelp(
                boost::program_options::options_description& desc,
                std::string const& masterPrefix = std::string{}) override
            {
                notifyPeriod.registerHelp(desc, masterPrefix + prefix);
            }

            void expandHelp(
                boost::program_options::options_description& desc,
                std::string const& masterPrefix = std::string{}) override
            {
            }


            void validateOptions() override
            {
            }

            size_t getNumPlugins() const override
            {
                return notifyPeriod.size();
            }

            std::string getDescription() const override
            {
                return description;
            }

            std::string getOptionPrefix() const
            {
                return prefix;
            }

            std::string getName() const override
            {
                return name;
            }

            std::string const name = "SortParticles";
            //! short description of the plugin
            std::string const description = "sort particles per cell";
            //! prefix used for command line arguments
            std::string const prefix = ParticlesType::FrameType::getName() + std::string("_sort");
        };

        //! must be implemented by the user
        static std::shared_ptr<plugins::multi::IHelp> getHelp()
        {
            return std::shared_ptr<plugins::multi::IHelp>(new Help{});
        }

        SortParticles(std::shared_ptr<plugins::multi::IHelp>& help, size_t const id, MappingDesc* cellDescription)
            : m_cellDescription(cellDescription)
            , m_help(std::static_pointer_cast<Help>(help))
            , m_id(id)
        {
            // set how often the plugin should be executed while PIConGPU is running
            Environment<>::get().PluginConnector().setNotificationPeriod(this, m_help->notifyPeriod.get(id));
        }

        ~SortParticles() override
        {
        }

        /** this code is executed if the current time step is supposed to compute
         * the energy
         */
        void notify(uint32_t currentStep) override
        {
            // call the method that calls the plugin kernel
            sortParticles<CORE + BORDER>(currentStep);
        }


        void restart(uint32_t restartStep, std::string const& restartDirectory) override
        {
        }

        void checkpoint(uint32_t currentStep, std::string const& checkpointDirectory) override
        {
        }

    private:
        //! method to call analysis and plugin-kernel calls
        template<uint32_t AREA>
        void sortParticles(uint32_t currentStep)
        {
            DataConnector& dc = Environment<>::get().DataConnector();

            // use data connector to get particle data
            auto particles = dc.get<ParticlesType>(ParticlesType::FrameType::getName());

            auto const mapper = makeAreaMapper<AREA>(*m_cellDescription);

            auto mallocMCBuffer = dc.get<MallocMCBuffer<DeviceHeap>>(MallocMCBuffer<DeviceHeap>::getName());

            auto workerCfg = lockstep::makeWorkerCfg(SuperCellSize{});
            PMACC_LOCKSTEP_KERNEL(SortPar{}, workerCfg)
            (mapper.getGridDim())(
                particles->getDeviceParticlesBox(),
                mallocMCBuffer->getDeviceHeap()->getAllocatorHandle(),
                mapper);
#if 0
            std::cout << "sort " << currentStep << " " << ParticlesType::FrameType::getName() << std::endl;
#endif
        }

        MappingDesc* m_cellDescription;
        std::shared_ptr<Help> m_help;
        size_t m_id;
    };

} // namespace picongpu
