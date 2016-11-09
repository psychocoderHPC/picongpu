/**
 * Copyright 2013-2016 Felix Schmitt, Rene Widera
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

#include "simulation_defines.hpp"
#include "memory/buffers/GridBuffer.hpp"
#include "memory/boxes/DataBoxDim1Access.hpp"
#include "simulationControl/MovingWindow.hpp"
#include "fields/Fields.hpp"
#include "dataManagement/DataConnector.hpp"

#include <pngwriter.h>

namespace picongpu
{

namespace gasProfiles
{

template<typename T_ParamClass>
struct FromPNGImpl : public T_ParamClass
{
    typedef T_ParamClass ParamClass;

    template<typename T_SpeciesType>
    struct apply
    {
        typedef FromPNGImpl<ParamClass> type;
    };

    HINLINE FromPNGImpl(uint32_t currentStep)
    {
        loadPNG();
        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
        totalGpuOffset = subGrid.getLocalDomain( ).offset;
    }

    /** Calculate the gas density from HDF5 file
     *
     * @param totalCellOffset total offset including all slides [in cells]
     */
    HDINLINE float_X operator()(const DataSpace<simDim>& totalCellOffset)
    {
        const DataSpace<simDim> localCellIdx(totalCellOffset - totalGpuOffset);
        DataSpace<simDim> offset(localCellIdx + SuperCellSize::toRT()*int(GUARD_SIZE));
        offset.z() = SuperCellSize::toRT().z();
        return precisionCast<float_X>(deviceDataBox(offset).x());
    }

private:

    void loadPNG()
    {
        DataConnector &dc = Environment<>::get().DataConnector();
        FieldTmp& fieldTmp = dc.getData<FieldTmp > (FieldTmp::getName(), true);
        PMACC_AUTO(&fieldBuffer, fieldTmp.getGridBuffer());

        deviceDataBox = fieldBuffer.getDeviceBuffer().getDataBox();

        GridController<simDim> &gc = Environment<simDim>::get().GridController();
        const PMacc::Selection<simDim>& localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();

        DataSpace<simDim> size = Environment<simDim>::get().SubGrid().getGlobalDomain().size;

        pngwriter png(1,1,0,"dummy.png");

        png.readfromfile(ParamClass::filename);
        DataSpace<simDim> guards = fieldBuffer.getGridLayout().getGuard();
        /* get the databox of the host buffer */
        PMACC_AUTO(
            dataBox,
            fieldBuffer.getHostBuffer().getDataBox().shift(guards)
        );

        DataSpace<simDim> offset(localDomain.offset);
        for( int y = 0; y < localDomain.size.y(); ++y )
            for( int x = 0; x < localDomain.size.x(); ++x )
            {
                //std::cout<<x<<","<<y<<" "<<png.dread(offset.x()+x,offset.y()+y)<<std::endl;
                dataBox( DataSpace<DIM3>( x, y, 0 ) ).x() = 1.0 - png.dread(
                    offset.x() + x + 1,
                    size.y() - ( offset.y() + y )
                );
            }

        /* copy host data to the device */
        fieldBuffer.hostToDevice();
        __getTransactionEvent().waitForFinished();


        return;
    }

    PMACC_ALIGN(deviceDataBox,FieldTmp::DataBoxType);
    PMACC_ALIGN(totalGpuOffset,DataSpace<simDim>);

};
} //namespace gasProfiles
} //namespace picongpu
