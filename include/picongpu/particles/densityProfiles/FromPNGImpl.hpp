/** Copyright 2013-2019 Felix Schmitt, Rene Widera
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

#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/memory/boxes/DataBoxDim1Access.hpp>
#include "picongpu/simulation/control/MovingWindow.hpp"
#include <pmacc/dataManagement/DataConnector.hpp>

#include <pngwriter.h>

namespace picongpu
{
namespace densityProfiles
{

    template<typename T_ParamClass>
    struct FromPNGImpl : public T_ParamClass
    {
        typedef T_ParamClass ParamClass;

        template<typename T_SpeciesType>
        struct apply
        {
            using type = FromPNGImpl< ParamClass >;
        };

        HINLINE FromPNGImpl( uint32_t currentStep )
        {
            loadPNG();
            SubGrid< simDim > const & subGrid = Environment< simDim >::get().SubGrid();
            totalGpuOffset = subGrid.getLocalDomain( ).offset;
            const uint32_t numSlides = MovingWindow::getInstance( ).getSlideCounter( currentStep );
            totalGpuOffset.y( ) += numSlides * subGrid.getLocalDomain( ).size.y( );
        }

        /** Calculate the gas density from HDF5 file
         *
         * @param totalCellOffset total offset including all slides [in cells]
         */
        HDINLINE float_X operator()(DataSpace< simDim > const & totalCellOffset)
        {
            const DataSpace< simDim > localCellIdx(totalCellOffset - totalGpuOffset);
            DataSpace< simDim > offset( localCellIdx + SuperCellSize::toRT() * GuardSize::toRT() );
            if( simDim == DIM3 )
                offset[ 2 ] = SuperCellSize::toRT()[ 2 ];
            return precisionCast< float_X >( deviceDataBox(offset).x() );
        }

    private:

        void loadPNG()
        {
            DataConnector &dc = Environment<>::get().DataConnector();
            auto fieldTmp = dc.get< FieldTmp >( FieldTmp::getUniqueId( 0 ), true );
            auto& fieldBuffer = fieldTmp->getGridBuffer();

            deviceDataBox = fieldBuffer.getDeviceBuffer().getDataBox();

            GridController< simDim > & gc = Environment< simDim >::get().GridController();
            pmacc::Selection< simDim > const & localDomain = Environment< simDim >::get().SubGrid().getLocalDomain();

            DataSpace< simDim > globalDomOffset = Environment< simDim >::get().SubGrid().getGlobalDomain().offset;

            pngwriter png(1,1,0,"dummy.png");

            png.readfromfile(ParamClass::filename);
            int const pngWidth = png.getwidth();
            int const pngHeight = png.getheight();

            std::cout<<pngWidth<<"x"<<pngHeight<<" goff.y()"<<globalDomOffset.y()<<std::endl;

            DataSpace< simDim > guards = fieldBuffer.getGridLayout().getGuard();
            /* get the databox of the host buffer */
            auto dataBox = fieldBuffer.getHostBuffer().getDataBox().shift(guards);


            DataSpace< simDim > lOffset(localDomain.offset);

            for( int y = 0; y < localDomain.size.y(); ++y )
                for( int x = 0; x < localDomain.size.x(); ++x )
                {
                    auto buffCoordinate = DataSpace< simDim >::create(0);
                    buffCoordinate.x() = x;
                    buffCoordinate.y() = y;

                    // transform coordinate into png coordinate system, origin left-bottom
                    DataSpace<DIM2> imgCoordinate(
                        pngWidth - ( lOffset.x() + x + globalDomOffset.x() ),
                        pngHeight - ( lOffset.y() + y + globalDomOffset.y() )
                    );

                    if(x==0)
                        std::cout<<"co: "<<imgCoordinate.x()<<","<<globalDomOffset.y()<<std::endl;

                    if(
                        imgCoordinate.x() > 0 && imgCoordinate.x() <= pngWidth &&
                        imgCoordinate.y() > 0 && imgCoordinate.y() <= pngHeight
                    )
                        dataBox( buffCoordinate ).x() = 1.0 - png.dread(
                            imgCoordinate.x(),
                            imgCoordinate.y()
                        );
                    else
                        dataBox( buffCoordinate ).x() = 0._X;
                }

            /* copy host data to the device */
            fieldBuffer.hostToDevice();
            __getTransactionEvent().waitForFinished();
        }

        PMACC_ALIGN( deviceDataBox,FieldTmp::DataBoxType );
        PMACC_ALIGN( totalGpuOffset,DataSpace< simDim > );

    };
} //namespace densityProfiles
} //namespace picongpu
