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

#ifdef GOL_IMG_PNG
    #include <pngwriter.h>
#endif

#include "types.h"                  // DIM2
#include "dimensions/DataSpace.hpp"

#include <string>                   // std::string

namespace gol
{
    struct GrayImgWriter
    {
        template<
            class DBox>
        void operator() (
            DBox const & data,
            PMacc::DataSpace<DIM2> const & dataSize,
            std::string const & sFileNameWithoutExt)
        {
#ifdef GOL_IMG_PNG
            std::string const sFileName(sFileNameWithoutExt+".png");
            pngwriter png(dataSize.x(), dataSize.y(), 0, sFileName.c_str());
            png.setcompressionlevel(9);

            for (int y = 0; y < dataSize.y(); ++y)
            {
                for (int x = 0; x < dataSize.x(); ++x)
                {
                    float p = data[y][x];
                    png.plot(x + 1, dataSize.y() - y, p, p, p);
                }
            }
            png.close();
#else
            std::string const sFileName(sFileNameWithoutExt+".tga");
            std::ofstream ofs(
                sFileName,
                std::ofstream::out | std::ofstream::binary);
            if(!ofs.is_open())
            {
                throw std::invalid_argument("Unable to open file: "+sFileName);
            }

            std::size_t const uiNumCols(dataSize.x());
            std::size_t const uiNumRows(dataSize.y());

            // Copy the data and multiply everything by 255.
            std::vector<std::uint8_t> dataCopy(uiNumCols * uiNumRows);
            
            for(std::size_t y(0); y < dataSize.y(); ++y)
            {
                for(std::size_t x(0); x < dataSize.x(); ++x)
                {
                    dataCopy[y*uiNumCols + x] = data[y][x]*255u;
                }
            }

            // Write tga image header.
            ofs.put(0x00);                      // Number of Characters in Identification Field.
            ofs.put(0x00);                      // Color Map Type.
            ofs.put(0x03);                      // Image Type Code.
            ofs.put(0x00);                      // Color Map Origin.
            ofs.put(0x00);
            ofs.put(0x00);                      // Color Map Length.
            ofs.put(0x00);
            ofs.put(0x00);                      // Color Map Entry Size.
            ofs.put(0x00);                      // X Origin of Image.
            ofs.put(0x00);
            ofs.put(0x00);                      // Y Origin of Image.
            ofs.put(0x00);
            ofs.put((uiNumCols & 0xFF));        // Width of Image.
            ofs.put((uiNumCols >> 8) & 0xFF);
            ofs.put((uiNumRows & 0xFF));        // Height of Image.
            ofs.put((uiNumRows >> 8) & 0xFF);
            ofs.put(0x08);                      // Image Pixel Size.
            ofs.put(0x20);                      // Image Descriptor Byte.
            // Write data.
            ofs.write(
                //reinterpret_cast<char const *>(data.getPointer()),
                reinterpret_cast<char const *>(dataCopy.data()),
                uiNumCols * uiNumRows * sizeof(std::uint8_t));
#endif
        }
    };
}
