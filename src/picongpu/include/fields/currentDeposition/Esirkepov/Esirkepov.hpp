/**
 * Copyright 2014 Axel Huebl, Heiko Burau, Rene Widera
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

namespace picongpu
{
namespace currentSolverEsirkepov
{

/**
 * \class Esirkepov implements the current deposition algorithm from T.Zh. Esirkepov
 * for an arbitrary particle assign function given as a template parameter.
 * See available shapes at "intermediateLib/particleShape".
 * paper: "Exact charge conservation scheme for Particle-in-Cell simulation
 *  with an arbitrary form-factor"
 */
template<unsigned DIM,typename ParticleAssign, typename NumericalCellType>
struct Esirkepov;

}// namespace currentSolverEsirkepov


} //namespace picongpu

#if(SIMDIM==DIM3)
#include "fields/currentDeposition/Esirkepov/Esirkepov3D.hpp"
#elif(SIMDIM==DIM2)
#include "fields/currentDeposition/Esirkepov/Esirkepov2D.hpp"
#endif
