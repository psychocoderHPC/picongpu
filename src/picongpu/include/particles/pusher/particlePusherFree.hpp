/**
 * Copyright 2013-2015 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz
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

#include "types.h"

namespace picongpu
{
    namespace particlePusherFree
    {
        template<class Velocity, class Gamma>
        struct Push
        {
            template<
                typename T_Acc,
                typename EType,
                typename BType,
                typename PosType,
                typename MomType,
                typename MassType,
                typename ChargeType>
            ALPAKA_FN_ACC void operator()(
                T_Acc const & acc,
                BType const & bField,
                EType const & eField,
                PosType & pos,
                MomType const & mom,
                MassType const & mass,
                ChargeType const & charge) const
            {

                Velocity velocity;
                const PosType vel = velocity(mom, mass);


                for(uint32_t d=0;d<simDim;++d)
                {
                    pos[d] += (vel[d] * DELTA_T) / cellSize[d];
                }
            }
        };
    } //namespace
}
