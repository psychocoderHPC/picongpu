/* Copyright 2022-2023 Brian Marre, Rene Widera
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

#include "picongpu/particles/atomicPhysics2/localHelperFields/LocalRateCacheField.hpp"

#include <memory>
#include <stdexcept>


namespace picongpu::particles::atomicPhysics2::stage
{
    /** pre-simulation stage initiating the localRateCacheField for atomicPhysics
     *
     * @tparam T_IonSpecies species for which to call the functor
     */
    template<typename T_IonSpecies>
    struct CreateLocalRateCacheField
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_IonSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        template<typename T_MappingDescription>
        ALPAKA_FN_HOST void operator()(DataConnector& dataConnector, T_MappingDescription const& mappingDesc) const
        {
            auto localRateCacheField
                = std::make_unique<picongpu::particles::atomicPhysics2::localHelperFields::
                                       LocalRateCacheField<picongpu::MappingDesc, IonSpecies>>(mappingDesc);
            dataConnector.consume(std::move(localRateCacheField));
        }
    };
} // namespace picongpu::particles::atomicPhysics2::stage