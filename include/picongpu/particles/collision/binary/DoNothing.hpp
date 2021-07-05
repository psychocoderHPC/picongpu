/* Copyright 2015-2021 Rene Widera, Pawel Ordyna
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

#include "picongpu/particles/collision/binary/RelativisticBinaryCollision.def"
#include "picongpu/unitless/collision.unitless"

#include <pmacc/random/distributions/Uniform.hpp>

#include <cmath>
#include <cstdio>
#include <type_traits>
#include <utility>

namespace picongpu
{
    namespace particles
    {
        namespace collision
        {
            namespace binary
            {
                namespace acc
                {
                    using namespace pmacc;
                    using namespace picongpu::particles::collision::precision;

                    /* Perform a single binary collision between two macro particles. (Device side functor)
                     *
                     * This algorithm was described in [Perez 2012] @url www.doi.org/10.1063/1.4742167.
                     * And it incorporates changes suggested in [Higginson 2020]
                     * @url www.doi.org/10.1016/j.jcp.2020.109450
                     */
                    struct DoNothing
                    {
                        float_COLL densitySqCbrt0;
                        float_COLL densitySqCbrt1;
                        uint32_t duplicationCorrection;
                        uint32_t potentialPartners;
                        float_COLL coulombLog;

                        /* Initialize device side functor.
                         *
                         * @param p_densitySqCbrt0 @f[ n_0^{2/3} @f] where @f[ n_0 @f] is the 1st species density.
                         * @param p_densitySqCbrt1 @f[ n_1^{2/3} @f] where @f[ n_1 @f] is the 2nd species density.
                         * @param p_potentialPartners number of potential collision partners for a macro particle in
                         *   the cell.
                         * @param p_coulombLog coulomb logarithm
                         */
                        HDINLINE DoNothing(
                            float_COLL p_densitySqCbrt0,
                            float_COLL p_densitySqCbrt1,
                            uint32_t p_potentialPartners,
                            float_COLL p_coulombLog)
                            : densitySqCbrt0(p_densitySqCbrt0)
                            , densitySqCbrt1(p_densitySqCbrt1)
                            , duplicationCorrection(1u)
                            , potentialPartners(p_potentialPartners)
                            , coulombLog(p_coulombLog){};


                        /** Execute the collision functor
                         *
                         * @param ctx collision context
                         * @param par0 1st colliding macro particle
                         * @param par1 2nd colliding macro particle
                         */
                        template<typename T_Context, typename T_Par0, typename T_Par1>
                        DINLINE void operator()(T_Context const& ctx, T_Par0& par0, T_Par1& par1) const
                        {

                        }
                    };
                } // namespace acc

                //! Host side binary collision functor
                struct DoNothing
                {
                    template<typename T_Species0, typename T_Species1>
                    struct apply
                    {
                        using type = DoNothing;
                    };

                    HINLINE DoNothing(uint32_t currentStep){};

                    /** create device manipulator functor
                     *
                     * @param acc alpaka accelerator
                     * @param offset (in supercells, without any guards) to the origin of the local domain
                     * @param workerCfg configuration of the worker
                     * @param density0 cell density of the 1st species
                     * @param density1 cell density of the 2nd species
                     * @param potentialPartners number of potential collision partners for a macro particle in
                     *   the cell.
                     * @param coulombLog Coulomb logarithm
                     */
                    template<typename T_WorkerCfg, typename T_Acc>
                    HDINLINE acc::DoNothing operator()(
                        T_Acc const& acc,
                        DataSpace<simDim> const& offset,
                        T_WorkerCfg const& workerCfg,
                        float_X const& density0,
                        float_X const& density1,
                        uint32_t const& potentialPartners,
                        float_X const& coulombLog) const
                    {
                        using namespace picongpu::particles::collision::precision;
                        return acc::DoNothing(
                            math::pow(precisionCast<float_COLL>(density0), 2.0_COLL / 3.0_COLL),
                            math::pow(precisionCast<float_COLL>(density1), 2.0_COLL / 3.0_COLL),
                            potentialPartners,
                            precisionCast<float_COLL>(coulombLog));
                    }

                    //! get the name of the functor
                    static HINLINE std::string getName()
                    {
                        return "DoNothing";
                    }
                };
            } // namespace binary
        } // namespace collision
    } // namespace particles
} // namespace picongpu
