/* Copyright 2013-2023 Axel Huebl, Benjamin Schneider, Felix Schmitt,
 *                     Heiko Burau, Rene Widera, Richard Pausch,
 *                     Benjamin Worpitz, Erik Zenker, Finn-Ole Carstens,
 *                     Franz Poeschel
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once


#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#include "picongpu/plugins/ILightweightPlugin.hpp"
#include "picongpu/plugins/ISimulationPlugin.hpp"

#include <pmacc/mappings/kernel/MappingDescription.hpp>
#include <pmacc/meta/AllCombinations.hpp>

#include <list>
#include <memory>


namespace picongpu
{
    using namespace pmacc;

    /**
     * Plugin management controller for user-level plugins.
     */
    class PluginController : public ILightweightPlugin
    {
    private:
        std::list<std::shared_ptr<ISimulationPlugin>> plugins;

        template<typename T_Type>
        struct PushBack
        {
            template<typename T>
            void operator()(T& list)
            {
                list.push_back(std::make_shared<T_Type>());
            }
        };

        struct TupleSpeciesPlugin
        {
            enum Names
            {
                species = 0,
                plugin = 1
            };

            /** apply the 1st vector component to the 2nd
             *
             * @tparam T_TupleVector vector of type
             *                       pmacc::math::CT::vector< Species, Plugin >
             *                       with two components
             */
            template<typename T_TupleVector>
            using Apply = typename boost::mpl::
                apply1<pmacc::mp_at_c<T_TupleVector, plugin>, pmacc::mp_at_c<T_TupleVector, species>>::type;

            /** Check the combination Species+Plugin in the Tuple
             *
             * @tparam T_TupleVector with Species, Plugin
             */
            template<typename T_TupleVector>
            struct IsEligible
            {
                using Species = pmacc::mp_at_c<T_TupleVector, species>;
                using Solver = pmacc::mp_at_c<T_TupleVector, plugin>;

                static constexpr bool value
                    = particles::traits::SpeciesEligibleForSolver<Species, Solver>::type::value;
            };
        };

        /**
         * Initializes the controller by adding all user plugins to its internal list.
         */
        virtual void init()
        {
            loadStandAlonePlugins();
            loadSpeciesPlugins();
        }

        void loadStandAlonePlugins();

        void loadSpeciesPlugins();

    public:
        PluginController()
        {
            init();
        }

        ~PluginController() override = default;

        void setMappingDescription(MappingDesc* cellDescription) override
        {
            PMACC_ASSERT(cellDescription != nullptr);

            for(auto iter = plugins.begin(); iter != plugins.end(); ++iter)
            {
                (*iter)->setMappingDescription(cellDescription);
            }
        }

        void pluginRegisterHelp(po::options_description&) override
        {
            // no help required at the moment
        }

        std::string pluginGetName() const override
        {
            return "PluginController";
        }

        void notify(uint32_t) override
        {
        }

        void pluginUnload() override
        {
            plugins.clear();
        }
    };

} // namespace picongpu
