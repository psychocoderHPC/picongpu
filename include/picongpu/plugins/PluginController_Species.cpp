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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/plugins/BinEnergyParticles.hpp"
#include "picongpu/plugins/CountParticles.hpp"
#include "picongpu/plugins/Emittance.hpp"
#include "picongpu/plugins/EnergyFields.hpp"
#include "picongpu/plugins/EnergyParticles.hpp"
#include "picongpu/plugins/PluginController.hpp"
#include "picongpu/plugins/output/images/PngCreator.hpp"
#include "picongpu/plugins/output/images/Visualisation.hpp"
#include "picongpu/plugins/transitionRadiation/TransitionRadiation.hpp"

#include <pmacc/assert.hpp>
/* That's an abstract plugin for image output with the possibility
 * to store the image as png file or send it via a sockets to a server.
 *
 * \todo rename PngPlugin to ImagePlugin or similar
 */
#include "picongpu/plugins/PngPlugin.hpp"

#if(SIMDIM == DIM3 && PIC_ENABLE_FFTW3 == 1 && ENABLE_OPENPMD == 1)
#    include "picongpu/plugins/shadowgraphy/Shadowgraphy.hpp"
#endif

#if(ENABLE_OPENPMD == 1)
#    include "picongpu/plugins/PhaseSpace/PhaseSpace.hpp"
#    include "picongpu/plugins/binning/BinningDispatcher.hpp"
#    include "picongpu/plugins/openPMD/openPMDWriter.hpp"
#    include "picongpu/plugins/particleCalorimeter/ParticleCalorimeter.hpp"
#endif

#include "picongpu/plugins/ChargeConservation.hpp"
#if(ENABLE_OPENPMD == 1)
#    include "picongpu/plugins/makroParticleCounter/PerSuperCell.hpp"
#endif

#if(ENABLE_ISAAC == 1) && (SIMDIM == DIM3)
#    include "picongpu/plugins/IsaacPlugin.hpp"
#endif

#if ENABLE_OPENPMD
#    include "picongpu/plugins/radiation/Radiation.hpp"
#    include "picongpu/plugins/radiation/VectorTypes.hpp"
#endif

#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#include "picongpu/plugins/Checkpoint.hpp"
#include "picongpu/plugins/ILightweightPlugin.hpp"
#include "picongpu/plugins/ISimulationPlugin.hpp"

#include <pmacc/mappings/kernel/MappingDescription.hpp>
#include <pmacc/meta/AllCombinations.hpp>

#include <list>
#include <memory>


namespace picongpu
{
    using namespace pmacc;

    void PluginController::loadSpeciesPlugins()
    {
        /* define species plugins */
        using UnspecializedSpeciesPlugins = pmacc::mp_list<
            plugins::multi::Master<EnergyParticles<boost::mpl::_1>>,
            plugins::multi::Master<CalcEmittance<boost::mpl::_1>>,
            plugins::multi::Master<BinEnergyParticles<boost::mpl::_1>>,
            CountParticles<boost::mpl::_1>,
            PngPlugin<Visualisation<boost::mpl::_1, PngCreator>>,
            plugins::transitionRadiation::TransitionRadiation<boost::mpl::_1>

#if(ENABLE_OPENPMD == 1)
            ,
            plugins::radiation::Radiation<boost::mpl::_1>,
            plugins::multi::Master<ParticleCalorimeter<boost::mpl::_1>>,
            plugins::multi::Master<PhaseSpace<particles::shapes::Counter::ChargeAssignment, boost::mpl::_1>>,
            PerSuperCell<boost::mpl::_1>
#endif
            >;

        using CombinedUnspecializedSpeciesPlugins
            = pmacc::AllCombinations<VectorAllSpecies, UnspecializedSpeciesPlugins>;

        using CombinedUnspecializedSpeciesPluginsEligible
            = pmacc::mp_copy_if<CombinedUnspecializedSpeciesPlugins, TupleSpeciesPlugin::IsEligible>;

        using SpeciesPlugins
            = pmacc::mp_transform<TupleSpeciesPlugin::Apply, CombinedUnspecializedSpeciesPluginsEligible>;


        meta::ForEach<SpeciesPlugins, PushBack<boost::mpl::_1>> pushBack;
        pushBack(plugins);
    }
} // namespace picongpu
