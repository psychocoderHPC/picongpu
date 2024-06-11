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
#include "picongpu/plugins/Checkpoint.hpp"
#include "picongpu/plugins/EnergyFields.hpp"
#include "picongpu/plugins/PluginController.hpp"
#include "picongpu/plugins/SumCurrents.hpp"
#include "picongpu/plugins/output/images/Visualisation.hpp"


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


namespace picongpu
{
    using namespace pmacc;

    void PluginController::loadStandAlonePlugins()
    {
        /* define stand alone plugins */
        using StandAlonePlugins = pmacc::mp_list<
            Checkpoint,
            EnergyFields,
            ChargeConservation,
            SumCurrents
#if(SIMDIM == DIM3 && PIC_ENABLE_FFTW3 == 1 && ENABLE_OPENPMD == 1)
            ,
            plugins::multi::Master<plugins::shadowgraphy::Shadowgraphy>
#endif
#if(ENABLE_OPENPMD == 1)
            ,
            plugins::binning::BinningDispatcher,
            plugins::multi::Master<openPMD::openPMDWriter>
#endif
#if(ENABLE_ISAAC == 1) && (SIMDIM == DIM3)
            ,
            isaacP::IsaacPlugin
#endif
            >;

        meta::ForEach<StandAlonePlugins, PushBack<boost::mpl::_1>> pushBack;
        pushBack(plugins);
    }
} // namespace picongpu
