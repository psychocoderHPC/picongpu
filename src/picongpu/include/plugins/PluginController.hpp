/**
 * Copyright 2013-2015 Axel Huebl, Benjamin Schneider, Felix Schmitt,
 *                     Heiko Burau, Rene Widera, Richard Pausch,
 *                     Benjamin Worpitz
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
#include "simulation_defines.hpp"
#include "simulation_types.hpp"

#include "plugins/CountParticles.hpp"
#include "plugins/EnergyParticles.hpp"
#include "plugins/EnergyFields.hpp"
#include "plugins/SumCurrents.hpp"
#include "plugins/PositionsParticles.hpp"
#include "plugins/BinEnergyParticles.hpp"
#if(ENABLE_HDF5 == 1)
#include "plugins/PhaseSpace/PhaseSpaceMulti.hpp"
#endif

#if (ENABLE_INSITU_VOLVIS == 1)
#include "plugins/InSituVolumeRenderer.hpp"
#endif

#if(ENABLE_RADIATION == 1)
#include "plugins/radiation/parameters.hpp"
#include "plugins/radiation/Radiation.hpp"
#endif

#include "simulation_classTypes.hpp"

#include "mappings/kernel/MappingDescription.hpp"

#if(PIC_ENABLE_LIVE_VIEW == 1)
#include "plugins/LiveViewPlugin.hpp"
#endif
#include "plugins/ILightweightPlugin.hpp"
#include "plugins/ISimulationPlugin.hpp"

#if(PIC_ENABLE_PNG==1)
#include "plugins/output/images/PngCreator.hpp"
#endif


/// That's an abstract plugin for Png and Binary Density output
/// \todo rename PngPlugin to ImagePlugin or similar
#include "plugins/PngPlugin.hpp"

#if(SIMDIM==DIM3)
#include "plugins/IntensityPlugin.hpp"
#endif
//#include "plugins/SliceFieldPrinterMulti.hpp"

#include "plugins/output/images/Visualisation.hpp"

#include <list>

#include "plugins/ISimulationPlugin.hpp"

#if (ENABLE_HDF5 == 1)
#include "plugins/hdf5/HDF5Writer.hpp"
#include "plugins/makroParticleCounter/PerSuperCell.hpp"
#endif

#if (ENABLE_ADIOS == 1)
#include "plugins/adios/ADIOSWriter.hpp"
#endif

namespace picongpu
{

using namespace PMacc;

/**
 * Plugin management controller for user-level plugins.
 */
class PluginController : public ILightweightPlugin
{
private:

    std::list<ISimulationPlugin*> plugins;

    template<typename T_Type>
    struct PushBack
    {

        template<typename T>
        void operator()(T& list)
        {
            list.push_back(new T_Type());
        }
    };

    /** apply the 1st vector component to the 2nd
     *
     * @tparam T_TupleVector vector of type PMacc::math::CT::vector<dataType,plugin>
     *                       with two components
     */
    template<
        typename T_TupleVector>
    struct ApplyDataToPlugin :
        bmpl::apply1<
            typename PMacc::math::CT::At_c<T_TupleVector, 0>::type,
            typename PMacc::math::CT::At_c<T_TupleVector, 1>::type>
    {};

    /* define stand alone plugins*/
    using StandalonePlugins =
        bmpl::vector<
            EnergyFields,
            SumCurrents
#if(SIMDIM==DIM3)
          , IntensityPlugin
#endif
#if (ENABLE_INSITU_VOLVIS == 1)
          , InSituVolumeRenderer
#endif
#if (ENABLE_ADIOS == 1)
          , adios::ADIOSWriter
#endif
#if (ENABLE_HDF5 == 1)
          , hdf5::HDF5Writer
#endif
        >;

    /* define field plugins */

//#if BOOST_COMP_MSVC
    using FieldPlugins =
        bmpl::vector</*SliceFieldPrinterMulti<FieldB>, SliceFieldPrinterMulti<FieldE>, SliceFieldPrinterMulti<FieldJ>*/>;
//#else
//    using UnspecializedFieldPlugins =
//        bmpl::vector<
//            /*SliceFieldPrinterMulti<bmpl::_1>*/
//        >;
//
//    using AllFields =
//        bmpl::vector<
//            FieldB,
//            FieldE,
//            FieldJ
//        >;
//
//    using CombinedUnspecializedFieldPlugins =
//        AllCombinations<
//            bmpl::vector<UnspecializedFieldPlugins, AllFields>
//        >::type;
//
//    using FieldPlugins =
//        bmpl::transform<
//            CombinedUnspecializedFieldPlugins,
//            ApplyDataToPlugin<bmpl::_1>
//        >::type;
//#endif

//#if BOOST_COMP_MSVC
    using SpeciesPlugins =
        bmpl::vector <
            CountParticles<PIC_Electrons>,
            CountParticles<PIC_Ions>,
            EnergyParticles<PIC_Electrons>,
            EnergyParticles<PIC_Ions>,
            BinEnergyParticles<PIC_Electrons>,
            BinEnergyParticles<PIC_Ions>,
            PositionsParticles<PIC_Electrons>,
            PositionsParticles<PIC_Ions>
#if(PIC_ENABLE_LIVE_VIEW == 1)
          , LiveViewPlugin<PIC_Electrons>
          , LiveViewPlugin<PIC_Ions>
#endif
#if(ENABLE_RADIATION == 1)
          , Radiation<PIC_Electrons>
          , Radiation<PIC_Ions>
#endif
#if(PIC_ENABLE_PNG==1)
          , PngPlugin< Visualisation<PIC_Electrons, PngCreator> >
          , PngPlugin< Visualisation<PIC_Ions, PngCreator> >
#endif
#if(ENABLE_HDF5 == 1)
          , PerSuperCell<bmpl::_1>
          , PhaseSpaceMulti<particles::shapes::Counter::ChargeAssignment, PIC_Electrons>
          , PhaseSpaceMulti<particles::shapes::Counter::ChargeAssignment, PIC_Ions>
#endif
        >;
//#else
//    /* define species plugins */
//    using UnspecializedSpeciesPlugins =
//        bmpl::vector<
//            CountParticles<bmpl::_1>,
//            EnergyParticles<bmpl::_1>,
//            BinEnergyParticles<bmpl::_1>,
//            PositionsParticles<bmpl::_1>
//#if(PIC_ENABLE_LIVE_VIEW == 1)
//          , LiveViewPlugin<bmpl::_1>
//#endif
//#if(ENABLE_RADIATION == 1)
//          , Radiation<bmpl::_1>
//#endif
//#if(PIC_ENABLE_PNG==1)
//          , PngPlugin< Visualisation<bmpl::_1, PngCreator> >
//#endif
//#if(ENABLE_HDF5 == 1)
//          , PerSuperCell<FieldB>
//          , PhaseSpaceMulti<particles::shapes::Counter::ChargeAssignment, bmpl::_1>
//#endif
//        >;
//
//    using CombinedUnspecializedSpeciesPlugins =
//        AllCombinations<
//            bmpl::vector<UnspecializedSpeciesPlugins, VectorAllSpecies>
//        >::type;
//
//    using SpeciesPlugins =
//        bmpl::transform<
//            CombinedUnspecializedSpeciesPlugins,
//            ApplyDataToPlugin<bmpl::_1>
//        >::type;
//#endif

    /* create sequence with all plugins*/
    using AllPlugins =
        MakeSeq<
            StandalonePlugins,
            FieldPlugins,
            SpeciesPlugins
        >::type;

    /**
     * Initialises the controller by adding all user plugins to its internal list.
     */
    virtual void init()
    {
        ForEach<AllPlugins, PushBack<bmpl::_1> > pushBack;
        pushBack(forward(plugins));
    }

public:

    PluginController()
    {
        init();
    }

    virtual ~PluginController()
    {

    }

    void setMappingDescription(MappingDesc *cellDescription)
    {
        assert(cellDescription != NULL);

        for (std::list<ISimulationPlugin*>::iterator iter = plugins.begin();
             iter != plugins.end();
             ++iter)
        {
            (*iter)->setMappingDescription(cellDescription);
        }
    }

    virtual void pluginRegisterHelp(po::options_description&)
    {
        // no help required at the moment
    }

    std::string pluginGetName() const
    {
        return "PluginController";
    }

    void notify(uint32_t)
    {

    }

    virtual void pluginUnload()
    {
        for (std::list<ISimulationPlugin*>::iterator iter = plugins.begin();
             iter != plugins.end();
             ++iter)
        {
            __delete(*iter);
        }
        plugins.clear();
    }
};

}
