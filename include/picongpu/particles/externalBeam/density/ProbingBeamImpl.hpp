/* Copyright 2014-2020 Pawel Ordyna, Rene Widera
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

#include "picongpu/particles/externalBeam/beam/ProbingBeam.hpp"
#include "picongpu/particles/externalBeam/beam/Side.hpp"
#include "pmacc/mappings/simulation/Selection.hpp"
namespace picongpu
{
    namespace particles
    {
        namespace externalBeam
        {
            namespace density
            {
                namespace detail
                {
                    using namespace picongpu::particles::externalBeam::beam;
                    using namespace picongpu::SI;
                    // Get the area of the cell side that is perpendicular to the beam propagation
                    template<typename T_Side>
                    struct GetCellAreaSI
                    {
                        static constexpr float_64 get();
                    };
                    template<>
                    struct GetCellAreaSI<XSide>
                    {
                        static constexpr float_64 get()
                        {
                            return CELL_DEPTH_SI * CELL_HEIGHT_SI;
                        }
                    };
                    template<>
                    struct GetCellAreaSI<XRSide>
                    {
                        static constexpr float_64 get()
                        {
                            return CELL_DEPTH_SI * CELL_HEIGHT_SI;
                        }
                    };
                    template<>
                    struct GetCellAreaSI<YSide>
                    {
                        static constexpr float_64 get()
                        {
                            return CELL_DEPTH_SI * CELL_WIDTH_SI;
                        }
                    };
                    template<>
                    struct GetCellAreaSI<YRSide>
                    {
                        static constexpr float_64 get()
                        {
                            return CELL_DEPTH_SI * CELL_WIDTH_SI;
                        }
                    };
                    template<>
                    struct GetCellAreaSI<ZSide>
                    {
                        static constexpr float_64 get()
                        {
                            return CELL_WIDTH_SI * CELL_HEIGHT_SI;
                        }
                    };
                    template<>
                    struct GetCellAreaSI<ZRSide>
                    {
                        static constexpr float_64 get()
                        {
                            return CELL_WIDTH_SI * CELL_HEIGHT_SI;
                        }
                    };

                } // namespace detail

                template<typename T_ParamClass>
                struct ProbingBeamImpl : public T_ParamClass
                {
                    using ParamClass = T_ParamClass;
                    using ProbingBeam = typename T_ParamClass::ProbingBeam;
                    // Defines from which side the beam enters the simulation box.
                    using Side = typename T_ParamClass::ProbingBeam::Side;
                    using SideCfg = typename ProbingBeam::SideCfg;

                    static constexpr float_64 photonFluxAtMaxBeamIntensity
                        = ParamClass::photonFluxAtMaxBeamIntensity_SI;

                private:
                    template<uint32_t idx>
                    using BeamToPicIdx_t = typename SideCfg::AxisSwapCT::template BeamToPicIdx<idx>::type;
                    template<uint32_t idx>
                    using PicToBeamIdx_t = typename SideCfg::AxisSwapCT::template PicToBeamIdx<idx>::type;
                    static constexpr float_X cellSizeCT[3] = {CELL_WIDTH, CELL_HEIGHT, CELL_DEPTH};
                    static constexpr float_X cellDepth{cellSizeCT[BeamToPicIdx_t<2u>::value]};
                    static constexpr bool reverse = SideCfg::Side::reverse[2];

                    static constexpr float_X reducedCellCenterBeam[3]
                        = {0.5_X, 0.5_X, 0.5_X * DELTA_T* SPEED_OF_LIGHT / cellDepth};
                    static constexpr float_X reducedCellCenter_x{reducedCellCenterBeam[PicToBeamIdx_t<0u>::value]};
                    static constexpr float_X reducedCellCenter_y{reducedCellCenterBeam[PicToBeamIdx_t<1u>::value]};
                    static constexpr float_X reducedCellCenter_z{reducedCellCenterBeam[PicToBeamIdx_t<2u>::value]};


                    // the area of the cell side perpendicular to the propagation
                    static constexpr float_64 CELL_AREA_SI = detail::GetCellAreaSI<Side>::get();
                    static constexpr float_X PHOTONS_IN_A_CELL
                        = static_cast<float_X>(photonFluxAtMaxBeamIntensity * SI::DELTA_T_SI * CELL_AREA_SI);
                    // the photon density in the cell in terms of the base density
                    static constexpr float_X REFERENCE_PHOTON_DENSITY = PHOTONS_IN_A_CELL / CELL_VOLUME / BASE_DENSITY;

                public:
                    template<typename T_SpeciesType>
                    struct apply
                    {
                        using type = ProbingBeamImpl<ParamClass>;
                    };

                    HINLINE explicit ProbingBeamImpl(uint32_t const& currentStep)
                        : probingBeam_m()
                        , currentStep_m(currentStep)
                    {
                        // TODO: modernize it offset  should be in subGrid already?
                        uint32_t const numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);
                        SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
                        DataSpace<simDim> const localCells = subGrid.getLocalDomain().size;
                        globalDomain_m = subGrid.getGlobalDomain();
                        globalDomain_m.offset.y() += numSlides * localCells.y();
                    }

                    DINLINE bool isInjectionBoundary(const DataSpace<simDim>& totalCellOffset)
                    {
                        const auto globalCellOffset = totalCellOffset - globalDomain_m.offset;
                        // Propagation axis
                        constexpr auto d{BeamToPicIdx_t<2u>::value};
                        bool boundary = false;
                        if constexpr(reverse)
                        {
                            boundary = globalCellOffset[d] == globalDomain_m.size[d] - 1;
                        }
                        else
                        {
                            boundary = globalCellOffset[d] == 0;
                        }
                        return boundary;
                    }

                    /** Calculate the normalized density
                     *
                     * @param totalCellOffset total offset including all slides [in cells]
                     */
                    DINLINE float_X operator()(const DataSpace<simDim>& totalCellOffset)
                    {
                        if(not isInjectionBoundary(totalCellOffset))
                        {
                            return 0.0_X;
                        }

                        float3_X totalCellOffsetVector;
                        if constexpr(simDim == 2u)
                        {
                            totalCellOffsetVector = float3_X{
                                static_cast<float_X>(totalCellOffset[0]),
                                static_cast<float_X>(totalCellOffset[1]),
                                0.0_X};
                        }
                        else
                        {
                            totalCellOffsetVector = precisionCast<float_X>(totalCellOffset);
                        }

                        float3_X reducedCellCenter = {reducedCellCenter_x, reducedCellCenter_y, reducedCellCenter_z};
                        if constexpr(reverse)
                        {
                            reducedCellCenter[BeamToPicIdx_t<2u>::value]
                                = 1.0_X - reducedCellCenter[BeamToPicIdx_t<2u>::value];
                        }

                        const float3_X globalCellPos((totalCellOffsetVector + reducedCellCenter) * cellSize);
                        // get the current position in the beam coordinate system
                        float3_X position_b = probingBeam_m.coordinateTransform(currentStep_m, globalCellPos);
                        // Get the relative beam intensity from the probing beam configuarion
                        float_X intensity = probingBeam_m(position_b);
                        return intensity * REFERENCE_PHOTON_DENSITY;
                    }

                private:
                    PMACC_ALIGN(probingBeam_m, ProbingBeam);
                    PMACC_ALIGN(globalDomain_m, pmacc::Selection<simDim>);
                    PMACC_ALIGN(currentStep_m, uint32_t);
                };
            } // namespace density
        } // namespace externalBeam
    } // namespace particles
} // namespace picongpu
