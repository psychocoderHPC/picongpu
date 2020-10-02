/* Copyright 2015-2020 Axel Huebl, Benjamin Worpitz
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
#include "picongpu/fields/currentInterpolation/None/None.def"
#include "picongpu/fields/differentiation/Curl.hpp"
#include "picongpu/algorithms/LinearInterpolateWithUpper.hpp"
#include "picongpu/traits/GetMargin.hpp"

#include <pmacc/traits/GetComponentsType.hpp>
#include <pmacc/dimensions/DataSpace.hpp>


namespace picongpu
{
namespace currentInterpolation
{
namespace detail
{
    template<uint32_t T_simDim, uint32_t T_plane>
    struct LinearInterpolateComponentPlaneUpper
    {
        static constexpr uint32_t dim = T_simDim;

        /* UpperMargin is actually 0 in direction of T_plane */
        using LowerMargin = typename pmacc::math::CT::make_Int<
            dim,
            0
        >::type;
        using UpperMargin = typename pmacc::math::CT::make_Int<
            dim,
            1
        >::type;

        template<typename DataBox>
        HDINLINE float_X operator()( DataBox const & field ) const
        {
            DataSpace< dim > const self;
            DataSpace< dim > up;
            up[(T_plane + 1) % dim] = 1;

            using Avg = LinearInterpolateWithUpper< dim >;

            typename Avg::template GetInterpolatedValue< ( T_plane + 2 ) % dim > const avg;

            return float_X( 0.5 ) * ( avg( field )[ T_plane ] + avg( field.shift( up ) )[ T_plane ] );
        }
    };

    /* shift a databox along a specific direction
     *
     * returns the identity (assume periodic symmetry) if direction is not
     * available, such as in a 2D simulation
     *
     * \todo accept a full CT::Vector and shift if possible
     * \todo call with CT::Vector of correct dimensionality that was created
     *       with AssignIfInRange...
     *
     * \tparam T_simDim maximum dimensionality of the mesh
     * \tparam T_direction (0)X (1)Y or (2)Z for the direction one wants to
     *                     shift to
     * \tparam isShiftAble auto-filled value that decides if this direction
     *                     is actually non-existent == periodic
     */
    template<
        uint32_t T_simDim,
        uint32_t T_direction,
        bool isShiftAble = ( T_direction < T_simDim )
    >
    struct ShiftMeIfYouCan
    {
        static constexpr uint32_t dim = T_simDim;
        static constexpr uint32_t dir = T_direction;

        HDINLINE ShiftMeIfYouCan()
        {
        }

        template< typename T_DataBox >
        HDINLINE T_DataBox operator()( T_DataBox const & dataBox ) const
        {
            DataSpace< dim > shift;
            shift[ dir ] = 1;
            return dataBox.shift( shift );
        }
    };

    template<
        uint32_t T_simDim,
        uint32_t T_direction
    >
    struct ShiftMeIfYouCan<
        T_simDim,
        T_direction,
        false
    >
    {
        HDINLINE ShiftMeIfYouCan()
        {
        }

        template< typename T_DataBox >
        HDINLINE T_DataBox operator()( T_DataBox const & dataBox ) const
        {
            return dataBox;
        }
    };

    /* that is not a "real" yee curl, but it looks a bit like it */
    template< typename T_Curl >
    struct ShiftCurl : T_Curl
    {
        using LowerMargin = typename traits::GetLowerMargin<T_Curl>::type;
        using UpperMargin = typename traits::GetUpperMargin<T_Curl>::type;

        template<class DataBox >
        HDINLINE typename DataBox::ValueType operator()( DataBox const & mem ) const
        {
            ShiftMeIfYouCan<
                simDim,
                0
            > const sx;
            ShiftMeIfYouCan<
                simDim,
                1
            > const sy;
            ShiftMeIfYouCan<
                simDim,
                2
            > const sz;

            // this-> is needed to access methods of dependent base class T_Curl
            return float3_X(
                this->yDerivative( sx( mem ) ).z( ) - this->zDerivative( sx( mem ) ).y( ),
                this->zDerivative( sy( mem ) ).x( ) - this->xDerivative( sy( mem ) ).z( ),
                this->xDerivative( sz( mem ) ).y( ) - this->yDerivative( sz( mem ) ).x( )
            );
        }
    };
} // namespace detail

    struct NoneDS
    {
        static constexpr uint32_t dim = simDim;

        typedef typename pmacc::math::CT::make_Int<dim, 0>::type LowerMargin;
        typedef typename pmacc::math::CT::make_Int<dim, 1>::type UpperMargin;

        template<
            typename T_DataBoxE,
            typename T_DataBoxB,
            typename T_DataBoxJ
        >
        HDINLINE void operator()(
            T_DataBoxE fieldE,
            T_DataBoxB fieldB,
            T_DataBoxJ const fieldJ
        )
        {
            using TypeJ = typename T_DataBoxJ::ValueType;
            using ComponentJ = typename GetComponentsType< TypeJ >::type;

            DataSpace< dim > const self;

            const ComponentJ deltaT = DELTA_T;
            ComponentJ const constE = ( float_X( 1.0 )  / EPS0 ) * deltaT;
            ComponentJ const constB = ( float_X( 0.25 ) / EPS0 ) * deltaT * deltaT;

            detail::LinearInterpolateComponentPlaneUpper<
                dim,
                0
            > const avgX;
            ComponentJ const jXavg = avgX( fieldJ );
            detail::LinearInterpolateComponentPlaneUpper<
                dim,
                1
            > const avgY;
            ComponentJ const jYavg = avgY( fieldJ );
            detail::LinearInterpolateComponentPlaneUpper<
                dim,
                2
            > const avgZ;
            ComponentJ const jZavg = avgZ( fieldJ );

            TypeJ const jAvgE = TypeJ(
                jXavg,
                jYavg,
                jZavg
            );
            fieldE( self ) -= jAvgE * constE;

            using CurlRight = fields::maxwellSolver::yee::CurlRight;
            using ShiftCurlRight = detail::ShiftCurl< CurlRight >;
            CurlRight curl;
            ShiftCurlRight shiftCurl;

            TypeJ const jAvgB = curl( fieldJ ) + shiftCurl( fieldJ );
            fieldB(self) += jAvgB * constB;
        }

        static pmacc::traits::StringProperty getStringProperties()
        {
            pmacc::traits::StringProperty propList(
                "name",
                "none"
            );
            return propList;
        }
    };

} // namespace currentInterpolation

namespace traits
{

    /* Get margin of the current interpolation
     *
     * This class defines a LowerMargin and an UpperMargin.
     */
    template< >
    struct GetMargin< picongpu::currentInterpolation::NoneDS >
    {
    private:
        using MyInterpolation = picongpu::currentInterpolation::NoneDS;

    public:
        using LowerMargin = typename MyInterpolation::LowerMargin;
        using UpperMargin = typename MyInterpolation::UpperMargin;
    };

} // namespace traits
} // namespace picongpu
