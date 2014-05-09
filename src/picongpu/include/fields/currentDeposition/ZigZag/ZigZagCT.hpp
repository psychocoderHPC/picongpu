/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera
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
#include "types.h"
#include "dimensions/DataSpace.hpp"
#include "basicOperations.hpp"
#include <cuSTL/cursor/tools/twistVectorFieldAxes.hpp>
#include "algorithms/FieldToParticleInterpolation.hpp"
#include "algorithms/ShiftCoordinateSystem.hpp"
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/copy.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/back_inserter.hpp>
#include "compileTime/AllCombinations.hpp"

namespace picongpu
{
namespace currentSolverZigZagCT
{
using namespace PMacc;

template<typename T_Value, int size = -1 >
struct CheckIt
{
    typedef typename T_Value::xxxxxxx type;
};

template<typename T_Shape, typename T_pos>
struct CallShape
{
    typedef typename T_Shape::ChargeAssignmentOnSupport ParticleAssign;

    HDINLINE float_X
    operator()(const float_X delta)
    {
        ParticleAssign shape;
        return shape(delta);
    }
};

template<>
struct CallShape<picongpu::particleShape::TSC, bmpl::integral_c<int,0> >
{
    typedef typename picongpu::particleShape::TSC::ChargeAssignmentOnSupport ParticleAssign;

    HDINLINE float_X
    operator()(const float_X delta)
    {
        // typedef CheckIt<ParticleAssign>::type type;
        return ParticleAssign::ff_1st_radius(algorithms::math::abs(delta));
    }
};

template<>
struct CallShape<picongpu::particleShape::TSC, bmpl::integral_c<int,1> >
{
    typedef typename picongpu::particleShape::TSC::ChargeAssignmentOnSupport ParticleAssign;

    HDINLINE float_X
    operator()(const float_X delta)
    {
        // typedef CheckIt<ParticleAssign>::type type;
        return ParticleAssign::ff_2nd_radius(delta);
    }
};

template<>
struct CallShape<picongpu::particleShape::TSC, bmpl::integral_c<int,-1> >
{
    typedef typename picongpu::particleShape::TSC::ChargeAssignmentOnSupport ParticleAssign;

    HDINLINE float_X
    operator()(const float_X delta)
    {
        //typedef CheckIt<ParticleAssign>::type type;
        return ParticleAssign::ff_2nd_radius(algorithms::math::abs(delta));
    }
};

template<>
struct CallShape<picongpu::particleShape::PCS, bmpl::integral_c<int,0> >
{
    typedef typename picongpu::particleShape::PCS::ChargeAssignmentOnSupport ParticleAssign;

    HDINLINE float_X
    operator()(const float_X delta)
    {
        // typedef CheckIt<ParticleAssign>::type type;
        return ParticleAssign::ff_1st_radius(algorithms::math::abs(delta));
    }
};

template<>
struct CallShape<picongpu::particleShape::PCS, bmpl::integral_c<int,1> >
{
    typedef typename picongpu::particleShape::PCS::ChargeAssignmentOnSupport ParticleAssign;

    HDINLINE float_X
    operator()(const float_X delta)
    {
        // typedef CheckIt<ParticleAssign>::type type;
        return ParticleAssign::ff_1st_radius(delta);
    }
};

template<>
struct CallShape<picongpu::particleShape::PCS, bmpl::integral_c<int,-1> >
{
    typedef typename picongpu::particleShape::PCS::ChargeAssignmentOnSupport ParticleAssign;

    HDINLINE float_X
    operator()(const float_X delta)
    {
        //typedef CheckIt<ParticleAssign>::type type;
        return ParticleAssign::ff_2nd_radius(algorithms::math::abs(delta));
    }
};

template<>
struct CallShape<picongpu::particleShape::PCS, bmpl::integral_c<int, -2> >
{
    typedef typename picongpu::particleShape::PCS::ChargeAssignmentOnSupport ParticleAssign;

    HDINLINE float_X
    operator()(const float_X delta)
    {
        //typedef CheckIt<ParticleAssign>::type type;
        return ParticleAssign::ff_2nd_radius(delta);
    }
};


template<typename T_MathVec, typename T_Shape, typename T_Vec>
struct ShapeIt_all
{

    template<typename T_Cursor>
    HDINLINE void
    operator()(const RefWrapper<T_Cursor> cursor, const float_X& F, const float3_X& pos  /*, const int xx*/)
    {
        typedef T_MathVec MathVec;

         const int x = MathVec::x::value;
         const int y = MathVec::y::value;
         const int z = MathVec::z::value;
       //  typedef CheckIt<MathVec,999999>::type type;
        //typedef typename CheckIt<T_Cursor, MathVec::x::value>::type Check;
      //  if(xx!=1)
        {
            CallShape<typename T_Shape::CloudShape, typename MathVec::x> shapeX;
            CallShape< T_Shape, typename MathVec::y> shapeY;
            CallShape< T_Shape, typename MathVec::z> shapeZ;

            const float_X shape_x = shapeX(float_X(x) - pos.x());
            const float_X shape_y = shapeY(float_X(y) - pos.y());
            const float_X shape_z = shapeZ(float_X(z) - pos.z());

            DataSpace<DIM3> jIdx;
            jIdx[T_Vec::x::value] = x;
            jIdx[T_Vec::y::value] = y;
            jIdx[T_Vec::z::value] = z;
            const float_X j = F * shape_x * shape_y*shape_z;
            atomicAddWrapper(&(cursor.get()(jIdx)[T_Vec::x::value]), j);
          //  printf("j=%f\n",j);
          //  printf("dim %i: %i %i %i shape=%f %f %f\n",T_Vec::x::value,jIdx.x(),jIdx.y(),jIdx.z(),shape_x,shape_y,shape_z);

           // abc.get()++;
        }

    }
};

/**
 * \class ZigZag charge conservation method
 * 1. order paper: "A new charge conservation method in electromagnetic particle-in-cell simulations"
 *                 by T. Umeda, Y. Omura, T. Tominaga, H. Matsumoto
 * 2. order paper: "Charge conservation methods for computing current densities in electromagnetic particle-in-cell simulations"
 *                 by T. Umeda, Y. Omura, H. Matsumoto
 * 3. order paper: "High-Order Interpolation Algorithms for Charge Conservation in Particle-in-Cell Simulation"
 *                 by Jinqing Yu, Xiaolin Jin, Weimin Zhou, Bin Li, Yuqiu Gu
 */
template<uint32_t T_Dim, typename T_ParticleShape>
struct ZigZagCT
{
    typedef T_ParticleShape ParticleShape;
    typedef typename ParticleShape::ChargeAssignmentOnSupport ParticleAssign;
    static const int supp = ParticleAssign::support;

    static const int currentLowerMargin = supp / 2 + 1;
    static const int currentUpperMargin = (supp + 1) / 2 + 1;
    typedef PMacc::math::CT::Int<currentLowerMargin, currentLowerMargin, currentLowerMargin> LowerMargin;
    typedef PMacc::math::CT::Int<currentUpperMargin, currentUpperMargin, currentUpperMargin> UpperMargin;

    static const int begin = -supp / 2 + (supp + 1) % 2;
    static const int end = begin + supp;


    static const int supp_dir = supp - 1;
    static const int dir_begin = -supp_dir / 2 + (supp_dir + 1) % 2;
    static const int dir_end = dir_begin + supp_dir;

    /* begin and end border is calculated for a particle with a support which travels
     * to the negative direction.
     * Later on all coordinates shifted thus we can solve the charge calculation
     * independend from the position of the particle. That means we must not look
     * if a particle position is >0.5 oder not (this is done by coordinate shifting to this defined range)
     *
     * (supp + 1) % 2 is 1 for even supports else 0
     */

    float_X charge;

    template<typename DataBoxJ, typename PosType, typename VelType, typename ChargeType >
    DINLINE void operator()(DataBoxJ dataBoxJ,
                            const PosType pos1,
                            const VelType velocity,
                            const ChargeType charge, const float_X deltaTime)
    {

        const float3_X deltaPos = float3_X(velocity.x() * deltaTime / cellSize.x(),
                                           velocity.y() * deltaTime / cellSize.y(),
                                           velocity.z() * deltaTime / cellSize.z());

        float3_X pos[2];
        pos[0] = (pos1 - deltaPos);
        pos[1] = (pos1);
   /*     printf("pos0: %f %f %f\n",pos1.x()-deltaPos.x(),pos1.y()-deltaPos.y(),pos1.z()-deltaPos.z());
        printf("pos1: %f %f %f\n",pos1.x(),pos1.y(),pos1.z());
        printf("cellsize: %f %f %f\n",cellSize.x(),cellSize.y(),cellSize.z());
        printf("delteT: %f \n",deltaTime);
        printf("vel: %f %f %f\n",velocity.x(),velocity.y(),velocity.z());
*/
        DataSpace<DIM3> I[2];
        float3_X r;


        for (int l = 0; l < 2; ++l)
        {
            for (uint32_t d = 0; d < DIM3; ++d)
            {
                I[l][d] = math::floor(pos[l][d]);
     //           printf("pos[%i][%i]: %f\n",l,d,pos[l][d]);
      //          printf("I[%i][%i]: %f\n",l,d,I[l][d]);
            }
        }
        for (uint32_t d = 0; d < DIM3; ++d)
        {
            r[d] = calc_r(I[0][d], I[1][d], pos[0][d], pos[1][d]);
      //      printf("r[%i]: %f\n",d,r[d]);
        }
        const float_X volume_reci = float_X(1.0) / float_X(CELL_VOLUME);

        using namespace cursor::tools;
        // floatD_X pos_tmp(pos1);

      //  BOOST_AUTO(cursorJ, dataBoxJ.toCursor());

       // float_X sign[]={1.0f,-1.0f};
//#pragma unroll 1
        for (float l = 0; l < 2; ++l)
        {
            float3_X IcP;
            float3_X F;
            const int parId=l;

            /* sign= 1 if l=0
             * sign=-1 if l=1
             */
            float_X sign = float_X(1.) - float_X(2.) * l;

            for (uint32_t d = 0; d < DIM3; ++d)
            {
                IcP[d] = calc_InCellPos(pos[parId][d], r[d], I[parId][d]);
                const float_X pos_tmp=pos[parId][d];
                const float_X r_tmp=r[d];
            //    if(l==1)
              //  {
                    F[d]=sign*calc_F(pos_tmp, r_tmp, deltaTime, charge) * volume_reci*cellSize[d];
             //   }
             //  else
             //       F[d]=calc_F(pos_tmp,r_tmp, deltaTime, charge) * volume_reci*cellSize[d];
            }
           // printf("pos_C[%i]: %f %f %f \n",l,pos[l].x(),pos[l].y(),pos[l].z());
          //  printf("r_c[%i]: %f %f %f \n",l,r.x(),r.y(),r.z());
        //    printf("IcP[%i]: %f %f %f \n",l,IcP.x(),IcP.y(),IcP.z());
            //printf("F=%f %f %f\n",F.x(),F.y(),F.z);

            BOOST_AUTO(cursorJ, dataBoxJ.shift(precisionCast<int>(I[parId])));

       //     printf("--x\n");
            DataBoxJ cursorJ_x(cursorJ);
            float3_X pos_x(IcP);
            ShiftCoordinateSystemOne<supp_dir, 0>()(cursorJ_x, pos_x, fieldSolver::NumericalCellType::getEFieldPosition().x());
            ShiftCoordinateSystemOne<supp, 1>()(cursorJ_x, pos_x, fieldSolver::NumericalCellType::getEFieldPosition().x());
            ShiftCoordinateSystemOne<supp, 2>()(cursorJ_x, pos_x, fieldSolver::NumericalCellType::getEFieldPosition().x());

            helper<PMacc::math::CT::Int < 0, 1, 2 > >(cursorJ_x, float3_X(pos_x[0], pos_x[1], pos_x[2]), F[0]);
       //     printf("--y\n");
            float3_X pos_y = IcP;
            DataBoxJ cursorJ_y(cursorJ);
            ShiftCoordinateSystemOne<supp, 0>()(cursorJ_y, pos_y, fieldSolver::NumericalCellType::getEFieldPosition().y());
            ShiftCoordinateSystemOne<supp_dir, 1>()(cursorJ_y, pos_y, fieldSolver::NumericalCellType::getEFieldPosition().y());
            ShiftCoordinateSystemOne<supp, 2>()(cursorJ_y, pos_y, fieldSolver::NumericalCellType::getEFieldPosition().y());
            helper<PMacc::math::CT::Int < 1, 2, 0 > >(cursorJ_y, float3_X(pos_y[1], pos_y[2], pos_y[0]), F[1]);
       //     printf("--z\n");
            float3_X pos_z = IcP;
            DataBoxJ cursorJ_z(cursorJ);
            ShiftCoordinateSystemOne<supp, 0>()(cursorJ_z, pos_z, fieldSolver::NumericalCellType::getEFieldPosition().z());
            ShiftCoordinateSystemOne<supp, 1>()(cursorJ_z, pos_z, fieldSolver::NumericalCellType::getEFieldPosition().z());
            ShiftCoordinateSystemOne<supp_dir, 2>()(cursorJ_z, pos_z, fieldSolver::NumericalCellType::getEFieldPosition().z());
            helper<PMacc::math::CT::Int < 2, 0, 1 > >(cursorJ_z, float3_X(pos_z[2], pos_z[0], pos_z[1]), F[2]);
        }
    }

    template<typename T_Vec, typename JCurser>
    DINLINE void helper(JCurser& dataBoxJ,
                        const float3_X& pos,
                        const float_X& F)
    {
      //  printf("calc pos[%i]: %f %f %f F=%f\n",T_Vec::x::value,pos.x(),pos.y(),pos.z(),F);

        //   typedef typename ParticleShape::CloudShape::ChargeAssignmentOnSupport CloudShapeAssign;
        //    PMACC_AUTO(shape, ParticleAssign());
        //    PMACC_AUTO(cloudShapeAssign, CloudShapeAssign());

        typedef boost::mpl::vector3<
            boost::mpl::range_c<int, dir_begin, dir_end >,
            boost::mpl::range_c<int, begin, end >,
            boost::mpl::range_c<int, begin, end > > Size;
        typedef typename AllCombinations<Size>::type CombiTypes;



        //  typedef typename CheckIt<CombiTypes,boost::mpl::size<CombiTypes>::value >::type Check;

        //  const float_X F = cellLength * currentDensity;


        /*   for (int x = dir_begin; x < dir_end; ++x)
           {
               const float_X F = cellLength * currentDensity * cloudShapeAssign(float_X(x) - pos.x());


                   //ForEach<boost::mpl::range_c<int, begin, end >, ShapeIt_y<void, ParticleShape,boost::mpl::range_c<int, begin, end > > > shapeIt;
                   //shapeIt(dataBoxJ,x,F, pos);
               ForEach<CombiTypes, ShapeIt_all<void,ParticleShape> > shapeIt;
               shapeIt(dataBoxJ,x,F, pos);
           }
         */
          //  for (int x = dir_begin; x < dir_end; ++x){




        ForEach<CombiTypes, ShapeIt_all<bmpl::_1, ParticleShape, T_Vec> > shapeIt;
        shapeIt(byRef(dataBoxJ), F, pos);
        //    }
      /*  if(abc!=18)
            printf("%i\n",abc);
*/
        //    }
        /*
                for (int x = dir_begin+1; x < dir_end; ++x)
                {
                float_X F = cellLength * currentDensity * cloudShapeAssign(float_X(x) - pos.x());
                ForEach<CombiTypes, ShapeIt_all<void,ParticleShape> > shapeIt;
                    shapeIt(dataBoxJ,x,F, pos);

                }
         */

    }

private:

    DINLINE float_X
    calc_r(const float_X i_1,const float_X i_2,const float_X x_1, const float_X x_2) const
    {

        const float_X min_1 = ::min(i_1, i_2) + float_X(1.0);
        const float_X max_1 = ::max(i_1, i_2);
        const float_X max_2 = ::max(max_1, (x_1 + x_2) / float_X(2.));
        const float_X x_r = ::min(min_1, max_2);
        return x_r;
    }

    DINLINE float_X
    calc_InCellPos(const float_X x, const float_X x_r,const float_X i) const
    {
        return (x + x_r) / (float_X(2.0)) - i;
    }

    /* for F_2 call with -x and -x_r*/
    DINLINE float_X
    calc_F(const float_X x, const float_X x_r, const float_X& delta_t, const float_X& q) const
    {
        return q * (x_r - x) / delta_t;
    }
};

} //namespace currentSolverZigZag

namespace traits
{

/*Get margin of a solver
 * class must define a LowerMargin and UpperMargin
 */
template<uint32_t T_Dim, typename T_ParticleShape>
struct GetMargin<picongpu::currentSolverZigZagCT::ZigZagCT<T_Dim, T_ParticleShape> >
{
private:
    typedef picongpu::currentSolverZigZagCT::ZigZagCT<T_Dim, T_ParticleShape> Solver;
public:
    typedef typename Solver::LowerMargin LowerMargin;
    typedef typename Solver::UpperMargin UpperMargin;
};

} //namespace traits

} //namespace picongpu


