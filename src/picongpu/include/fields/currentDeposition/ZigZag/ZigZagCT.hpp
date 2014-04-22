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
#include "dimensions/TVec.h"
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

template<typename T_Value,int size=-1>
struct CheckIt
{
    typedef typename T_Value::xxxxxxx type;
};

template<typename T_Shape,int T_pos>
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
struct CallShape<picongpu::particleShape::TSC,0>
{
    typedef typename picongpu::particleShape::TSC::ChargeAssignmentOnSupport ParticleAssign;
    HDINLINE float_X 
    operator()(const float_X delta)
    {
       // typedef CheckIt<ParticleAssign>::type type;
        return ParticleAssign::ff_1st_radius(delta);
    }
};

template<>
struct CallShape<picongpu::particleShape::TSC,1>
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
struct CallShape<picongpu::particleShape::TSC,-1>
{
    typedef typename picongpu::particleShape::TSC::ChargeAssignmentOnSupport ParticleAssign;
    HDINLINE float_X 
    operator()(const float_X delta)
    {
        //typedef CheckIt<ParticleAssign>::type type;
        return ParticleAssign::ff_2nd_radius(delta);
    }
};

template<>
struct CallShape<picongpu::particleShape::PCS,0>
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
struct CallShape<picongpu::particleShape::PCS,1>
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
struct CallShape<picongpu::particleShape::PCS,-1>
{
    typedef typename picongpu::particleShape::PCS::ChargeAssignmentOnSupport ParticleAssign;
    HDINLINE float_X 
    operator()(const float_X delta)
    {
        //typedef CheckIt<ParticleAssign>::type type;
        return ParticleAssign::ff_2nd_radius(delta);
    }
};

template<>
struct CallShape<picongpu::particleShape::PCS,2>
{
    typedef typename picongpu::particleShape::PCS::ChargeAssignmentOnSupport ParticleAssign;
    HDINLINE float_X 
    operator()(const float_X delta)
    {
        //typedef CheckIt<ParticleAssign>::type type;
        return ParticleAssign::ff_2nd_radius(delta);
    }
};

template<typename T_z, typename T_Shape>
struct ShapeIt_z
{
       
    
    template<typename T_Cursor>
    HDINLINE void 
    operator()(T_Cursor& cursor, const int x, const int y, const float_X F, const float3_X &pos)
    {
        CallShape<T_Shape, T_z::value> shape;
        
        const DataSpace<DIM3> jIdx(x, y, T_z::value);
        const float_X abs_pos = algorithms::math::abs(float_X(T_z::value) - pos.z());
        float_X j = F * shape(abs_pos);
        //if (j != float_X(0.0))
            atomicAddWrapper(&((*cursor(jIdx)).x()), j);
    }
};
template<typename T_y, typename T_Shape,typename range>
struct ShapeIt_y
{
       
    
    template<typename T_Cursor>
    HDINLINE void 
    operator()(T_Cursor& cursor, const int x, const float_X F, const float3_X &pos)
    {
        CallShape<T_Shape, T_y::value> shape;
        
        const float_X abs_pos = algorithms::math::abs(float_X(T_y::value) - pos.y());
        const float_X shape_y = shape(abs_pos);

        ForEach<range, ShapeIt_z<void, T_Shape> > shapeIt;
        shapeIt(cursor,x, float_X(T_y::value), F*shape_y, pos);
    }
};

template<typename T_MathVec, typename T_Shape>
struct ShapeIt_all
{
    typedef T_MathVec MathVec;
    
    static const int x=MathVec::x::value;
    static const int y=MathVec::y::value;
    static const int z=MathVec::z::value;
    
 //   typedef typename CheckIt<MathVec, MathVec::x::value>::type Check;
    
    template<typename T_Cursor>
    HDINLINE void 
    operator()(T_Cursor& cursor,const int xx,const float_X F,const float3_X &pos)
    {
        if(xx!=1)
        {
        CallShape<typename T_Shape::CloudShape,x> shapeX;
        CallShape< T_Shape, y> shapeY;
        CallShape< T_Shape, z> shapeZ;
        
        const float_X abs_x = algorithms::math::abs(float_X(x) - pos.x());
        const float_X abs_y = algorithms::math::abs(float_X(y) - pos.y());
        const float_X abs_z = algorithms::math::abs(float_X(z) - pos.z());
        
        const float_X shape_x = shapeX(abs_x);
        const float_X shape_y = shapeY(abs_y);
        const float_X shape_z = shapeZ(abs_z);

        const DataSpace<DIM3> jIdx(x, y, z);
        const float_X j = F * shape_x*shape_y*shape_z;
        atomicAddWrapper(&((*cursor(jIdx)).x()), j);
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

        DataSpace<DIM3> I[2];
        float3_X r;


        for (int l = 0; l < 2; ++l)
        {
            for (uint32_t d = 0; d < DIM3; ++d)
            {
                I[l][d] = math::floor(pos[l][d]);
            }
        }
        for (uint32_t d = 0; d < DIM3; ++d)
        {
            r[d] = calc_r(I[0][d], I[1][d], pos[0][d], pos[1][d]);

        }
        const float_X volume_reci = float_X(1.0) / float_X(CELL_VOLUME);

        using namespace cursor::tools;
        // floatD_X pos_tmp(pos1);

        BOOST_AUTO(cursorJ, dataBoxJ.toCursor());

        for (int l = 0; l < 2; ++l)
        {
            float3_X IcP;
            /* sign= 1 if l=0
             * sign=-1 if l=1
             */
            const float_X sign = float_X(1.0) - float_X(2.0) * float_X(l);
            for (uint32_t d = 0; d < DIM3; ++d)
            {
                IcP[d] = calc_InCellPos(pos[l][d], r[d], I[l][d]);
            }

            BOOST_AUTO(cursorJ, dataBoxJ.shift(precisionCast<int>(I[l])).toCursor());
            BOOST_AUTO(cursorJ_x, cursorJ);

            float3_X pos_tmp(IcP);
            const float_X currentDensity_x = calc_F(sign * pos[l][0], sign * r[0], deltaTime, charge) * volume_reci;
            ShiftCoordinateSystemOne<supp_dir, 0>()(cursorJ_x, pos_tmp, fieldSolver::NumericalCellType::getEFieldPosition().x());
            ShiftCoordinateSystemOne<supp, 1>()(cursorJ_x, pos_tmp, fieldSolver::NumericalCellType::getEFieldPosition().x());
            ShiftCoordinateSystemOne<supp, 2>()(cursorJ_x, pos_tmp, fieldSolver::NumericalCellType::getEFieldPosition().x());
            helper(cursorJ_x, pos_tmp, currentDensity_x, cellSize.x());

            pos_tmp = IcP;
            BOOST_AUTO(cursorJ_y, cursorJ);
            const float_X currentDensity_y = calc_F(sign * pos[l][1], sign * r[1], deltaTime, charge) * volume_reci;
            ShiftCoordinateSystemOne<supp, 0>()(cursorJ_y, pos_tmp, fieldSolver::NumericalCellType::getEFieldPosition().y());
            ShiftCoordinateSystemOne<supp_dir, 1>()(cursorJ_y, pos_tmp, fieldSolver::NumericalCellType::getEFieldPosition().y());
            ShiftCoordinateSystemOne<supp, 2>()(cursorJ_y, pos_tmp, fieldSolver::NumericalCellType::getEFieldPosition().y());
            helper(twistVectorFieldAxes<PMacc::math::CT::Int < 1, 0, 2 > >(cursorJ_y), float3_X(pos_tmp[1], pos_tmp[0], pos_tmp[2]), currentDensity_y, cellSize.y());

            pos_tmp = IcP;
            BOOST_AUTO(cursorJ_z, cursorJ);
            const float_X currentDensity_z = calc_F(sign * pos[l][2], sign * r[2], deltaTime, charge) * volume_reci;
            ShiftCoordinateSystemOne<supp, 0>()(cursorJ_z, pos_tmp, fieldSolver::NumericalCellType::getEFieldPosition().z());
            ShiftCoordinateSystemOne<supp, 1>()(cursorJ_z, pos_tmp, fieldSolver::NumericalCellType::getEFieldPosition().z());
            ShiftCoordinateSystemOne<supp_dir, 2>()(cursorJ_z, pos_tmp, fieldSolver::NumericalCellType::getEFieldPosition().z());
            helper(twistVectorFieldAxes<PMacc::math::CT::Int < 2, 0, 1 > >(cursorJ_z), float3_X(pos_tmp[2], pos_tmp[0], pos_tmp[1]), currentDensity_z, cellSize.z());
        }
    }

    template<typename JCurser>
    DINLINE void helper(JCurser dataBoxJ,
                        const float3_X& pos,
                        const float_X currentDensity,
                        const float_X cellLength)
    {


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
         for (int x = dir_begin; x < dir_end; ++x){
        
            
       
        float_X F = cellLength * currentDensity; // * cloudShapeAssign(float_X(x) - pos.x());
        ForEach<CombiTypes, ShapeIt_all<void,ParticleShape> > shapeIt;
            shapeIt(dataBoxJ,x,F, pos);        

        }
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
    calc_r(float_X i_1, float_X i_2, float_X x_1, float_X x_2)
    {

        const float_X min_1 = ::min(i_1, i_2) + float_X(1.0);
        const float_X max_1 = ::max(i_1, i_2);
        const float_X max_2 = ::max(max_1, (x_1 + x_2) / float_X(2.));
        const float_X x_r = ::min(min_1, max_2);
        return x_r;
    }

    DINLINE float_X
    calc_InCellPos(float_X x, float_X x_r, float_X i)
    {
        return (x + x_r) / (float_X(2.0)) - i;
    }

    /* for F_2 call with -x and -x_r*/
    DINLINE float_X
    calc_F(float_X x, float_X x_r, const float_X& delta_t, const float_X& q)
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


