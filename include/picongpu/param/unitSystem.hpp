#pragma once

#include <boost/program_options.hpp>
#include <pmacc/ppFunctions.hpp>
#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <iostream>

#define PRINT_DO(v) #v
#define PRINT(v) PRINT_DO(v)

#define DEPAREN(X) ESC(ISH X)
#define ISH(...) ISH __VA_ARGS__
#define ESC(...) ESC_(__VA_ARGS__)
#define ESC_(...) VAN ## __VA_ARGS__
#define VANISH

#define UNIT_SYSTEN(name)           \
struct PMACC_JOIN(name,_t) {};      \
constexpr PMACC_JOIN(name,_t) name

namespace picongpu
{

    namespace units
    {
        UNIT_SYSTEN(SI);
        UNIT_SYSTEN(PIC);
        UNIT_SYSTEN(RENE);
    };
    namespace detail
    {
        template<typename T_Param, typename T_UnitTo>
        struct Param;

        template<typename T_Param, typename T_FromUnit, typename T_UnitTo>
        struct Unit;

    } // namespace detail




    struct IGlobalVariable
    {

        virtual void help(boost::program_options::options_description& desc) = 0;

    };

    struct IGlobalUnit
    {

        virtual void convert() = 0;

        template<typename T_Type>
        void uploadToDevice(T_Type *ptr, T_Type const value)
        {
            cudaMemcpy(ptr, &value, sizeof(T_Type), cudaMemcpyHostToDevice);
            std::cout << "copy to device" << std::endl;
        }

    };

    struct IGlobalVariableConversion
    {
        virtual void doconvert() = 0;

        template<typename T_Type>
        void uploadToDevice(T_Type *ptr, T_Type const value)
        {
            cudaMemcpy(ptr, &value, sizeof(T_Type), cudaMemcpyHostToDevice);
            std::cout << "copy to device" << std::endl;
        }
    };

    struct IGlobalSetDefault
    {
        virtual void setDefault() = 0;
    };


    struct GlobalRegister
    {
        static GlobalRegister &inst()
        {
            static GlobalRegister instance;
            return instance;
        }

        void announce(IGlobalVariable *handle)
        {
            data.push_back(handle);
        }

        void announceUnitConversion(IGlobalUnit *handle)
        {
            units.push_back(handle);
        }

        void announceVariableConversion(IGlobalVariableConversion *handle)
        {
            vars.push_back(handle);
        }

        void announceDefault(IGlobalSetDefault *handle)
        {
            defs.push_back(handle);
        }

        void loadHelp(boost::program_options::options_description& desc)
        {
            for (auto &d : defs)
            {
                d->setDefault();
            }

            for (auto &d : data)
            {
                d->help(desc);
            }


        }

        void update()
        {
            for (auto &d : units)
            {
                d->convert();
            }

            for (auto &d : vars)
            {
                d->doconvert();
            }
        }

        std::list<IGlobalVariable *> data;
        std::list<IGlobalUnit *> units;
        std::list<IGlobalVariableConversion *> vars;
        std::list<IGlobalSetDefault *> defs;
    };

} // namespace picongpu

template<typename T>
void setGlobalVarValue(T& ptr, const T& value)
{
    ptr = value;
};


#if( CUPLA_DEVICE_COMPILE == 1) //we are on gpu
#   define PICONGPU_VAR_NAMESPACE picongpu_var_device
#else
#   define PICONGPU_VAR_NAMESPACE picongpu_var_host
#endif

#define PICONGPU_VAR_DECL(type, name)                          \
    namespace picongpu_var_device               \
    {                                                          \
        __constant__ DEPAREN(type) name;                       \
    } /*namespace pmacc_static_const_vector_device + id */     \
    namespace picongpu_var_host                 \
    {                                                          \
       DEPAREN(type) name;                                     \
    } /*namespace pmacc_static_const_vector_device + id */     \

#define REG_VAR_DEFAULT(name, ...)  \
namespace PMACC_JOIN(simulation_,__COUNTER__)                                                                                  \
{                                                                                                                       \
    struct GloablObjectDefault : IGlobalSetDefault                                                                      \
    {                                                                                                                   \
        GloablObjectDefault()                                                                                           \
        {                                                                                                               \
            GlobalRegister::inst().announceDefault(this);                                                               \
        }                                                                                                               \
        void setDefault() override                                                                                      \
        {                                                                                                               \
            std::cout<<"set default"<< PRINT(name) << std::endl;                                                        \
            auto f = [&](){ return __VA_ARGS__;};                                                                        \
            setGlobalVarValue(PICONGPU_VAR_NAMESPACE::PMACC_JOIN(name, _SI) ,f());                                                        \
        }                                                                                                               \
    };                                                                                                                  \
    static GloablObjectDefault dummyDefault;                                                                            \
}

#define REG_CMD_VAR(name, cmdName, cmdDescription)                                                                      \
                                                                                                                        \
namespace PMACC_JOIN(simulation_,__COUNTER__)                                                                                  \
{                                                                                                                       \
    struct GloablCmdVar : IGlobalVariable                                                                               \
    {                                                                                                                   \
        GloablCmdVar()                                                                                                  \
        {                                                                                                               \
            GlobalRegister::inst().announce(this);                                                                      \
        }                                                                                                               \
        void help(boost::program_options::options_description& desc) override                                           \
        {                                                                                                               \
            using destinationType = decltype(PICONGPU_VAR_NAMESPACE::PMACC_JOIN(name, _SI));                                                                                                            \
            desc.add_options()                                                                                          \
                (cmdName, po::value<destinationType>(&PICONGPU_VAR_NAMESPACE::PMACC_JOIN(name, _SI))->default_value(PICONGPU_VAR_NAMESPACE::PMACC_JOIN(name, _SI)), cmdDescription);                          \
            std::cout<<cmdName <<" : " <<cmdDescription <<" : default = "<< PICONGPU_VAR_NAMESPACE::PMACC_JOIN(name, _SI)<<std::endl; \
        }                                                                                                               \
    };                                                                                                                  \
    static GloablCmdVar dummy;                                                                                          \
}

#define UNIT_VAR_NAME(name, fromUnit, toUnit) PMACC_JOIN(name, PMACC_JOIN(_, PMACC_JOIN(fromUnit, PMACC_JOIN(_, toUnit))))


#define REG_UNIT_CONVERSION(name, fromUnit, toUnit, ...)                                                \
                                                                                                        \
namespace PMACC_JOIN(simulation_,__COUNTER__)                                  \
{                                                                                                       \
    struct GloablObject : IGlobalUnit                                                                   \
    {                                                                                                   \
        GloablObject()                                                                                  \
        {                                                                                               \
            GlobalRegister::inst().announceUnitConversion(this);                                        \
        }                                                                                               \
        void convert() override                                                                         \
        {                                                                                               \
            std::cout<<"fill unit "<< PRINT(UNIT_VAR_NAME(name, fromUnit,toUnit)) << std::endl;         \
            using destinationType = decltype(PICONGPU_VAR_NAMESPACE::UNIT_VAR_NAME(name, fromUnit, toUnit));                 \
            auto f = __VA_ARGS__;                                                                       \
            setGlobalVarValue(PICONGPU_VAR_NAMESPACE::UNIT_VAR_NAME(name, fromUnit, toUnit), static_cast<destinationType>(f()));                        \
            destinationType* ptr = nullptr;               \
            cudaGetSymbolAddress((void **)&ptr, picongpu_var_device::UNIT_VAR_NAME(name, fromUnit, toUnit));        \
            uploadToDevice(ptr, PICONGPU_VAR_NAMESPACE::UNIT_VAR_NAME(name, fromUnit, toUnit)); \
        }                                                                                               \
    };                                                                                                  \
    static GloablObject dummy;                                                                          \
}

#define VAR_NAME(name, unitSystem) PMACC_JOIN(name, PMACC_JOIN(_, unitSystem))

#define REG_PARAM_CONVERSION(name, unitSystem, ...)                                                     \
                                                                                                        \
namespace PMACC_JOIN(simulation_,__COUNTER__)                                           \
{                                                                                                       \
    struct GloablObjectConversion : IGlobalVariableConversion                                                     \
    {                                                                                                   \
        GloablObjectConversion()                                                                                  \
        {                                                                                               \
            GlobalRegister::inst().announceVariableConversion(this);                                    \
        }                                                                                               \
        void doconvert() override                                                                       \
        {                                                                                               \
            std::cout<<"fill "<< PRINT(PMACC_JOIN(name, PMACC_JOIN(_, unitSystem))) << std::endl;       \
            using destinationType = decltype(PICONGPU_VAR_NAMESPACE::VAR_NAME(name, unitSystem));       \
            auto f = __VA_ARGS__;                                                                       \
            setGlobalVarValue(PICONGPU_VAR_NAMESPACE::VAR_NAME(name, unitSystem), static_cast<destinationType>(f()));                                   \
            destinationType* ptr = nullptr;                          \
            cudaGetSymbolAddress((void **)&ptr, picongpu_var_device::VAR_NAME(name, unitSystem));   \
            uploadToDevice(ptr, PICONGPU_VAR_NAMESPACE::VAR_NAME(name, unitSystem));                    \
        }                                                                                               \
    };                                                                                                  \
    static GloablObjectConversion dummy;                                                                          \
}


template<typename T_Param, typename T_Unit = picongpu::units::SI_t>
HDINLINE auto getParam(T_Param const param, T_Unit unit = T_Unit{})
{
    return picongpu::detail::Param<T_Param, T_Unit>::get();
}


template<typename T_Param, typename T_FromUnit, typename T_ToUnit>
HDINLINE auto scale(T_Param const param, T_FromUnit const, T_ToUnit const)
{
    return picongpu::detail::Unit<T_Param, T_FromUnit, T_ToUnit>::get();
}


#define DEF_UNIT(definition, nameCategory, nameSingular, nameAbbreviation)  \
struct PMACC_JOIN(definition, _t)                                           \
{                                                                           \
    static std::string category()                                           \
    {                                                                       \
        return nameCategory;                                                \
    }                                                                       \
    static std::string name()                                               \
    {                                                                       \
        return nameSingular;                                                \
    }                                                                       \
    static std::string abbreviation()                                       \
    {                                                                       \
        return nameAbbreviation;                                            \
    }                                                                       \
};                                                                          \
constexpr PMACC_JOIN(definition, _t) definition

#define PIC_DEF_UNIT(type, definition, nameCategory, nameSingular, nameAbbreviation, ...) \
    DEF_UNIT(definition, nameCategory, nameSingular, nameAbbreviation);                   \
    UNIT_ADD(type, definition, PIC);                                                      \
    UNIT_BASE(definition, PIC, __VA_ARGS__)

#define DEF_ALIAS(name)                                                     \
struct PMACC_JOIN(name, _t)                                                 \
{                                                                           \
    template<typename T_ToUnit = picongpu::units::PIC_t>                              \
    HDINLINE auto operator()( T_ToUnit const unit = T_ToUnit{} ) const      \
    {                                                                       \
        return getParam(PMACC_JOIN(name, _t){}, unit);                      \
    }                                                                       \
};                                                                          \
constexpr PMACC_JOIN(name, _t) name


#define CREATE_GLOBAL_VAR(type, name, unitSystem)                           \
PICONGPU_VAR_DECL(type,  PMACC_JOIN(name, PMACC_JOIN(_, unitSystem)));      \
static HDINLINE auto name(picongpu::units::PMACC_JOIN(unitSystem,_t) const) \
{                                                       \
    return PICONGPU_VAR_NAMESPACE::PMACC_JOIN(name, PMACC_JOIN(_, unitSystem));      \
}




#define INIT(name, unitSystem, ...)                                         \
    REG_PARAM_CONVERSION(name, unitSystem, __VA_ARGS__)

#define DEF_GLOBAL_VAR(type, name, ...) \
    CREATE_GLOBAL_VAR(type, name, SI);                                      \
    INIT(name, SI, __VA_ARGS__)

#define DEF_PARAMETER(type, name, cmd, help)                                \
    CREATE_GLOBAL_VAR(type, name, SI);                                      \
    REG_CMD_VAR(name, cmd, help)

#define PARAM_ADD(type, name, unitSystem)  CREATE_GLOBAL_VAR(type, name, unitSystem)
#define GLOBAL_VAR_ADD(type, name, unitSystem)  CREATE_GLOBAL_VAR(type, name, unitSystem)


#define UNIT_ADD_DO(type, name, fromSystem, unitSystem)                                                                 \
PICONGPU_VAR_DECL(type,UNIT_VAR_NAME(name, fromSystem, unitSystem));                                                    \
namespace detail{                                                                                                       \
    template<>                                                                                                          \
    struct Unit<PMACC_JOIN(name, _t), PMACC_JOIN(picongpu::units::fromSystem,_t),PMACC_JOIN(picongpu::units::unitSystem,_t)>                \
    {                                                                                                                   \
        static HDINLINE auto get()                                                                                      \
        {                                                                                                               \
            return PICONGPU_VAR_NAMESPACE::UNIT_VAR_NAME(name, fromSystem, unitSystem);                                                         \
        }                                                                                                               \
    };                                                                                                                  \
}

#define UNIT_ADD(type, name, unitSystem)                    \
    UNIT_ADD_DO(type, name, SI, unitSystem);                \
    UNIT_ADD_DO(type, name, unitSystem, SI)

#define PARAM_DEFAULT(name, ...)                            \
    REG_VAR_DEFAULT(name, __VA_ARGS__)



#define UNIT_SET_SCALING(name, fromSystem, unitSystem, ...) \
    REG_UNIT_CONVERSION(name, fromSystem, unitSystem, __VA_ARGS__)

#define UNIT_BASE(name, destinationUnitSystem, ...)         \
    UNIT_SET_SCALING(name, destinationUnitSystem, SI, __VA_ARGS__); \
    UNIT_SET_SCALING(name, SI, destinationUnitSystem, [&](){auto f = __VA_ARGS__; return 1.0/f();})



#define LEGACY_UNIT(name, unitName)                     \
template<typename T_UnitSystem>                         \
static HDINLINE auto name(T_UnitSystem const unitSystem) \
{                                                       \
    return scale(unitName, unitSystem, picongpu::units::SI);      \
}                                                       \
static HDINLINE auto name()                              \
{                                                       \
    return scale(unitName, picongpu::units::PIC, picongpu::units::SI);      \
}
