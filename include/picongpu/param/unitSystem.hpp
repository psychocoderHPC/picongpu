#include <iostream>
#include <list>
#include <array>

#define PRINT_DO(v) #v
#define PRINT(v) PRINT_DO(v)

#define DEPAREN(X) ESC(ISH X)
#define ISH(...) ISH __VA_ARGS__
#define ESC(...) ESC_(__VA_ARGS__)
#define ESC_(...) VAN##__VA_ARGS__
#define VANISH

#if 0
#    define PMACC_JOIN_DO(x, y) x##y
#    define PMACC_JOIN(x, y) PMACC_JOIN_DO(x, y)

#    if !defined(__CUDACC__)
#        define HDINLINE __device__ __host__
#        define HINLINE __host__
#    else
#        define HDINLINE
#        define HINLINE
#        define __constant__
#        define __global__
#    endif


#    if defined(__CUDACC__)
#        define PMACC_NO_NVCC_HDWARNING _Pragma("hd_warning_disable")
#    else
#        define PMACC_NO_NVCC_HDWARNING
#    endif
#endif

#define PRINT_DO(v) #v
#define PRINT(v) PRINT_DO(v)

#define DEPAREN(X) ESC(ISH X)
#define ISH(...) ISH __VA_ARGS__
#define ESC(...) ESC_(__VA_ARGS__)
#define ESC_(...) VAN##__VA_ARGS__
#define VANISH

#define UNIT_SYSTEN(name)                                                                                             \
    struct PMACC_JOIN(name, _t)                                                                                       \
    {                                                                                                                 \
    };                                                                                                                \
    constexpr PMACC_JOIN(name, _t) name


#include "units.h"

#define PIC_UNIT_FRACTION_DEF(namespaceName, fraction, name)                                                          \
    constexpr auto PMACC_JOIN(fraction, name) = namespaceName::PMACC_JOIN(PMACC_JOIN(fraction, name), _t)(1.0)
#define PIC_UNIT_DEF(namespaceName, name)                                                                             \
    constexpr auto name = namespaceName::PMACC_JOIN(name, _t)(1.0);                                                   \
    PIC_UNIT_FRACTION_DEF(namespaceName, femto, name);                                                                \
    PIC_UNIT_FRACTION_DEF(namespaceName, pico, name);                                                                 \
    PIC_UNIT_FRACTION_DEF(namespaceName, nano, name);                                                                 \
    PIC_UNIT_FRACTION_DEF(namespaceName, micro, name);                                                                \
    PIC_UNIT_FRACTION_DEF(namespaceName, milli, name);                                                                \
    PIC_UNIT_FRACTION_DEF(namespaceName, centi, name);                                                                \
    PIC_UNIT_FRACTION_DEF(namespaceName, deci, name);                                                                 \
    PIC_UNIT_FRACTION_DEF(namespaceName, deca, name);                                                                 \
    PIC_UNIT_FRACTION_DEF(namespaceName, hecto, name);                                                                \
    PIC_UNIT_FRACTION_DEF(namespaceName, kilo, name);                                                                 \
    PIC_UNIT_FRACTION_DEF(namespaceName, mega, name);                                                                 \
    PIC_UNIT_FRACTION_DEF(namespaceName, giga, name);                                                                 \
    PIC_UNIT_FRACTION_DEF(namespaceName, tera, name);                                                                 \
    PIC_UNIT_FRACTION_DEF(namespaceName, peta, name)

namespace picongpu
{
    namespace units
    {
        PIC_UNIT_DEF(::units::length, meter);

        PIC_UNIT_DEF(::units::mass, gram);

        PIC_UNIT_DEF(::units::time, second);

        PIC_UNIT_DEF(::units::angle, radian);

        PIC_UNIT_DEF(::units::current, ampere);
        constexpr auto scalar = ::units::dimensionless::scalar_t(1.0);

        constexpr auto celsius = ::units::temperature::celsius_t(1.0);
        constexpr auto fahrenheit = ::units::temperature::fahrenheit_t(1.0);
        constexpr auto kelvin = ::units::temperature::kelvin_t(1.0);

        constexpr auto mole = ::units::substance::mole_t(1.0);

        PIC_UNIT_DEF(::units::luminous_intensity, candela);

        PIC_UNIT_DEF(::units::charge, coulomb);
        PIC_UNIT_DEF(::units::charge, ampere_hour);
        PIC_UNIT_DEF(::units::energy, joule);
        PIC_UNIT_DEF(::units::voltage, volt);

        PIC_UNIT_DEF(::units::solid_angle, steradian);

        PIC_UNIT_DEF(::units::frequency, hertz);

    } // namespace units
} // namespace picongpu

#include <type_traits>


namespace picongpu
{
    namespace base
    {
        UNIT_SYSTEN(SI);
        UNIT_SYSTEN(PIC);
    }; // namespace base
    namespace detail
    {
        template<typename T_Param, typename T_UnitTo>
        struct Param;

        template<typename T_Param, typename T_FromUnit, typename T_UnitTo>
        struct Category;

    } // namespace detail


    struct IGlobalVariableHelp
    {
        virtual void help(boost::program_options::options_description& desc) = 0;
    };

    struct IGlobalCategory
    {
        virtual void convert() = 0;

        template<typename T_Type>
        void uploadToDevice(T_Type* ptr, T_Type const value)
        {
            cudaMemcpy(ptr, &value, sizeof(T_Type), cudaMemcpyHostToDevice);
            std::cout << "copy to device" << std::endl;
        }
    };

    struct IGlobalVariableConversion
    {
        virtual void doconvert() = 0;

        template<typename T_Type>
        void uploadToDevice(T_Type* ptr, T_Type const value)
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
        static GlobalRegister& inst()
        {
            static GlobalRegister instance;
            return instance;
        }

        void announce(IGlobalVariableHelp* handle)
        {
            data.push_back(handle);
        }

        void announceUnitConversion(IGlobalCategory* handle)
        {
            units.push_back(handle);
        }

        void announceVariableConversion(IGlobalVariableConversion* handle)
        {
            vars.push_back(handle);
        }

        void announceDefault(IGlobalSetDefault* handle)
        {
            defs.push_back(handle);
        }

        void loadHelp(boost::program_options::options_description& desc)
        {
            for(auto& d : defs)
            {
                d->setDefault();
            }

            for(auto& d : data)
            {
                d->help(desc);
            }
        }

        void update()
        {
            for(auto& d : units)
            {
                d->convert();
            }

            for(auto& d : vars)
            {
                d->doconvert();
            }
        }

        std::list<IGlobalVariableHelp*> data;
        std::list<IGlobalCategory*> units;
        std::list<IGlobalVariableConversion*> vars;
        std::list<IGlobalSetDefault*> defs;
    };


    template<typename T>
    void setGlobalVarValue(T& ptr, const T& value)
    {
        ptr = value;
    };


#if(CUPLA_DEVICE_COMPILE == 1)
#    define PICONGPU_VAR_NAMESPACE picongpu_var_device
#else
#    define PICONGPU_VAR_NAMESPACE picongpu_var_host
#endif

#define PICONGPU_VAR_DECL(type, name)                                                                                 \
    namespace picongpu_var_device                                                                                     \
    {                                                                                                                 \
        __constant__ DEPAREN(type) name;                                                                              \
    } /*namespace pmacc_static_const_vector_device + id */                                                            \
    namespace picongpu_var_host                                                                                       \
    {                                                                                                                 \
        DEPAREN(type) name;                                                                                           \
    } /*namespace pmacc_static_const_vector_device + id */


    template<typename T_From, typename T_To>
    HDINLINE auto convert(const T_From from, const T_To to) ->
        typename std::enable_if<!std::is_fundamental<T_From>::value, T_To>::type
    {
        T_To tmp = from;
        return tmp;
    }

    template<typename T_From, typename T_To>
    HDINLINE auto convert(const T_From from, const T_To) ->
        typename std::enable_if<std::is_fundamental<T_From>::value, T_To>::type
    {
        return T_To(from);
    }

#define REG_VAR_DEFAULT(name, ...)                                                                                    \
    namespace PMACC_JOIN(simulation_, __COUNTER__)                                                                    \
    {                                                                                                                 \
        struct GloablObjectDefault : IGlobalSetDefault                                                                \
        {                                                                                                             \
            GloablObjectDefault()                                                                                     \
            {                                                                                                         \
                GlobalRegister::inst().announceDefault(this);                                                         \
            }                                                                                                         \
            void setDefault() override                                                                                \
            {                                                                                                         \
                using destinationType = decltype(PICONGPU_VAR_NAMESPACE::PMACC_JOIN(name, _SI));                      \
                auto f = [&]() { return (__VA_ARGS__); };                                                             \
                std::cout << "set default " << PRINT(name) << " " << convert(f(), name.unit()) << std::endl;          \
                setGlobalVarValue(                                                                                    \
                    PICONGPU_VAR_NAMESPACE::PMACC_JOIN(name, _SI),                                                    \
                    static_cast<destinationType>(convert(f(), name.unit()).value()));                                 \
            }                                                                                                         \
        };                                                                                                            \
        static GloablObjectDefault dummyDefault;                                                                      \
    }

#define REG_CMD_VAR(name, cmdName, cmdDescription)                                                                    \
                                                                                                                      \
    namespace PMACC_JOIN(simulation_, __COUNTER__)                                                                    \
    {                                                                                                                 \
        struct GloablCmdVar : IGlobalVariableHelp                                                                     \
        {                                                                                                             \
            GloablCmdVar()                                                                                            \
            {                                                                                                         \
                GlobalRegister::inst().announce(this);                                                                \
            }                                                                                                         \
            void help(boost::program_options::options_description& desc) override                                     \
            {                                                                                                         \
                using destinationType = decltype(PICONGPU_VAR_NAMESPACE::PMACC_JOIN(name, _SI));                      \
                desc.add_options()(                                                                                   \
                    cmdName,                                                                                          \
                    po::value<destinationType>(&PICONGPU_VAR_NAMESPACE::PMACC_JOIN(name, _SI))                        \
                        ->default_value(PICONGPU_VAR_NAMESPACE::PMACC_JOIN(name, _SI)),                               \
                    cmdDescription);                                                                                  \
                std::cout << cmdName << " : " << cmdDescription << " : default = " << name(base::SI) * name.unit()    \
                          << std::endl;                                                                               \
            }                                                                                                         \
        };                                                                                                            \
        static GloablCmdVar dummy;                                                                                    \
    }

#define UNIT_VAR_NAME(name, fromUnit, toUnit)                                                                         \
    PMACC_JOIN(name, PMACC_JOIN(_, PMACC_JOIN(fromUnit, PMACC_JOIN(_, toUnit))))


#define REG_UNIT_CONVERSION(name, fromUnit, toUnit, ...)                                                              \
                                                                                                                      \
    namespace PMACC_JOIN(simulation_, __COUNTER__)                                                                    \
    {                                                                                                                 \
        struct GloablObject : IGlobalCategory                                                                         \
        {                                                                                                             \
            GloablObject()                                                                                            \
            {                                                                                                         \
                GlobalRegister::inst().announceUnitConversion(this);                                                  \
            }                                                                                                         \
            void convert() override                                                                                   \
            {                                                                                                         \
                using destinationType = decltype(PICONGPU_VAR_NAMESPACE::UNIT_VAR_NAME(name, fromUnit, toUnit));      \
                auto f = __VA_ARGS__;                                                                                 \
                setGlobalVarValue(                                                                                    \
                    PICONGPU_VAR_NAMESPACE::UNIT_VAR_NAME(name, fromUnit, toUnit),                                    \
                    static_cast<destinationType>(f()));                                                               \
                destinationType* ptr = nullptr;                                                                       \
                cudaGetSymbolAddress((void**) &ptr, picongpu_var_device::UNIT_VAR_NAME(name, fromUnit, toUnit));      \
                uploadToDevice(ptr, PICONGPU_VAR_NAMESPACE::UNIT_VAR_NAME(name, fromUnit, toUnit));                   \
                std::cout << "fill unit " << PRINT(UNIT_VAR_NAME(name, fromUnit, toUnit)) << " "                      \
                          << PICONGPU_VAR_NAMESPACE::UNIT_VAR_NAME(name, fromUnit, toUnit) << std::endl;              \
            }                                                                                                         \
        };                                                                                                            \
        static GloablObject dummy;                                                                                    \
    }

#define VAR_NAME(name, unitSystem) PMACC_JOIN(name, PMACC_JOIN(_, unitSystem))

#define REG_PARAM_CONVERSION(name, unitSystem, ...)                                                                   \
                                                                                                                      \
    namespace PMACC_JOIN(simulation_, __COUNTER__)                                                                    \
    {                                                                                                                 \
        struct GloablObjectConversion : IGlobalVariableConversion                                                     \
        {                                                                                                             \
            GloablObjectConversion()                                                                                  \
            {                                                                                                         \
                GlobalRegister::inst().announceVariableConversion(this);                                              \
            }                                                                                                         \
            void doconvert() override                                                                                 \
            {                                                                                                         \
                std::cout << "fill " << PRINT(PMACC_JOIN(name, PMACC_JOIN(_, unitSystem))) << std::endl;              \
                using destinationType = decltype(PICONGPU_VAR_NAMESPACE::VAR_NAME(name, unitSystem));                 \
                auto f = __VA_ARGS__;                                                                                 \
                setGlobalVarValue(                                                                                    \
                    PICONGPU_VAR_NAMESPACE::VAR_NAME(name, unitSystem),                                               \
                    static_cast<destinationType>(f(name(base::unitSystem))));                                         \
                destinationType* ptr = nullptr;                                                                       \
                cudaGetSymbolAddress((void**) &ptr, picongpu_var_device::VAR_NAME(name, unitSystem));                 \
                uploadToDevice(ptr, PICONGPU_VAR_NAMESPACE::VAR_NAME(name, unitSystem));                              \
            }                                                                                                         \
        };                                                                                                            \
        static GloablObjectConversion dummy;                                                                          \
    }


#define DEF_CATEGORY_ALIAS(definition, baseUnit)                                                                      \
    struct PMACC_JOIN(definition, _t)                                                                                 \
    {                                                                                                                 \
        static auto unit()                                                                                            \
        {                                                                                                             \
            return decltype(baseUnit)(1.);                                                                            \
        }                                                                                                             \
        template<typename U1>                                                                                         \
        HDINLINE auto operator()(const U1 u1) const                                                                   \
        {                                                                                                             \
            return picongpu::detail::Category<PMACC_JOIN(definition, _t), U1, base::SI_t>::get();                     \
        }                                                                                                             \
        template<typename U1, typename U2>                                                                            \
        HDINLINE auto operator()(const U1 u1, const U2 u2) const                                                      \
        {                                                                                                             \
            return picongpu::detail::Category<PMACC_JOIN(definition, _t), U1, U2>::get();                             \
        }                                                                                                             \
    };                                                                                                                \
    constexpr PMACC_JOIN(definition, _t) definition


#define ADD_CATEGORY_DO(type, name, fromSystem, unitSystem)                                                           \
    PICONGPU_VAR_DECL(type, UNIT_VAR_NAME(name, fromSystem, unitSystem));                                             \
    namespace detail                                                                                                  \
    {                                                                                                                 \
        template<>                                                                                                    \
        struct Category<                                                                                              \
            PMACC_JOIN(name, _t),                                                                                     \
            PMACC_JOIN(picongpu::base::fromSystem, _t),                                                               \
            PMACC_JOIN(picongpu::base::unitSystem, _t)>                                                               \
        {                                                                                                             \
            static HDINLINE auto get()                                                                                \
            {                                                                                                         \
                return PICONGPU_VAR_NAMESPACE::UNIT_VAR_NAME(name, fromSystem, unitSystem);                           \
            }                                                                                                         \
        };                                                                                                            \
    }

#define ADD_CATEGORY(type, name, unitSystem)                                                                          \
    ADD_CATEGORY_DO(type, name, SI, unitSystem);                                                                      \
    ADD_CATEGORY_DO(type, name, unitSystem, SI)


#define PIC_DEF_CATEGORY(type, definition, baseUnit)                                                                  \
    DEF_CATEGORY_ALIAS(definition, baseUnit);                                                                         \
    ADD_CATEGORY(type, definition, PIC);


#define DEF_ALIAS(name, baseUnit)                                                                                     \
    struct PMACC_JOIN(name, _t)                                                                                       \
    {                                                                                                                 \
        PMACC_NO_NVCC_HDWARNING                                                                                       \
        template<typename T_ToUnit>                                                                                   \
        HDINLINE auto operator()(T_ToUnit const unitSystem) const                                                     \
        {                                                                                                             \
            return PMACC_JOIN(get_, name)(unitSystem);                                                                \
        }                                                                                                             \
        HDINLINE static auto unit()                                                                                   \
        {                                                                                                             \
            return baseUnit;                                                                                          \
        }                                                                                                             \
        template<typename U1, typename U2>                                                                            \
        HDINLINE static auto sfactor(const U1 u1, const U2 u2)                                                        \
        {                                                                                                             \
            return scaling(unit(), u1, u2).value();                                                                   \
        }                                                                                                             \
    };                                                                                                                \
    constexpr PMACC_JOIN(name, _t) name


#define CREATE_GLOBAL_VAR(type, name, unitSystem)                                                                     \
    PICONGPU_VAR_DECL(type, PMACC_JOIN(name, PMACC_JOIN(_, unitSystem)));                                             \
                                                                                                                      \
    HDINLINE auto PMACC_JOIN(get_, name)(picongpu::base::PMACC_JOIN(unitSystem, _t) const)                            \
    {                                                                                                                 \
        return PICONGPU_VAR_NAMESPACE::PMACC_JOIN(name, PMACC_JOIN(_, unitSystem));                                   \
    }


#define INIT(name, unitSystem)                                                                                        \
    REG_PARAM_CONVERSION(name, unitSystem, [&](auto x) {                                                              \
        return name(picongpu::base::SI) * scaling(PMACC_JOIN(name, _t)::unit(), base::SI, base::unitSystem);          \
    })


#define GLOBAL_VAR_CONVERSION(name, unitSystem, ...) REG_PARAM_CONVERSION(name, unitSystem, __VA_ARGS__)


#define DEF_GLOBAL_VAR(type, name, unit, ...)                                                                         \
    CREATE_GLOBAL_VAR(type, name, SI);                                                                                \
    DEF_ALIAS(name, unit);                                                                                            \
    REG_PARAM_CONVERSION(name, SI, __VA_ARGS__)


#define DEF_PARAMETER(type, name, cmd, help, unit)                                                                    \
    CREATE_GLOBAL_VAR(type, name, SI);                                                                                \
    DEF_ALIAS(name, unit);                                                                                            \
    REG_CMD_VAR(name, cmd, help)

#define PIC_DEF_PARAMETER(type, name, cmd, help, unit)                                                                \
    CREATE_GLOBAL_VAR(type, name, SI);                                                                                \
    DEF_ALIAS(name, unit);                                                                                            \
    REG_CMD_VAR(name, cmd, help)                                                                                      \
    INIT(name, PIC);


#define GLOBAL_VAR_ADD(type, name, unitSystem) CREATE_GLOBAL_VAR(type, name, unitSystem)


#define PARAM_DEFAULT(name, ...) REG_VAR_DEFAULT(name, __VA_ARGS__)


#define UNIT_SET_SCALING(name, fromSystem, unitSystem, ...)                                                           \
    REG_UNIT_CONVERSION(name, fromSystem, unitSystem, __VA_ARGS__)

#define UNIT_BASE(name, destinationUnitSystem, ...)                                                                   \
    UNIT_SET_SCALING(name, destinationUnitSystem, SI, __VA_ARGS__);                                                   \
    UNIT_SET_SCALING(name, SI, destinationUnitSystem, [&]() {                                                         \
        auto f = __VA_ARGS__;                                                                                         \
        return 1.0 / f();                                                                                             \
    })


    template<typename T_Scale, typename T>
    constexpr auto myPow(T_Scale scale, T v, int exp) -> T
    {
        return exp == 0 ? v : myPow(scale, v * scale, exp - 1);
    }


} // namespace picongpu
