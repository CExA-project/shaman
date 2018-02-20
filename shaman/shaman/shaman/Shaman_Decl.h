//
// Created by nestor on 17/01/18.
//

#ifndef SHAMAN_DECL_H
#define SHAMAN_DECL_H

#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include "Shaman.h"
#include "NumericalDebugger.h"

//-----------------------------------------------------------------------------
// OPERATOR CASTING MANAGEMENT
#ifdef EXPLICIT_CASTING

// arithmetic operator (between n and S<n,e,p> only)
#define set_Soperator_casts(OPERATOR) \
template<typename n, typename e, typename p> \
inline const S<n,e,p> operator OPERATOR (const n& n1, const S<n,e,p>& n2) \
{ \
    return S<n,e,p>(n1) OPERATOR n2; \
} \
template<typename n, typename e, typename p> \
inline const S<n,e,p> operator OPERATOR (const S<n,e,p>& n1, const n& n2) \
{ \
    return n1 OPERATOR S<n,e,p>(n2); \
} \

// boolean operator (between n and S<n,e,p> only)
#define set_Sbool_operator_casts(OPERATOR) \
template<typename n, typename e, typename p> \
inline bool operator OPERATOR (const n& n1, const S<n,e,p>& n2) \
{ \
    return S<n,e,p>(n1) OPERATOR n2; \
} \
template<typename n, typename e, typename p> \
inline bool operator OPERATOR (const S<n,e,p>& n1, const n& n2) \
{ \
    return n1 OPERATOR S<n,e,p>(n2); \
} \

// function (between n and S<n,e,p> only)
#define set_Sfunction2_casts(FUN) \
template<typename n, typename e, typename p> \
inline const S<n,e,p> FUN (const n& n1, const S<n,e,p>& n2) \
{ \
    return FUN(S<n,e,p>(n1), n2); \
} \
template<typename n, typename e, typename p> \
inline const S<n,e,p> FUN (const S<n,e,p>& n1, const n& n2) \
{ \
    return FUN(n1, S<n,e,p>(n2)); \
} \

// function with 3 arguments (between n and S<n,e,p> only)
#define set_Sfunction3_casts(FUN) \
template<typename n, typename e, typename p> \
inline const S<n,e,p> FUN (const n& n1, const n& n2, const S<n,e,p>& n3) \
{ \
    return FUN(S<n,e,p>(n1), S<n,e,p>(n2), n3); \
} \
template<typename n, typename e, typename p> \
inline const S<n,e,p> FUN (const n& n1, const S<n,e,p>& n2, const n& n3) \
{ \
    return FUN(S<n,e,p>(n1), n2, S<n,e,p>(n3)); \
} \
template<typename n, typename e, typename p> \
inline const S<n,e,p> FUN (const S<n,e,p>& n1, const n& n2, const n& n3) \
{ \
    return FUN(n1, S<n,e,p>(n2), S<n,e,p>(n3)); \
} \
template<typename n, typename e, typename p> \
inline const S<n,e,p> FUN (const n& n1, const S<n,e,p>& n2, const S<n,e,p>& n3) \
{ \
    return FUN(S<n,e,p>(n1), n2, n3); \
} \
template<typename n, typename e, typename p> \
inline const S<n,e,p> FUN (const S<n,e,p>& n1, const S<n,e,p>& n2, const n& n3) \
{ \
    return FUN(n1, n2, S<n,e,p>(n3)); \
} \
template<typename n, typename e, typename p> \
inline const S<n,e,p> FUN (const S<n,e,p>& n1, const n& n2, const S<n,e,p>& n3) \
{ \
    return FUN(n1, S<n,e,p>(n2), n3); \
} \

#else

/*
 * this section uses templates and macro (and the curry howard isomorphism)
 * to take an operator and produce overloads for all the needed combinaison of types
 * while preserving C++ implicit cast conventions
 */

#define arithmeticTYPE(T) T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type

// takes a value and builds an S type around its type
template<typename T>
inline S<T,T,T> makeStype(T t) { return S<T,T,T>(t); }; // default case
inline Sfloat makeStype(float t) { return Sfloat(t); };
inline Sdouble makeStype(double t) { return Sdouble(t); };
inline Slong_double makeStype(long double t) { return Slong_double(t); };
templated inline Snum makeStype(Snum s) { return s; };

// takes two values and builds a S type around the type C++ would use as a return type for their sum
#define SreturnType(t1,t2) decltype(makeStype(t1 + t2))
// takes three values and builds a S type around the type C++ would use as a return type for their fma
#define SreturnType3(t1,t2,t3) decltype(makeStype(std::fma(t1, t2, t3)))

//-----

// defines overload for arithmetic operators
#define set_Soperator_casts(OPERATOR) \
template<typename N, typename E, typename P, typename arithmeticTYPE(T)> \
inline auto operator OPERATOR (const S<N,E,P>& n1, const T& n2) \
{ \
    return SreturnType(n1.number,n2)(n1) OPERATOR SreturnType(n1.number,n2)(n2); \
} \
template<typename N, typename E, typename P, typename arithmeticTYPE(T)> \
inline auto operator OPERATOR (const T& n1, const S<N,E,P>& n2) \
{ \
    return SreturnType(n1,n2.number)(n1) OPERATOR SreturnType(n1,n2.number)(n2); \
} \
template<typename N1, typename E1, typename P1, typename N2, typename E2, typename P2> \
inline auto operator OPERATOR (const S<N1,E1,P1>& n1, const S<N2,E2,P2>& n2) \
{ \
    return SreturnType(n1.number,n2.number)(n1) OPERATOR SreturnType(n1.number,n2.number)(n2); \
} \

// defines overload for boolean operators
#define set_Sbool_operator_casts(OPERATOR) \
template<typename N, typename E, typename P, typename arithmeticTYPE(T)> \
inline bool operator OPERATOR (const S<N,E,P>& n1, const T& n2) \
{ \
    return SreturnType(n1.number,n2)(n1) OPERATOR SreturnType(n1.number,n2)(n2); \
} \
template<typename N, typename E, typename P, typename arithmeticTYPE(T)> \
inline bool operator OPERATOR (const T& n1, const S<N,E,P>& n2) \
{ \
    return SreturnType(n1,n2.number)(n1) OPERATOR SreturnType(n1,n2.number)(n2); \
} \
template<typename N1, typename E1, typename P1, typename N2, typename E2, typename P2> \
inline bool operator OPERATOR (const S<N1,E1,P1>& n1, const S<N2,E2,P2>& n2) \
{ \
    return SreturnType(n1.number,n2.number)(n1) OPERATOR SreturnType(n1.number,n2.number)(n2); \
} \

// defines overload for function taking two arguments
#define set_Sfunction2_casts(FUN) \
template<typename N, typename E, typename P, typename arithmeticTYPE(T)> \
inline auto FUN (const S<N,E,P>& n1, const T& n2) \
{ \
    return FUN(SreturnType(n1.number,n2)(n1), SreturnType(n1.number,n2)(n2)); \
} \
template<typename N, typename E, typename P, typename arithmeticTYPE(T)> \
inline auto FUN (const T& n1, const S<N,E,P>& n2) \
{ \
    return FUN(SreturnType(n1,n2.number)(n1), SreturnType(n1,n2.number)(n2)); \
} \
template<typename N1, typename E1, typename P1, typename N2, typename E2, typename P2> \
inline auto FUN (const S<N1,E1,P1>& n1, const S<N2,E2,P2>& n2) \
{ \
    return FUN(SreturnType(n1.number,n2.number)(n1), SreturnType(n1.number,n2.number)(n2)); \
} \

// defines overload for function taking three arguments
#define set_Sfunction3_casts(FUN) \
template<typename N1, typename E1, typename P1, typename N2, typename E2, typename P2, typename N3, typename E3, typename P3> \
inline auto FUN (const S<N1,E1,P1>& n1, const S<N2,E2,P2>& n2, const S<N3,E3,P3>& n3) \
{ \
    return FUN(SreturnType3(n1.number,n2.number,n3.number)(n1), SreturnType3(n1.number,n2.number,n3.number)(n2), SreturnType3(n1.number,n2.number,n3.number)(n3)); \
} \
template<typename N1, typename E1, typename P1, typename N2, typename E2, typename P2, typename arithmeticTYPE(T)> \
inline auto FUN (const T& n1, const S<N1,E1,P1>& n2, const S<N2,E2,P2>& n3) \
{ \
    return FUN(SreturnType3(n1,n2.number,n3.number)(n1), SreturnType3(n1,n2.number,n3.number)(n2), SreturnType3(n1,n2.number,n3.number)(n3)); \
} \
template<typename N1, typename E1, typename P1, typename N2, typename E2, typename P2, typename arithmeticTYPE(T)> \
inline auto FUN (const S<N1,E1,P1>& n1, const T& n2, const S<N2,E2,P2>& n3) \
{ \
    return FUN(SreturnType3(n1.number,n2,n3.number)(n1), SreturnType3(n1.number,n2,n3.number)(n2), SreturnType3(n1.number,n2,n3.number)(n3)); \
} \
template<typename N1, typename E1, typename P1, typename N2, typename E2, typename P2, typename arithmeticTYPE(T)> \
inline auto FUN (const S<N1,E1,P1>& n1, const S<N2,E2,P2>& n2, const T& n3) \
{ \
    return FUN(SreturnType3(n1.number,n2.number,n3)(n1), SreturnType3(n1.number,n2.number,n3)(n2), SreturnType3(n1.number,n2.number,n3)(n3)); \
} \
template<typename N, typename E, typename P, typename arithmeticTYPE(T1), typename arithmeticTYPE(T2)> \
inline auto FUN (const S<N,E,P>& n1, const T1& n2, const T2& n3) \
{ \
    return FUN(SreturnType3(n1.number,n2,n3)(n1), SreturnType3(n1.number,n2,n3)(n2), SreturnType3(n1.number,n2,n3)(n3)); \
} \
template<typename N, typename E, typename P, typename arithmeticTYPE(T1), typename arithmeticTYPE(T2)> \
inline auto FUN (const T1& n1, const S<N,E,P>& n2, const T2& n3) \
{ \
    return FUN(SreturnType3(n1,n2.number,n3)(n1), SreturnType3(n1,n2.number,n3)(n2), SreturnType3(n1,n2.number,n3)(n3)); \
} \
template<typename N, typename E, typename P, typename arithmeticTYPE(T1), typename arithmeticTYPE(T2)> \
inline auto FUN (const T1& n1, const T2& n2, const S<N,E,P>& n3) \
{ \
    return FUN(SreturnType3(n1,n2,n3.number)(n1), SreturnType3(n1,n2,n3.number)(n2), SreturnType3(n1,n2,n3.number)(n3)); \
} \

#endif
//-----------------------------------------------------------------------------
// ARITHMETIC OPERATORS

// basic EFT for a sum
template <typename T> inline const T eft2Sum(const T n1, const T n2, const T result)
{
    T n22 = result - n1;
    T n11 = result - n22;
    T epsilon2 = n2 - n22;
    T epsilon1 = n1 - n11;
    T error = epsilon1 + epsilon2;
    return error;
}

// fast EFT for a sum
template <typename T> inline const T eftFast2Sum(const T n1, const T n2, const T result)
{
    T n22 = result - n1;
    T error = n2 - n22;
    return error;
}

// fast EFT for a multiplication
template <typename T> inline const T eftFast2Mult(const T n1, const T n2, const T result)
{
    T error = std::fma(n1,n2,-result);
    return error;
}

// EFT for an FMA (cf "Some Functions Computable with a Fused-mac")
template <typename T> inline const T eftErrFma(const T n1, const T n2, const T n3, const T result)
{
    T u1 = n1*n2;
    T u2 = eftFast2Mult(n1,n2,u1);

    T alpha1 = n3 + u2;
    T alpha2 = eft2Sum(n3,u2,alpha1);

    T beta1 = u1 + alpha1;
    T beta2 = eft2Sum(u1,alpha1,beta1);

    T gamma = (beta1 - result) + beta2;
    T error1 = gamma + alpha2;
    T error2 = eftFast2Sum(gamma,alpha2,error1);

    // TODO if we sum error1 and error2 it might be better to ignore error2 (or to add it later)
    return error1 + error2;
}

//-----

// -
templated inline const Snum operator-(const Snum& n)
{
    return Snum(-n.number, -n.error);
};

// +
templated inline const Snum operator+(const Snum& n1, const Snum& n2)
{
    numberType result = n1.number + n2.number;

    numberType intermediateEFT = result - n1.number;
    numberType remainder = (n1.number - (result - intermediateEFT)) + (n2.number - intermediateEFT);
    errorType newError = remainder + (n1.error + n2.error);

    #ifdef NUMERICAL_DEBUGGER
    if (Snum::isCancelation(Snum::minPrecision(n1,n2), result, newError))
    {
        NumericalDebugger::cancelations++;
        NumericalDebugger::unstability();
    }
    #endif

    return Snum(result, newError);
};
set_Soperator_casts(+);

// -
templated inline const Snum operator-(const Snum& n1, const Snum& n2)
{
    numberType result = n1.number - n2.number;

    numberType intermediateEFT = result - n1.number;
    numberType remainder = (n1.number - (result - intermediateEFT)) + ((-n2.number) - intermediateEFT);
    errorType newError = remainder + (n1.error - n2.error);

    #ifdef NUMERICAL_DEBUGGER
    if (Snum::isCancelation(Snum::minPrecision(n1,n2), result, newError))
    {
        NumericalDebugger::cancelations++;
        NumericalDebugger::unstability();
    }
    #endif

    return Snum(result, newError);
};
set_Soperator_casts(-);

// *
templated inline const Snum operator*(const Snum& n1, const Snum& n2)
{
    numberType result = n1.number * n2.number;

    numberType remainder = std::fma(n1.number, n2.number, - result);
    //errorType newError = remainder + (n1.number*n2.error + n2.number*n1.error);
    errorType newError = std::fma(n1.number, n2.error, std::fma(n2.number, n1.error, remainder));

    #ifdef NUMERICAL_DEBUGGER
    if (n1.non_significativ() && n2.non_significativ())
    {
        NumericalDebugger::unstableMultiplications++;
        NumericalDebugger::unstability();
    }
    #endif

    return Snum(result, newError);
};
set_Soperator_casts(*);

// /
templated inline const Snum operator/(const Snum& n1, const Snum& n2)
{
    numberType result = n1.number / n2.number;

    numberType remainder = - std::fma(n2.number, result, -n1.number);
    //errorType newError = ((remainder + n1.error) - result*n2.error) / (n2.number + n2.error);
    errorType newError = - std::fma(result, n2.error, -(remainder + n1.error)) / (n2.number + n2.error);

    #ifdef NUMERICAL_DEBUGGER
    if (n2.non_significativ())
    {
        NumericalDebugger::unstableDivisions++;
        NumericalDebugger::unstability();
    }
    #endif

    return Snum(result, newError);
};
set_Soperator_casts(/);

//-----------------------------------------------------------------------------
// MATHEMATICAL FUNCTIONS

// isifnite
templated inline bool isfinite(const Snum& n)
{
    return std::isfinite(n.number);
};

// abs
templated inline const Snum abs(const Snum& n)
{
    if (n.number >= 0)
    {
        return Snum(n.number, n.error);
    }
    else
    {
        return Snum(-n.number, -n.error);
    }
};

// fabs
templated inline const Snum fabs(const Snum& n)
{
    return abs(n);
};

// sqrt
templated inline const Snum sqrt(const Snum& n)
{
    numberType result = std::sqrt(n.number);

    errorType newError;
    if (result == 0)
    {
        newError = (errorType) std::sqrt((preciseType) std::abs(n.error));
    }
    else
    {
        numberType remainder = - std::fma(result, result, - n.number);
        newError = (remainder + n.error) / (result + result);
    }
    // alternativ formula that is not based on derivation :
    // newError = (errorType) std::sqrt((preciseType) std::abs(n.number + (n.error - remainder))) - result;

    #ifdef NUMERICAL_DEBUGGER
    if (n.non_significativ())
    {
        NumericalDebugger::unstableFunctions++;
        NumericalDebugger::unstability();
    }
    #endif

    return Snum(result, newError);
};

// cubic root
// TODO might be improved using a formula similar to the one used for the square root
templated inline const Snum cbrt(const Snum& n)
{
    numberType result = std::cbrt(n.number);
    preciseType preciseCorrectedResult = std::cbrt((preciseType) n.number + n.error);
    errorType newError = (errorType) (preciseCorrectedResult - result);

    #ifdef NUMERICAL_DEBUGGER
    if (n.non_significativ())
    {
        NumericalDebugger::unstableFunctions++;
        NumericalDebugger::unstability();
    }
    #endif

    return Snum(result, newError);
};

// exp
templated inline const Snum exp(const Snum& n)
{
    numberType result = std::exp(n.number);
    preciseType preciseCorrectedResult = std::exp((preciseType) n.number + n.error);
    errorType newError = (errorType) (preciseCorrectedResult - result);

    #ifdef NUMERICAL_DEBUGGER
    if (n.non_significativ())
    {
        NumericalDebugger::unstableFunctions++;
        NumericalDebugger::unstability();
    }
    #endif

    return Snum(result, newError);
};

// sin
templated inline const Snum sin(const Snum& n)
{
    numberType result = std::sin(n.number);
    preciseType preciseCorrectedResult = std::sin((preciseType) n.number + n.error);
    errorType newError = (errorType) (preciseCorrectedResult - result);

    #ifdef NUMERICAL_DEBUGGER
    if (n.non_significativ())
    {
        NumericalDebugger::unstableFunctions++;
        NumericalDebugger::unstability();
    }
    #endif

    return Snum(result, newError);
};

// cos
templated inline const Snum cos(const Snum& n)
{
    numberType result = std::cos(n.number);
    preciseType preciseCorrectedResult = std::cos((preciseType) n.number + n.error);
    errorType newError = (errorType) (preciseCorrectedResult - result);

    #ifdef NUMERICAL_DEBUGGER
    if (n.non_significativ())
    {
        NumericalDebugger::unstableFunctions++;
        NumericalDebugger::unstability();
    }
    #endif

    return Snum(result, newError);
};

// tan
templated inline const Snum tan(const Snum& n)
{
    numberType result = std::tan(n.number);
    preciseType preciseCorrectedResult = std::tan((preciseType) n.number + n.error);
    errorType newError = (errorType) (preciseCorrectedResult - result);

    #ifdef NUMERICAL_DEBUGGER
    if (n.non_significativ())
    {
        NumericalDebugger::unstableFunctions++;
        NumericalDebugger::unstability();
    }
    #endif

    return Snum(result, newError);
};

// pow
templated inline const Snum pow(const Snum& n1, const Snum& n2)
{
    numberType result = std::pow(n1.number, n2.number);
    preciseType preciseCorrectedResult = std::pow((preciseType) n1.number + n1.error, (preciseType) n2.number + n2.error);
    errorType newError = (errorType) (preciseCorrectedResult - result);

    #ifdef NUMERICAL_DEBUGGER
    if (n1.non_significativ() || n2.non_significativ())
    {
        NumericalDebugger::unstablePowerFunctions++;
        NumericalDebugger::unstability();
    }
    #endif

    return Snum(result, newError);
};
set_Sfunction2_casts(pow);

// min
templated inline const Snum min(const Snum& n1, const Snum& n2)
{
    #ifdef NUMERICAL_DEBUGGER
    if (Snum::isUnstableBranchings(n1, n2))
    {
        NumericalDebugger::unstableBranchings++;
        NumericalDebugger::unstability();
    }
    #endif

    if (n1.number <= n2.number)
    {
        return n1;
    }
    else
    {
        return n2;
    }
};
set_Sfunction2_casts(min);

// max
templated inline const Snum max(const Snum& n1, const Snum& n2)
{
    #ifdef NUMERICAL_DEBUGGER
    if (Snum::isUnstableBranchings(n1, n2))
    {
        NumericalDebugger::unstableBranchings++;
        NumericalDebugger::unstability();
    }
    #endif

    if (n1.number >= n2.number)
    {
        return n1;
    }
    else
    {
        return n2;
    }
};
set_Sfunction2_casts(max);

// fma
templated inline const Snum fma(const Snum& n1, const Snum& n2, const Snum& n3)
{
    numberType result = std::fma(n1.number, n2.number, n3.number);

    numberType remainder = eftErrFma(n1.number, n2.number, n3.number, result);
    //errorType newError = remainder + (n1.number*n2.error + n2.number*n1.error) + n3.error;
    errorType newError = std::fma(n2.number, n1.error, std::fma(n1.number, n2.error, remainder + n3.error));

    #ifdef NUMERICAL_DEBUGGER
    if (Snum::isCancelation(Snum::minPrecision(n1,Snum::minPrecision(n2,n3)), result, newError))
    {
        NumericalDebugger::cancelations++;
        NumericalDebugger::unstability();
    }
    #endif

    return Snum(result, newError);
}
set_Sfunction3_casts(fma);

//-----------------------------------------------------------------------------
// CLASS OPERATORS

// +=
templated inline Snum& Snum::operator+=(const Snum& n)
{
    *this = (*this) + n;

    return *this;
}

// -=
templated inline Snum& Snum::operator-=(const Snum& n)
{
    *this = (*this) - n;

    return *this;
}

// *=
templated inline Snum& Snum::operator*=(const Snum& n)
{
    *this = (*this) * n;

    return *this;
}

// /=
templated inline Snum& Snum::operator/=(const Snum& n)
{
    *this = (*this) / n;

    return *this;
}

/*
 * streaming operator, displays only the significative digits
 * (alternative version : outputs the number, the error and the number of significative digits)
 */
templated inline std::ostream& operator<<(std::ostream& os, const Snum& n)
{
    // full information
    //os << n.number << " (error:" << n.error << " digits:" << Snum::digits(n) << ')';
    int nbDigitsMax = 17;
    numberType fdigits = std::floor(Snum::digits(n));

    if (fdigits <= 0)
    {
        os << "@.0";    // the number has no meaning
    }
    else
    {
        int digits;
        if (fdigits >= nbDigitsMax)
        {
            digits = nbDigitsMax; // no need for too many digits
        }
        else
        {
            digits = (int) fdigits; // no risk of overflow
        }

        os << std::scientific << std::setprecision(digits-1) << n.number;
    }

    return os;
}

/*
 * convert a Ntype into a string
 *
 * using the streaming operator
 * there is probably a more efficient implementation
 * but it is the easier way to benefit from std::scientific and std::setprecision
 */
templated Snum::operator std::string() const
{
    std::ostringstream stream;
    stream << this;
    return stream.str();
}

//-----------------------------------------------------------------------------
// BOOLEAN OPERATORS

// ==
templated inline bool operator==(const Snum& n1, const Snum& n2)
{
    #ifdef NUMERICAL_DEBUGGER
    if (Snum::isUnstableBranchings(n1, n2))
    {
        NumericalDebugger::unstableBranchings++;
        NumericalDebugger::unstability();
    }
    #endif

    return n1.number == n2.number;
};
set_Sbool_operator_casts(==);

// !=
templated inline bool operator!=(const Snum& n1, const Snum& n2)
{
    #ifdef NUMERICAL_DEBUGGER
    if (Snum::isUnstableBranchings(n1, n2))
    {
        NumericalDebugger::unstableBranchings++;
        NumericalDebugger::unstability();
    }
    #endif

    return n1.number != n2.number;
};
set_Sbool_operator_casts(!=);

// <
templated inline bool operator<(const Snum& n1, const Snum& n2)
{
    #ifdef NUMERICAL_DEBUGGER
    if (Snum::isUnstableBranchings(n1, n2))
    {
        NumericalDebugger::unstableBranchings++;
        NumericalDebugger::unstability();
    }
    #endif

    return n1.number < n2.number;
};
set_Sbool_operator_casts(<);

// <=
templated inline bool operator<=(const Snum& n1, const Snum& n2)
{
    #ifdef NUMERICAL_DEBUGGER
    if (Snum::isUnstableBranchings(n1, n2))
    {
        NumericalDebugger::unstableBranchings++;
        NumericalDebugger::unstability();
    }
    #endif

    return n1.number <= n2.number;
};
set_Sbool_operator_casts(<=);

// >
templated inline bool operator>(const Snum& n1, const Snum& n2)
{
    #ifdef NUMERICAL_DEBUGGER
    if (Snum::isUnstableBranchings(n1, n2))
    {
        NumericalDebugger::unstableBranchings++;
        NumericalDebugger::unstability();
    }
    #endif

    return n1.number > n2.number;
};
set_Sbool_operator_casts(>);

// >=
templated inline bool operator>=(const Snum& n1, const Snum& n2)
{
    #ifdef NUMERICAL_DEBUGGER
    if (Snum::isUnstableBranchings(n1, n2))
    {
        NumericalDebugger::unstableBranchings++;
        NumericalDebugger::unstability();
    }
    #endif

    return n1.number >= n2.number;
};
set_Sbool_operator_casts(>=);

//-----------------------------------------------------------------------------
// SIGNIFICATIV DIGITS

/*
 * returns the number of significative digits of a couple (number,error)
 */
templated inline numberType Snum::digits(numberType number, errorType error)
{
    if (error == 0)
    {
        return INFINITY;
    }
    else
    {
        numberType relativeError = std::abs((numberType) error/number);

        if (relativeError >= 1)
        {
            return 0;
        }
        else
        {
            return -log10(relativeError);
        }
    }
}

/*
 * returns the number of significative digits of a S
 */
templated inline numberType Snum::digits(const Snum& n)
{
    return digits(n.number, n.error);
}

//-----------------------------------------------------------------------------
// UNSTABILITY DETECTION

/*
 * returns true if the couple (number,error) has no significative digits in the base
 *
 * NOTE : 'error != 0' is facultative
 * it slightly improves the performances on some test cases
 * since it is common to have numbers with 0 error
 * (any number that just been turned into a S)
 */
templated inline bool Snum::non_significativ(numberType number, errorType error)
{
    int base = 10;
    return (error != 0) && (std::abs(number) < base * std::abs(error));
}

/*
 * returns true if the current S has no significative digits in the base
 */
templated inline bool Snum::non_significativ() const
{
    return non_significativ(number, error);
}

/*
 * return the Snum with the lowest precision
 */
templated inline Snum Snum::minPrecision(const Snum& n1, const Snum& n2)
{
    if ( std::abs(n1.error * n2.number) > std::abs(n1.number * n2.error) )
    {
        return n1;
    }
    else
    {
        return n2;
    }
}

/*
 * detects cancelations
 */
templated inline bool Snum::isCancelation(const Snum& n, numberType result, errorType resultingError)
{
    int cancel_level = 4;
    int base = 10;

    // have we lost more than cancel_level significative digits ?
    return (n.error != 0) && (std::abs(resultingError * n.number) > pow(base, cancel_level) * std::abs(n.error * result));
}

/*
 * detects unstable branchings
 */
templated inline bool Snum::isUnstableBranchings(const Snum& n1, const Snum& n2)
{
    return non_significativ(n1.number - n2.number, n1.error - n2.error);
}

#endif //SHAMAN_DECL_H