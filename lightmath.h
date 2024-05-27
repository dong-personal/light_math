// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2007-2011 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CORE_H
#define EIGEN_CORE_H

// first thing Eigen does: stop the compiler from reporting useless warnings.
#include "Core/util/DisableStupidWarnings.h"

// then include this file where all our macros are defined. It's really important to do it first because
// it's where we do all the compiler/OS/arch detections and define most defaults.
#include "Core/util/Macros.h"

// This detects SSE/AVX/NEON/etc. and configure alignment settings
#include "Core/util/ConfigureVectorization.h"

// We need cuda_runtime.h/hip_runtime.h to ensure that
// the EIGEN_USING_STD macro works properly on the device side
#if defined(EIGEN_CUDACC)
  #include <cuda_runtime.h>
#elif defined(EIGEN_HIPCC)
  #include <hip/hip_runtime.h>
#endif


#ifdef EIGEN_EXCEPTIONS
  #include <new>
#endif

// Disable the ipa-cp-clone optimization flag with MinGW 6.x or newer (enabled by default with -O3)
// See http://eigen.tuxfamily.org/bz/show_bug.cgi?id=556 for details.
#if EIGEN_COMP_MINGW && EIGEN_GNUC_AT_LEAST(4,6) && EIGEN_GNUC_AT_MOST(5,5)
  #pragma GCC optimize ("-fno-ipa-cp-clone")
#endif

// Prevent ICC from specializing std::complex operators that silently fail
// on device. This allows us to use our own device-compatible specializations
// instead.
#if defined(EIGEN_COMP_ICC) && defined(EIGEN_GPU_COMPILE_PHASE) \
    && !defined(_OVERRIDE_COMPLEX_SPECIALIZATION_)
#define _OVERRIDE_COMPLEX_SPECIALIZATION_ 1
#endif
#include <complex>
using std::complex_literals::operator""if;
using std::complex_literals::operator""i;
using std::complex_literals::operator""il;
// this include file manages BLAS and MKL related macros
// and inclusion of their respective header files
#include "Core/util/MKL_support.h"


#if defined(EIGEN_HAS_CUDA_FP16) || defined(EIGEN_HAS_HIP_FP16)
  #define EIGEN_HAS_GPU_FP16
#endif

#if defined(EIGEN_HAS_CUDA_BF16) || defined(EIGEN_HAS_HIP_BF16)
  #define EIGEN_HAS_GPU_BF16
#endif

#if (defined _OPENMP) && (!defined EIGEN_DONT_PARALLELIZE)
  #define EIGEN_HAS_OPENMP
#endif

#ifdef EIGEN_HAS_OPENMP
#include <omp.h>

#endif

// MSVC for windows mobile does not have the errno.h file
#if !(EIGEN_COMP_MSVC && EIGEN_OS_WINCE) && !EIGEN_COMP_ARM
#define EIGEN_HAS_ERRNO
#endif

#ifdef EIGEN_HAS_ERRNO
#include <cerrno>
#endif
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <random>
#include <cassert>
#include <functional>
#ifndef EIGEN_NO_IO
  #include <sstream>
  #include <iosfwd>
#endif
#include <cstring>
#include <string>
#include <limits>
#include <climits> // for CHAR_BIT
// for min/max:
#include <algorithm>

#if EIGEN_HAS_CXX11
#include <array>
#endif

// for std::is_nothrow_move_assignable
#ifdef EIGEN_INCLUDE_TYPE_TRAITS
#include <type_traits>
#endif

// for outputting debug info
#ifdef EIGEN_DEBUG_ASSIGN
#include <iostream>
#endif

// required for __cpuid, needs to be included after cmath
// also required for _BitScanReverse on Windows on ARM
#if EIGEN_COMP_MSVC && (EIGEN_ARCH_i386_OR_x86_64 || EIGEN_ARCH_ARM64) && !EIGEN_OS_WINCE
  #include <intrin.h>
#endif

#if defined(EIGEN_USE_SYCL)
  #undef min
  #undef max
  #undef isnan
  #undef isinf
  #undef isfinite
  #include <CL/sycl.hpp>
  #include <map>
  #include <memory>
  #include <utility>
  #include <thread>
  #ifndef EIGEN_SYCL_LOCAL_THREAD_DIM0
  #define EIGEN_SYCL_LOCAL_THREAD_DIM0 16
  #endif
  #ifndef EIGEN_SYCL_LOCAL_THREAD_DIM1
  #define EIGEN_SYCL_LOCAL_THREAD_DIM1 16
  #endif
#endif


#if defined EIGEN2_SUPPORT_STAGE40_FULL_EIGEN3_STRICTNESS || defined EIGEN2_SUPPORT_STAGE30_FULL_EIGEN3_API || defined EIGEN2_SUPPORT_STAGE20_RESOLVE_API_CONFLICTS || defined EIGEN2_SUPPORT_STAGE10_FULL_EIGEN2_API || defined EIGEN2_SUPPORT
// This will generate an error message:
#error Eigen2-support is only available up to version 3.2. Please go to "http://eigen.tuxfamily.org/index.php?title=Eigen2" for further information
#endif

namespace Eigen {

// we use size_t frequently and we'll never remember to prepend it with std:: every time just to
// ensure QNX/QCC support
using std::size_t;
// gcc 4.6.0 wants std:: for ptrdiff_t
using std::ptrdiff_t;

}

/** \defgroup Core_Module Core module
  * This is the main module of Eigen providing dense matrix and vector support
  * (both fixed and dynamic size) with all the features corresponding to a BLAS library
  * and much more...
  *
  * \code
  * #include <Eigen/Core>
  * \endcode
  */

#include "Core/util/Constants.h"
#include "Core/util/Meta.h"
#include "Core/util/ForwardDeclarations.h"
#include "Core/util/StaticAssert.h"
#include "Core/util/XprHelper.h"
#include "Core/util/Memory.h"
#include "Core/util/IntegralConstant.h"
#include "Core/util/SymbolicIndex.h"

#include "Core/NumTraits.h"
#include "Core/MathFunctions.h"
#include "Core/GenericPacketMath.h"
#include "Core/MathFunctionsImpl.h"
#include "Core/arch/Default/ConjHelper.h"
// Generic half float support
#include "Core/arch/Default/Half.h"
#include "Core/arch/Default/BFloat16.h"
#include "Core/arch/Default/TypeCasting.h"
#include "Core/arch/Default/GenericPacketMathFunctionsFwd.h"

#if defined EIGEN_VECTORIZE_AVX512
  #include "Core/arch/SSE/PacketMath.h"
  #include "Core/arch/SSE/TypeCasting.h"
  #include "Core/arch/SSE/Complex.h"
  #include "Core/arch/AVX/PacketMath.h"
  #include "Core/arch/AVX/TypeCasting.h"
  #include "Core/arch/AVX/Complex.h"
  #include "Core/arch/AVX512/PacketMath.h"
  #include "Core/arch/AVX512/TypeCasting.h"
  #include "Core/arch/AVX512/Complex.h"
  #include "Core/arch/SSE/MathFunctions.h"
  #include "Core/arch/AVX/MathFunctions.h"
  #include "Core/arch/AVX512/MathFunctions.h"
#elif defined EIGEN_VECTORIZE_AVX
  // Use AVX for floats and doubles, SSE for integers
  #include "Core/arch/SSE/PacketMath.h"
  #include "Core/arch/SSE/TypeCasting.h"
  #include "Core/arch/SSE/Complex.h"
  #include "Core/arch/AVX/PacketMath.h"
  #include "Core/arch/AVX/TypeCasting.h"
  #include "Core/arch/AVX/Complex.h"
  #include "Core/arch/SSE/MathFunctions.h"
  #include "Core/arch/AVX/MathFunctions.h"
#elif defined EIGEN_VECTORIZE_SSE
  #include "Core/arch/SSE/PacketMath.h"
  #include "Core/arch/SSE/TypeCasting.h"
  #include "Core/arch/SSE/MathFunctions.h"
  #include "Core/arch/SSE/Complex.h"
#elif defined(EIGEN_VECTORIZE_ALTIVEC) || defined(EIGEN_VECTORIZE_VSX)
  #include "Core/arch/AltiVec/PacketMath.h"
  #include "Core/arch/AltiVec/MathFunctions.h"
  #include "Core/arch/AltiVec/Complex.h"
#elif defined EIGEN_VECTORIZE_NEON
  #include "Core/arch/NEON/PacketMath.h"
  #include "Core/arch/NEON/TypeCasting.h"
  #include "Core/arch/NEON/MathFunctions.h"
  #include "Core/arch/NEON/Complex.h"
#elif defined EIGEN_VECTORIZE_SVE
  #include "Core/arch/SVE/PacketMath.h"
  #include "Core/arch/SVE/TypeCasting.h"
  #include "Core/arch/SVE/MathFunctions.h"
#elif defined EIGEN_VECTORIZE_ZVECTOR
  #include "Core/arch/ZVector/PacketMath.h"
  #include "Core/arch/ZVector/MathFunctions.h"
  #include "Core/arch/ZVector/Complex.h"
#elif defined EIGEN_VECTORIZE_MSA
  #include "Core/arch/MSA/PacketMath.h"
  #include "Core/arch/MSA/MathFunctions.h"
  #include "Core/arch/MSA/Complex.h"
#endif

#if defined EIGEN_VECTORIZE_GPU
  #include "Core/arch/GPU/PacketMath.h"
  #include "Core/arch/GPU/MathFunctions.h"
  #include "Core/arch/GPU/TypeCasting.h"
#endif

#if defined(EIGEN_USE_SYCL)
  #include "Core/arch/SYCL/SyclMemoryModel.h"
  #include "Core/arch/SYCL/InteropHeaders.h"
#if !defined(EIGEN_DONT_VECTORIZE_SYCL)
  #include "Core/arch/SYCL/PacketMath.h"
  #include "Core/arch/SYCL/MathFunctions.h"
  #include "Core/arch/SYCL/TypeCasting.h"
#endif
#endif

#include "Core/arch/Default/Settings.h"
// This file provides generic implementations valid for scalar as well
#include "Core/arch/Default/GenericPacketMathFunctions.h"

#include "Core/functors/TernaryFunctors.h"
#include "Core/functors/BinaryFunctors.h"
#include "Core/functors/UnaryFunctors.h"
#include "Core/functors/NullaryFunctors.h"
#include "Core/functors/StlFunctors.h"
#include "Core/functors/AssignmentFunctors.h"

// Specialized functors to enable the processing of complex numbers
// on CUDA devices
#ifdef EIGEN_CUDACC
#include "Core/arch/CUDA/Complex.h"
#endif

#include "Core/util/IndexedViewHelper.h"
#include "Core/util/ReshapedHelper.h"
//#include "Core/ArithmeticSequence.h"
#ifndef EIGEN_NO_IO
  #include "Core/IO.h"
#endif
#include "Core/DenseCoeffsBase.h"
#include "Core/DenseBase.h"
//#include "Core/MatrixBase.h"
#include "Core/EigenBase.h"

//#include "Core/Product.h"
#include "Core/CoreEvaluators.h"
#include "Core/AssignEvaluator.h"

#ifndef EIGEN_PARSED_BY_DOXYGEN // work around Doxygen bug triggered by Assign.h r814874
                                // at least confirmed with Doxygen 1.5.5 and 1.5.6
  #include "Core/Assign.h"
#endif

#include "Core/ArrayBase.h"
#include "Core/util/BlasUtil.h"
#include "Core/DenseStorage.h"
// #include "Core/NestByValue.h"

// #include "Core/ForceAlignedAccess.h"

//#include "Core/ReturnByValue.h"
// #include "Core/NoAlias.h"
#include "Core/PlainObjectBase.h"
//#include "Core/Matrix.h"
#include "Core/Array.h"
#include "Core/CwiseTernaryOp.h"
#include "Core/CwiseBinaryOp.h"
#include "Core/CwiseUnaryOp.h"
#include "Core/CwiseNullaryOp.h"
//#include "Core/CwiseUnaryView.h"
#include "Core/SelfCwiseBinaryOp.h"
#include "Core/Dot.h"
//#include "Core/StableNorm.h"
#include "Core/Stride.h"
#include "Core/MapBase.h"
#include "Core/Map.h"
//#include "Core/Ref.h"
#include "Core/Block.h"
//#include "Core/VectorBlock.h"
//#include "Core/IndexedView.h"
//#include "Core/Reshaped.h"
//#include "Core/Transpose.h"
//#include "Core/DiagonalMatrix.h"
//#include "Core/Diagonal.h"
//#include "Core/DiagonalProduct.h"
#include "Core/Redux.h"
//#include "Core/Visitor.h"
//#include "Core/Fuzzy.h"
//#include "Core/Swap.h"
#include "Core/CommaInitializer.h"
//#include "Core/GeneralProduct.h"
//#include "Core/Solve.h"
// #include "Core/Inverse.h"
//#include "Core/SolverBase.h"
//#include "Core/PermutationMatrix.h"
//#include "Core/Transpositions.h"
//#include "Core/TriangularMatrix.h"
//#include "Core/SelfAdjointView.h"
//#include "Core/products/GeneralBlockPanelKernel.h"
//#include "Core/products/Parallelizer.h"
//#include "Core/ProductEvaluators.h"
//#include "Core/products/GeneralMatrixVector.h"
//#include "Core/products/GeneralMatrixMatrix.h"
//#include "Core/SolveTriangular.h"
//#include "Core/products/GeneralMatrixMatrixTriangular.h"
//#include "Core/products/SelfadjointMatrixVector.h"
//#include "Core/products/SelfadjointMatrixMatrix.h"
//#include "Core/products/SelfadjointProduct.h"
//#include "Core/products/SelfadjointRank2Update.h"
//#include "Core/products/TriangularMatrixVector.h"
//#include "Core/products/TriangularMatrixMatrix.h"
//#include "Core/products/TriangularSolverMatrix.h"
//#include "Core/products/TriangularSolverVector.h"
//#include "Core/BandMatrix.h"
//#include "Core/CoreIterators.h"
//#include "Core/ConditionEstimator.h"

#if defined(EIGEN_VECTORIZE_ALTIVEC) || defined(EIGEN_VECTORIZE_VSX)
  #include "Core/arch/AltiVec/MatrixProduct.h"
#elif defined EIGEN_VECTORIZE_NEON
  #include "Core/arch/NEON/GeneralBlockPanelKernel.h"
#endif

//#include "Core/BooleanRedux.h"
//#include "Core/Select.h"
//#include "Core/VectorwiseOp.h"
//#include "Core/PartialReduxEvaluator.h"
#include "Core/Random.h"
//#include "Core/Replicate.h"
//#include "Core/Reverse.h"
//#include "Core/ArrayWrapper.h"
//#include "Core/StlIterators.h"

#ifdef EIGEN_USE_BLAS
#include "Core/products/GeneralMatrixMatrix_BLAS.h"
#include "Core/products/GeneralMatrixVector_BLAS.h"
#include "Core/products/GeneralMatrixMatrixTriangular_BLAS.h"
#include "Core/products/SelfadjointMatrixMatrix_BLAS.h"
#include "Core/products/SelfadjointMatrixVector_BLAS.h"
#include "Core/products/TriangularMatrixMatrix_BLAS.h"
#include "Core/products/TriangularMatrixVector_BLAS.h"
#include "Core/products/TriangularSolverMatrix_BLAS.h"
#endif // EIGEN_USE_BLAS

#ifdef EIGEN_USE_MKL_VML
#include "Core/Assign_MKL.h"
#endif

#include "Core/GlobalFunctions.h"

#include "Core/util/ReenableStupidWarnings.h"

#endif // EIGEN_CORE_H
