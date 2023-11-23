//==============================================================================
//
// Copyright (c) 2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

// Compiler builtin intrinsic functions should be specified in this file

#ifndef BUILTIN_INTRINSICS_H_
#define BUILTIN_INTRINSICS_H_

// Branch prediction
#if defined(__clang__)

#define HEX_LIKELY(x)   __builtin_expect(!!(x), 1)
#define HEX_UNLIKELY(x) __builtin_expect(!!(x), 0)

#define HEX_ASSUME      __builtin_assume
#define HEX_UNREACHABLE __builtin_unreachable

#elif defined(_MSC_VER)

#define HEX_LIKELY(x)   (x)
#define HEX_UNLIKELY(x) (x)

#define HEX_ASSUME        __assume
#define HEX_UNREACHABLE() __assume(0)

#elif defined(__GNUC__)
//No equivalent __builtin_assume in GNUC. Hence leaving empty.
#define HEX_ASSUME(cond)

#define HEX_LIKELY(x)   __builtin_expect(!!(x), 1)
#define HEX_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define HEX_UNREACHABLE __builtin_unreachable

#endif // defined(__clang__)

// Overflow detection
#if defined(__clang__) || defined(__GNUC__)

#define HEX_ADD_OVERFLOW __builtin_add_overflow
#define HEX_MUL_OVERFLOW __builtin_mul_overflow

#elif defined(_MSC_VER)

#include <limits>

template <typename _T> static inline bool HEX_ADD_OVERFLOW(_T a, _T b, _T *out)
{
    *out = a + b;
    return ((b > 0) && (a > std::numeric_limits<_T>::max() - b)) ||
           ((b < 0) && (a < std::numeric_limits<_T>::min() - b));
}

template <typename _T> static inline bool HEX_MUL_OVERFLOW(_T a, _T b, _T *out)
{
    *out = a * b;
    return ((b > 0) && (a > std::numeric_limits<_T>::max() / b || a < std::numeric_limits<_T>::min() / b)) ||
           ((b < 0) && (a > std::numeric_limits<_T>::min() / b || a < std::numeric_limits<_T>::max() / b));
}

#endif // __clang__

// Count bits

#include <bitset>

template <typename _T> static inline int HEX_COUNT_ONE_BIT(_T x)
{
    return std::bitset<sizeof(_T) * 8>(x).count();
}

#define HEX_COUNT_ONE_BIT_ULL HEX_COUNT_ONE_BIT
#define HEX_COUNT_ONE_BIT_UL  HEX_COUNT_ONE_BIT

#if defined(__clang__) || defined(__GNUC__)

#define HEX_COUNT_LEADING_ZERO     __builtin_clz
#define HEX_COUNT_LEADING_ZERO_UL  __builtin_clzl
#define HEX_COUNT_LEADING_ZERO_ULL __builtin_clzll

#define HEX_COUNT_TRAILING_ZERO     __builtin_ctz
#define HEX_COUNT_TRAILING_ZERO_UL  __builtin_ctzl
#define HEX_COUNT_TRAILING_ZERO_ULL __builtin_ctzll

#elif defined(_MSC_VER)

#include <intrin.h>

// Returns the number of leading 0-bits in x, starting at the most significant
// bit position. If x is 0, the result is undefined.
static inline int HEX_COUNT_LEADING_ZERO_ULL(unsigned long long x)
{
    unsigned long where;
    if (_BitScanReverse64(&where, x)) return static_cast<int>(63 - where);
    return 64; // Undefined behavior
}

static inline int HEX_COUNT_LEADING_ZERO(unsigned int x)
{
    unsigned long where;
    if (_BitScanReverse(&where, x)) return static_cast<int>(31 - where);
    return 32; // Undefined Behavior.
}

static inline int HEX_COUNT_LEADING_ZERO_UL(unsigned long x)
{
    return sizeof(x) == 8 ? HEX_COUNT_LEADING_ZERO_ULL(x) : HEX_COUNT_LEADING_ZERO(static_cast<unsigned int>(x));
}

// Returns the number of trailing 0-bits in x, starting at the least significant
// bit position. If x is 0, the result is undefined.
static inline int HEX_COUNT_TRAILING_ZERO_ULL(unsigned long long x)
{
    unsigned long where;
    if (_BitScanForward64(&where, x)) return static_cast<int>(where);
    return 64; // Undefined Behavior.
}

static inline int HEX_COUNT_TRAILING_ZERO(unsigned int x)
{
    unsigned long where;
    if (_BitScanForward(&where, x)) return static_cast<int>(where);
    return 32; // Undefined Behavior.
}

static inline int HEX_COUNT_TRAILING_ZERO_UL(unsigned long x)
{
    return sizeof(x) == 8 ? HEX_COUNT_TRAILING_ZERO_ULL(x) : HEX_COUNT_TRAILING_ZERO(static_cast<unsigned int>(x));
}

#endif // defined(__clang__)

// Atomic operation

#if defined(__clang__) || defined(__GNUC__)

#define HEX_ATOMIC_FETCH_AND_ADD __sync_fetch_and_add

#define HEX_ATOMIC_FETCH_AND_AND __sync_fetch_and_and
#define HEX_ATOMIC_FETCH_AND_OR  __sync_fetch_and_or

#define HEX_ATOMIC_VAL_COMPARE_AND_SWAP  __sync_val_compare_and_swap
#define HEX_ATOMIC_BOOL_COMPARE_AND_SWAP __sync_bool_compare_and_swap

#elif defined(_MSC_VER)

#include <intrin.h>

#define HEX_ATOMIC_FETCH_AND_ADD(_p, _v)                                                                               \
    (sizeof *(_p) == sizeof(__int64) ? _InterlockedExchangeAdd64((__int64 *)_p, (__int64)_v)                           \
                                     : _InterlockedExchangeAdd((long *)_p, (long)v))

template <typename _T> static inline _T HEX_ATOMIC_FETCH_AND_AND(_T volatile *_p, _T _v)
{
    _InterlockedAnd((long *)_p, (long)_v);
    return static_cast<_T>(*_p);
}

template <typename _T> static inline _T HEX_ATOMIC_FETCH_AND_OR(_T volatile *_p, _T _v)
{
    _InterlockedOr((long *)_p, (long)_v);
    return static_cast<_T>(*_p);
}

#define HEX_ATOMIC_VAL_COMPARE_AND_SWAP(_p, _old, _new)                                                                \
    (sizeof *(_p) == sizeof(__int64) ? _InterlockedCompareExchange64((__int64 *)_p, (__int64)_new, (__int64)_old)      \
                                     : _InterlockedCompareExchange((long *)_p, (long)_new, (long)_old))

#define HEX_ATOMIC_BOOL_COMPARE_AND_SWAP(_p, _old, _new) (HEX_ATOMIC_VAL_COMPARE_AND_SWAP(_p, _old, _new) == _old)

#endif // defined(__clang__)

#endif /* BUILTIN_INTRINSICS_H_ */
