#ifndef AWS_COMMON_CONFIG_H
#define AWS_COMMON_CONFIG_H

#if (defined(__x86_64__) || defined(_M_X64))
#define AWS_ARCH_INTEL_X64
#define AWS_ARCH_INTEL
#define AWS_USE_CPU_EXTENSIONS
#elif (defined(__i386__) || defined(_M_IX86))
#define AWS_ARCH_INTEL
#define AWS_USE_CPU_EXTENSIONS
#endif  // __x86_64__

#if (defined(__aarch64__) || defined(_M_ARM64))
#define AWS_ARCH_ARM64
#define AWS_HAVE_ARM32_CRC
#define AWS_HAVE_ARMv8_1
#define AWS_USE_CPU_EXTENSIONS
#endif  // __aarch64__

#if defined(__GNUC__) || defined(__GNUG__)
#define AWS_HAVE_GCC_OVERFLOW_MATH_EXTENSIONS
#define AWS_HAVE_GCC_INLINE_ASM

#if defined(AWS_ARCH_INTEL_X64)
#define AWS_HAVE_AVX2_INTRINSICS
#define AWS_HAVE_AVX512_INTRINSICS
#define AWS_HAVE_MM256_EXTRACT_EPI64
#define AWS_HAVE_CLMUL
#endif  // AWS_ARCH_INTEL_X64

#elif defined(_MSC_VER)
#if defined(AWS_ARCH_INTEL_X64)
#define AWS_HAVE_MSVC_INTRINSICS_X64
#endif
#endif


#if defined(__APPLE__)
#define AWS_HAVE_POSIX_LARGE_FILE_SUPPORT
#define AWS_HAVE_EXECINFO
#elif defined(__linux__)
#define AWS_HAVE_POSIX_LARGE_FILE_SUPPORT
#define AWS_HAVE_EXECINFO
#define AWS_HAVE_LINUX_IF_LINK_H
#elif defined(_WIN32)
#define AWS_HAVE_WINAPI_DESKTOP
#endif

#endif  // AWS_COMMON_CONFIG_H
