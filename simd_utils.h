#include <xmmintrin.h>
#include <emmintrin.h>
#include <stdint.h>

/*
<mmintrin.h>  MMX
<xmmintrin.h> SSE
<emmintrin.h> SSE2
<pmmintrin.h> SSE3
<tmmintrin.h> SSSE3
<smmintrin.h> SSE4.1
<nmmintrin.h> SSE4.2
<ammintrin.h> SSE4A
<wmmintrin.h> AES
<immintrin.h> AVX, AVX2, FMA
*/

#ifdef _MSC_VER /* visual c++ */
	#define ALIGN16_BEG __declspec(align(16))
	#define ALIGN16_END
	#define inline	__forceinline
#else /* gcc or icc */
	# define ALIGN16_BEG
	# define ALIGN16_END __attribute__((aligned(16)))
#endif

// MMX - Header for MMX intrinsics
#ifdef __MMX__
    #include <mmintrin.h>
#endif

// SSE - Header for SSE intrinsics
#ifdef __SSE__
    #include <xmmintrin.h>
#endif

// SSE2 - Header for SSE2 intrinsics
#ifdef __SSE2__
    #define USE_SSE2
    #include <emmintrin.h>
#endif

// SSE3 - Header for SSE3 intrinsics
#ifdef __SSE3__
    #define USE_SSE2
    #define USE_SSE3
    #include <pmmintrin.h>
#endif

// SSSE3 - Header for SSSE3 intrinsics
#ifdef __SSSE3__
    #define USE_SSE3
    #include <tmmintrin.h>
#endif

// SSE4.1 - Header for SSE4.1 intrinsics
#ifdef __SSE4_1__
    #define USE_SSE4_1
    #include <smmintrin.h>
#endif

// SSE4.2 - Header for SSE4.2 intrinsics
#ifdef __SSE4_2__
    #define USE_SSE4_2
    #include <nmmintrin.h>
#endif

// SSE4A - Header for SSE4A intrinsics
#ifdef __SSE4A__
    #define USE_SSE4A
    #include <ammintrin.h>
#endif

// AES - Header for AES intrinsics
#ifdef __AES__
    #define USE_AES
    #include <wmmintrin.h>
#endif

// AVX - Header for AVX and AVX2 intrinsics
#ifdef __AVX__
    #define USE_AVX
    #include <immintrin.h>
#endif

// AVX2 - Header for AVX2 intrinsics
#ifdef __AVX2__
    #define USE_AVX2
    #include <immintrin.h>
#endif

// FMA - Header for FMA intrinsics
#ifdef __FMA__
    #define USE_FMA
    #include <immintrin.h>
#endif

// AVX512F - Header for AVX512 intrinsics
#ifdef __AVX512F__
    #define USE_AVX512
    #include <immintrin.h>
#endif

// Optimizations based on architecture
#if defined(__SSE3__) || defined(__SSSE3__)
    #define USE_SSE2
    #define USE_SSE3
#endif

#if defined(__SSE4__) || defined(__SSE4_1__) || defined(__SSE4_2__) || ((_M_IX86_FP > 1) && !defined(_M_AMD64))
    #define USE_SSE2
    #define USE_SSE3
    #define USE_SSE4
#endif

#if !defined(__FMA__) && defined(__AVX2__)
    #define __FMA__
#endif



#ifdef __AVX512F__
    #define PACKAGE_FLOAT __m512
	#define PACKAGE_INT __m512i
    #define PACKAGE_LENGTH 16
    #define PACKAGE_ALIGNMENT 64
    #define PACKAGE_SIZE 64
    #define _SETZERO() _mm512_setzero_ps()
	#define _SETZERO_SI() _mm512_setzero_si512()
    #define _SET1_PS(x) _mm512_set1_ps(x)
	#define _SET1_EPI32(x) _mm512_set1_epi32(x)
    #define _LOAD(x) _mm512_load_ps(x)
	#define _LOADU_SI(x) _mm512_loadu_si512(x)
    #define _STORE(x, y) _mm512_store_ps(x, y)
	#define _STOREU(x,y) _mm512_storeu_ps(x,y)
    #define _CMP(x, y, z) _mm512_cmp_ps_mask(x, y, z)
	#define _OR_PS(x,y) _mm512_or_ps(x,y)
	#define _CMPGT_PS(x, y) _mm512_cmp_ps_mask(x, y, _CMP_GT_OQ)
	#define _CMPLE_PS(x, y) _mm512_cmp_ps_mask(x, y, _CMP_LE_OQ)
	#define _CMPLT_PS(x, y) _mm512_cmp_ps_mask(x, y, _CMP_LT_OQ)
    #define _CMPGT_EPI32(x, y) _mm512_cmpgt_epi32_mask(x, y)
	#define _CMPEQ_EPI32(x, y) _mm512_cmpeq_epi32_mask(x, y)
    #define _BLEND(x, y, z) _mm512_mask_blend_ps(z, x, y)
    #define _BLENDV(x, y, z) _mm512_mask_blend_ps(_mm512_test_epi32_mask(z, z), x, y)
    #define _ADD_PS(x, y) _mm512_add_ps(x, y)
	#define _ADD_EPI32(x, y) _mm512_add_epi32(x, y)
    #define _SUB_PS(x, y) _mm512_sub_ps(x, y)
	#define _SUB_EPI32(x,y) _mm512_sub_epi32(x,y)
    #define _MUL(x, y) _mm512_mul_ps(x, y)
    #define _DIV(x, y) _mm512_div_ps(x, y)
#ifdef __AVX__
	#define _LOAD1(x) _mm512_broadcast_ss(x)
#else
	#define _LOAD1(x) load1_ps(x)
#endif
	#define _LOAD_SS(x) _mm512_load_ss(x)
#ifdef __FMA__
	#define _FMADD(x, y, z) _mm512_fmadd_ps(x, y, z)
#endif
	#define _MASKZ_MOV(x,y) _mm512_maskz_mov_ps(x,y)
	#define _MOVEMASK(x) _mm512_movm_epi64(x)
	#define _MOVEMASK_EPI8(x) _mm512_movm_epi8(x)
#ifdef __SSE4_1__
	#define _TESTZSI(x,y) _mm512_test_epi32_mask(x,y) == 0
#else
	#define _TESTZSI(x,y) testz_si(x,y)
	#define _BLEND(x,y,z) blend_ps(x,y,z)
#endif
	#define _CVT_PS_EPI32(x) _mm512_cvttps_epi32(x)
	#define _SLLI_EPI32(x,y) _mm512_slli_epi32(x,y)
	#define _CVT_EPI32_PS(x) _mm512_cvtepi32_ps(x)
	#define _CVT_EPI8_EPI32(x) _mm512_cvtepi8_epi32(x)
	#define _SRLI_EPI32(x,y) _mm512_srli_epi32(x,y)
	#define _MIN_PS(x,y) _mm512_min_ps(x,y)
	#define _MAX_PS(x,y) _mm512_max_ps(x,y)
#ifdef __INTEL_COMPILER
	#define _EXP(x) _mm512_exp_ps(x)
	#define _LOG(x) _mm512_log_ps(x)
#else
	#define _LOG(x) log_ps(x)
	#define _EXP(x) exp_ps(x) // Use a custom implementation or approximation
#endif

#elif defined __AVX__
	#define PACKAGE_FLOAT __m256
	#define PACKAGE_INT __m256i
	#define PACKAGE_LENGTH 8
	#define PACKAGE_ALIGNEMENT 32
	#define PACKAGE_SIZE 32
	#define _SETZERO() _mm256_setzero_ps()
	#define _SETZERO_SI() _mm256_setzero_si256()
	#define _SET1_PS(x) _mm256_set1_ps(x)
	#define _SET1_EPI32(x) _mm256_set1_epi32(x)
#ifdef __AVX__
	#define _LOAD1(x) _mm256_broadcast_ss(x)
	#define _SHUFFLE(x,y,z) _mm256_shuffle_ps(x,y,z)
	#define _CMP(x,y,z) _mm256_cmp_ps(x,y,z)
	#define _MASKSTORE(x,y,z) _mm256_maskstore_ps(x,y,z)
	#define _SLLI_EPI32(x,y) _mm256_slli_epi32(x,y)
#else
	#define _MASKSTORE(x,y,z) maskstore_ps(x,y,z)
	#define _LOAD1(x) load1_ps(x)
#endif
	#define _LOAD_SS(x) _mm256_load_ss(x)
	#define _CMPGT_PS(x, y) _mm256_cmp_ps(x, y, _CMP_GT_OQ)
	#define _CMPLE_PS(x, y) _mm256_cmp_ps(x, y, _CMP_LE_OQ)
	#define _CMPLT_PS(x, y) _mm256_cmp_ps(x, y, _CMP_LT_OQ)
	#define _CMPGT_EPI32(x,y) _mm256_cmpgt_epi32(x,y)
	#define _CMPEQ_EPI32(x, y) _mm256_cmpeq_epi32(x, y)
	#define _ADD_PS(x,y) _mm256_add_ps(x,y)
	#define _ADD_EPI32(x, y) _mm256_add_epi32(x, y)
	#define _SUB_PS(x,y) _mm256_sub_ps(x,y)
	#define _SUB_EPI32(x,y) _mm256_sub_epi32(x,y)
	#define _MUL(x,y) _mm256_mul_ps(x,y)
	#define _DIV(x,y) _mm256_div_ps(x,y)
	#define _LOAD(x) _mm256_load_ps(x)
	#define _LOADU_SI(x) _mm256_loadu_si256(x)
	#define _STORE(x,y) _mm256_store_ps(x,y)
	#define _STOREU(x,y) _mm256_storeu_ps(x,y)
#ifdef __FMA__
	#define _FMADD(x,y,z) _mm256_fmadd_ps(x,y,z)
	#define _FNMADD(x,y,z) _mm256_fnmadd_ps(x,y,z)
#endif
	#define _FMSUB(x,y,z) _mm256_fmsub_ps(x,y,z)
	#define _MAX(x,y) _mm256_max_ps(x,y)
	#define _CASTSI_PS(x) _mm256_castsi256_ps(x)
	#define _CASTPS_SI(x) _mm256_castps_si256(x)
	#define _SET1_EPI32(x) _mm256_set1_epi32(x)
	#define _AND(x,y) _mm256_and_ps(x,y)
	#define _ANDSI(x,y) _mm256_and_si256(x,y)
	#define _ANDNOT(x,y) _mm256_andnot_ps(x,y)
	#define _OR_PS(x,y) _mm256_or_ps(x,y)
	#define _MASKLOAD(x,y) _mm256_maskload_ps(x,y)
	#define _RCP(x) _mm256_rcp_ps(x)
	#define _MOVEMASK(x) _mm256_movemask_ps(x)
	#define _MOVEMASK_EPI8(x) _mm256_movemask_epi8(x)
	#define _SET(a,b,c,d,e,f,g,h) _mm256_set_ps(a,b,c,d,e,f,g,h)
#ifdef __SSE4_1__
	#define _BLEND(x,y,z) _mm256_blend_ps(x,y,z)
	#define _BLENDV(x,y,z) _mm256_blendv_ps(x,y,z)
	#define _TESTZSI(x,y) _mm256_testz_si256(x,y)
#else
	#define _TESTZSI(x,y) testz_si(x,y)
	#define _BLEND(x,y,z) blend_ps(x,y,z)
#endif
	#define _SRLI_EPI32(x,y) _mm256_srli_epi32(x,y)
	#define _CVT_PS_EPI32(x) _mm256_cvttps_epi32(x)
	#define _CVT_EPI32_PS(x) _mm256_cvtepi32_ps(x)
	#define _CVT_EPI8_EPI32(x) _mm256_cvtepi8_epi32(x)
	#define _MIN_PS(x,y) _mm256_min_ps(x,y)
	#define _MAX_PS(x,y) _mm256_max_ps(x,y)
#ifdef __INTEL_COMPILER
	#define _EXP(x) _mm256_exp_ps(x)
	#define _LOG(x) _mm256_log_ps(x)
#else
	#define _EXP(x) exp_ps(x)
	#define _LOG(x) log_ps(x)
#endif

#else
	#define PACKAGE_FLOAT __m128
	#define PACKAGE_INT __m128i
	#define PACKAGE_LENGTH 4
	#define PACKAGE_ALIGNEMENT 16
	#define PACKAGE_SIZE 16
	#define _SETZERO() _mm_setzero_ps()
	#define _SETZERO_SI() _mm_setzero_si128()
	#define _SET1_PS(x) _mm_set1_ps(x)
	#define _SET1_EPI32(x) _mm_set1_epi32(x)
#ifdef __AVX__
	#define _LOAD1(x) _mm_broadcast_ss(x)
	#define _CMP(x,y,z) _mm_cmp_ps(x,y,z)
	#define _MASKSTORE(x,y,z) _mm_maskstore_ps(x,y,z)
#else
	#define _MASKSTORE(x,y,z) maskstore_ps(x,y,z)
	#define _LOAD1(x) load1_ps(x)
#endif
	#define _SHUFFLE(x,y,z) _mm_shuffle_ps(x,y,z)
	#define _LOAD_SS(x) _mm_load_ss(x)
	#define _CMPGT_PS(x, y) _mm_cmpgt_ps(x, y)
	#define _CMPLE_PS(x, y) _mm_cmple_ps(x, y)
	#define _CMPLT_PS(x, y) _mm_cmplt_ps(x, y)
	#define _CMPGT_EPI32(x,y) _mm_cmpgt_epi32(x,y)
	#define _CMPEQ_EPI32(x, y) _mm_cmpeq_epi32(x, y)
	#define _ADD_PS(x,y) _mm_add_ps(x,y)
	#define _ADD_EPI32(x, y) _mm_add_epi32(x, y)
	#define _SUB_PS(x,y) _mm_sub_ps(x,y)
	#define _SUB_EPI32(x,y) _mm_sub_epi32(x,y)
	#define _MUL(x,y) _mm_mul_ps(x,y)
	#define _DIV(x,y) _mm_div_ps(x,y)
	#define _LOAD(x) _mm_load_ps(x)
	#define _LOADU_SI(x) _mm_loadu_si128(x)
	#define _STORE(x,y) _mm_store_ps(x,y)
	#define _STOREU(x,y) _mm_storeu_ps(x,y)
#ifdef __FMA__
	#define _FMADD(x,y,z) _mm_fmadd_ps(x,y,z)
	#define _FNMADD(x,y,z) _mm_fnmadd_ps(x,y,z)
#endif
	#define _FMSUB(x,y,z) _mm_fmsub_ps(x,y,z)
	#define _CASTSI_PS(x) _mm_castsi128_ps(x)
	#define _CASTPS_SI(x) _mm_castps_si128(x)
	#define _SET1_EPI32(x) _mm_set1_epi32(x)
	#define _AND(x,y) _mm_and_ps(x,y)
	#define _ANDNOT(x,y) _mm_andnot_ps(x,y)
	#define _ANDSI(x,y) _mm_and_si128(x,y)
	#define _OR_PS(x,y) _mm_or_ps(x,y)
	#define _RCP(x) _mm_rcp_ps(x)
	#define _MOVEMASK(x) _mm_movemask_ps(x)
	#define _MOVEMASK_EPI8(x) _mm_movemask_epi8(x)
	#define _MASKLOAD(x,y) _mm_maskload_ps(x,y)
	#define _SET(a,b,c,d) _mm_set_ps(a,b,c,d)
#ifdef __SSE4_1__
	#define _TESTZSI(x,y) _mm_testz_si128(x,y)
	#define _BLEND(x,y,z) _mm_blend_ps(x,y,z)
	#define _BLENDV(x,y,z) _mm_blendv_ps(x,y,z)
#else
	#define _TESTZSI(x,y) testz_si(x,y)
	#define _BLEND(x,y,z) blend_ps(x,y,z)
#endif
	#define _SRLI_EPI32(x,y) _mm_srli_epi32(x,y)
	#define _SLLI_EPI32(x,y) _mm_slli_epi32(x,y)
	#define _CVT_PS_EPI32(x) _mm_cvttps_epi32(x)
	#define _CVT_EPI8_EPI32(x) _mm_cvtepi8_epi32(x)
	#define _CVT_EPI32_PS(x) _mm_cvtepi32_ps(x)
	#define _MIN_PS(x,y) _mm_min_ps(x,y)
	#define _MAX_PS(x,y) _mm_max_ps(x,y)

#ifdef __INTEL_COMPILER
	#define _EXP(x) _mm_exp_ps(x)
	#define _LOG(x) _mm_log_ps(x)
#else
	#define _EXP(x) exp_ps(x)
	#define _LOG(x) log_ps(x)
#endif

	
#endif

// Unified function to generate random values
PACKAGE_FLOAT exp_ps(PACKAGE_FLOAT x);
PACKAGE_FLOAT log_ps(PACKAGE_FLOAT x);
template <typename Generator, typename Distribution>
PACKAGE_FLOAT _RAND(Generator& generator, Distribution& distribution);
float _HSUM(PACKAGE_FLOAT a);
float _HMAX(PACKAGE_FLOAT a);
void _TRANSPOSE(PACKAGE_FLOAT (&rows)[PACKAGE_LENGTH]);

template<uint16_t offset>
PACKAGE_INT remainderMaskSI();

template<uint16_t offset>
PACKAGE_FLOAT remainderMask();

int testz_si(PACKAGE_INT a, PACKAGE_INT b);
PACKAGE_FLOAT blend_ps(PACKAGE_FLOAT a, PACKAGE_FLOAT b, const int imm8);
PACKAGE_FLOAT load1_ps(const float* p);

void maskstore_ps(float *a, PACKAGE_INT mask, PACKAGE_FLOAT b);