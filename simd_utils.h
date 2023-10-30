#include <immintrin.h>


#ifdef __AVX512F__
	#define PACKAGE_M512

#elif defined __AVX__
	#define PACKAGE_M256

#else
	#define PACKAGE_M128
#endif

// TODO PACKAGE_M512
#ifdef PACKAGE_M512
	#define PACKAGE_TYPE __m512

#elif defined PACKAGE_M256
	#define PACKAGE_TYPE __m256
	#define PACKAGE_LENGTH 8
	#define PACKAGE_ALIGNEMENT 32
	#define PACKAGE_SIZE 32
	#define _SETZERO() _mm256_setzero_ps()
	#define _SET1(x) _mm256_set1_ps(x)
	#define _LOAD1(x) _mm256_broadcast_ss(x)
	#define _CMP(x,y,z) _mm256_cmp_ps(x,y,z)
	#define _BLENDV(x,y,z) _mm256_blendv_ps(x,y,z)
	#define _EXP(x) _mm256_exp_ps(x)
	#define _ADD(x,y) _mm256_add_ps(x,y)
	#define _SUB(x,y) _mm256_sub_ps(x,y)
	#define _MUL(x,y) _mm256_mul_ps(x,y)
	#define _DIV(x,y) _mm256_div_ps(x,y)
	#define _LOAD(x) _mm256_load_ps(x)
	#define _STORE(x,y) _mm256_store_ps(x,y)
	#define _FMADD(x,y,z) _mm256_fmadd_ps(x,y,z)
	#define _MAX(x,y) _mm256_max_ps(x,y)
	#define _MASKZ_MOV(x,y) _mm256_maskz_mov_ps(x,y)
	#define _CASTSI(x) _mm256_castsi256_ps(x)
	#define _SET1_EPI32(x) _mm256_set1_epi32(x)
	#define _AND(x,y) _mm256_and_ps(x,y)
	#define _MASKLOAD(x,y) _mm256_maskload_ps(x,y)
	#define _MASKSTORE(x,y,z) _mm256_maskstore_ps(x,y,z)

#else
	#define PACKAGE_TYPE __m128
	#define PACKAGE_LENGTH 4
	#define PACKAGE_ALIGNEMENT 16
	#define PACKAGE_SIZE 16
	#define _SETZERO() _mm_setzero_ps()
	#define _SET1(x) _mm_set1_ps(x)
	#define _LOAD1(x) _mm_broadcast_ss(x)
	#define _CMP(x,y,z) _mm_cmp_ps(x,y,z)
	#define _BLENDV(x,y,z) _mm_blendv_ps(x,y,z)
	#define _EXP(x) _mm_exp_ps(x)
	#define _ADD(x,y) _mm_add_ps(x,y)
	#define _SUB(x,y) _mm_sub_ps(x,y)
	#define _MUL(x,y) _mm_mul_ps(x,y)
	#define _DIV(x,y) _mm_div_ps(x,y)
	#define _LOAD(x) _mm_load_ps(x)
	#define _STORE(x,y) _mm_store_ps(x,y)
	#define _FMADD(x,y,z) _mm_fmadd_ps(x,y,z)
	#define _MAX(x,y) _mm_max_ps(x,y)
	#define _MASKZ_MOV(x,y) _mm_maskz_mov_ps(x,y)
	#define _CASTSI(x) _mm_castsi128_ps(x)
	#define _SET1_EPI32(x) _mm_set1_epi32(x)
	#define _AND(x,y) _mm_and_ps(x,y)
	#define _MASKLOAD(x,y) _mm_maskload_ps(x,y)
	#define _MASKSTORE(x,y) _mm_maskstore_ps(x,y)
#endif



inline float horizontal_sum8(__m256 x) {
	// hiQuad = ( x7, x6, x5, x4 )
	const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
	// loQuad = ( x3, x2, x1, x0 )
	const __m128 loQuad = _mm256_castps256_ps128(x);
	// sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
	const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
	// loDual = ( -, -, x1 + x5, x0 + x4 )
	const __m128 loDual = sumQuad;
	// hiDual = ( -, -, x3 + x7, x2 + x6 )
	const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
	// sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
	const __m128 sumDual = _mm_add_ps(loDual, hiDual);
	// lo = ( -, -, -, x0 + x2 + x4 + x6 )
	const __m128 lo = sumDual;
	// hi = ( -, -, -, x1 + x3 + x5 + x7 )
	const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
	// sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
	const __m128 sum = _mm_add_ss(lo, hi);

	return _mm_cvtss_f32(sum);
}


inline void transpose8(__m256& row0, __m256& row1, __m256& row2, __m256& row3, __m256& row4, __m256& row5, __m256& row6, __m256& row7)
{
	__m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
	__m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;

	__t0 = _mm256_unpacklo_ps(row0, row1);
	__t1 = _mm256_unpackhi_ps(row0, row1);
	__t2 = _mm256_unpacklo_ps(row2, row3);
	__t3 = _mm256_unpackhi_ps(row2, row3);
	__t4 = _mm256_unpacklo_ps(row4, row5);
	__t5 = _mm256_unpackhi_ps(row4, row5);
	__t6 = _mm256_unpacklo_ps(row6, row7);
	__t7 = _mm256_unpackhi_ps(row6, row7);

	__tt0 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0));
	__tt1 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2));
	__tt2 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0));
	__tt3 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2));
	__tt4 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(1, 0, 1, 0));
	__tt5 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(3, 2, 3, 2));
	__tt6 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(1, 0, 1, 0));
	__tt7 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(3, 2, 3, 2));

	row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
	row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
	row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
	row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
	row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
	row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
	row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
	row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}