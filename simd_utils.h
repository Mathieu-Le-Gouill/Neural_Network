#pragma once

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
	#define _MASKZ_MOV(x,y) _mm512_maskz_mov_ps(x,y)

#elif defined PACKAGE_M256
	#define PACKAGE_TYPE __m256
	#define PACKAGE_LENGTH 8
	#define PACKAGE_ALIGNEMENT 32
	#define PACKAGE_SIZE 32
	#define _SETZERO() _mm256_setzero_ps()
	#define _SET1(x) _mm256_set1_ps(x)
	#define _LOAD1(x) _mm256_broadcast_ss(x)
	#define _CMP(x,y,z) _mm256_cmp_ps(x,y,z)
	#define _CMPGT_EPI32(x,y) _mm256_cmpgt_epi32(x,y)
	#define _BLEND(x,y,z) _mm256_blend_ps(x,y,z)
	#define _BLENDV(x,y,z) _mm256_blendv_ps(x,y,z)
	#define _EXP(x) _mm256_exp_ps(x)
	#define _ADD(x,y) _mm256_add_ps(x,y)
	#define _SUB(x,y) _mm256_sub_ps(x,y)
	#define _MUL(x,y) _mm256_mul_ps(x,y)
	#define _DIV(x,y) _mm256_div_ps(x,y)
	#define _LOAD(x) _mm256_loadu_ps(x)
	#define _STORE(x,y) _mm256_storeu_ps(x,y)
	#define _FMADD(x,y,z) _mm256_fmadd_ps(x,y,z)
	#define _FNMADD(x,y,z) _mm256_fnmadd_ps(x,y,z)
	#define _FMSUB(x,y,z) _mm256_fmsub_ps(x,y,z)
	#define _MAX(x,y) _mm256_max_ps(x,y)
	#define _CASTSI_PS(x) _mm256_castsi256_ps(x)
	#define _CASTPS_SI(x) _mm256_castps_si256(x)
	#define _SET1_EPI32(x) _mm256_set1_epi32(x)
	#define _AND(x,y) _mm256_and_ps(x,y)
	#define _ANDSI(x,y) _mm256_and_si256(x,y)
	#define _ANDNOT(x,y) _mm256_andnot_ps(x,y)
	#define _MASKLOAD(x,y) _mm256_maskload_ps(x,y)
	#define _RCP(x) _mm256_rcp_ps(x)
	#define _MASKSTORE(x,y,z) _mm256_maskstore_ps(x,y,z)
	#define _LOG(x) _mm256_log_ps(x)

#else
	#define PACKAGE_TYPE __m128
	#define PACKAGE_LENGTH 4
	#define PACKAGE_ALIGNEMENT 16
	#define PACKAGE_SIZE 16
	#define _SETZERO() _mm_setzero_ps()
	#define _SET1(x) _mm_set1_ps(x)
	#define _LOAD1(x) _mm_broadcast_ss(x)
	#define _CMP(x,y,z) _mm_cmp_ps(x,y,z)
	#define _BLEND(x,y,z) _mm_blend_ps(x,y,z)
	#define _BLENDV(x,y,z) _mm_blendv_ps(x,y,z)
	#define _EXP(x) _mm_exp_ps(x)
	#define _ADD(x,y) _mm_add_ps(x,y)
	#define _SUB(x,y) _mm_sub_ps(x,y)
	#define _MUL(x,y) _mm_mul_ps(x,y)
	#define _DIV(x,y) _mm_div_ps(x,y)
	#define _LOAD(x) _mm_loadu_ps(x)
	#define _STORE(x,y) _mm_storeu_ps(x,y)
	#define _FMADD(x,y,z) _mm_fmadd_ps(x,y,z)
	#define _FNMADD(x,y,z) _mm_fnmadd_ps(x,y,z)
	#define _FMSUB(x,y,z) _mm_fmsub_ps(x,y,z)
	#define _MAX(x,y) _mm_max_ps(x,y)
	#define _CASTSI(x) _mm_castsi128_ps(x)
	#define _SET1_EPI32(x) _mm_set1_epi32(x)
	#define _AND(x,y) _mm_and_ps(x,y)
	#define _ANDNOT(x,y) _mm_andnot_ps(x,y)
	#define _ANDSI(x,y) _mm_and_si128(x,y)
	#define _RCP(x) _mm_rcp_ps(x)
	#define _MASKLOAD(x,y) _mm_maskload_ps(x,y)
	#define _MASKSTORE(x,y) _mm_maskstore_ps(x,y)
	#define _LOG(x) _mm_log_ps(x)
#endif

// Compute the sum of a vector of 4 floats
// Ref : https://stackoverflow.com/a/35270026/22910685
inline float horizontal_sum4(__m128 v) {
	__m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0
	__m128 sums = _mm_add_ps(v, shuf);
	shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
	sums = _mm_add_ss(sums, shuf);
	return        _mm_cvtss_f32(sums);
}


// Compute the sum of a vector of 8 floats with the first element of the returned vector containing the sum
// Ref : https://stackoverflow.com/a/40514255/22910685
inline __m256 hsums(__m256 v)
{
	auto x = _mm256_permute2f128_ps(v, v, 1);
	auto y = _mm256_add_ps(v, x);
	x = _mm256_shuffle_ps(y, y, _MM_SHUFFLE(2, 3, 0, 1));
	x = _mm256_add_ps(x, y);
	y = _mm256_shuffle_ps(x, x, _MM_SHUFFLE(1, 0, 3, 2));
	return _mm256_add_ps(x, y);
}


// Compute the sum of a vector of 8 floats
// Ref : https://stackoverflow.com/a/40514255/22910685
inline float horizontal_sum8(__m256 v)
{
	return _mm_cvtss_f32(_mm256_castps256_ps128(hsums(v)));
}


// Compute the transpose of a 8x8 matrix
// Ref : https://stackoverflow.com/a/25627536/22910685
inline void transpose8(__m256& row0, __m256& row1, __m256& row2, __m256& row3,
					   __m256& row4, __m256& row5, __m256& row6, __m256& row7)
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


// Find the maximum value in a vector of 8 floats
// Ref : https://stackoverflow.com/a/73439472/22910685
inline float horizontal_max8(__m256 v1)
{
	const __m256 v2 = _mm256_permute_ps(v1, 0b10'11'00'01);
	const __m256 v3 = _mm256_max_ps(v1, v2);
	const __m256 v4 = _mm256_permute_ps(v3, 0b00'00'10'10);
	const __m256 v5 = _mm256_max_ps(v3, v4);
	const __m128 v6 = _mm256_extractf128_ps(v5, 1);
	const __m128 v7 = _mm_max_ps(_mm256_castps256_ps128(v5), v6);

	return _mm_cvtss_f32(v7);
}