#include "simd_utils.h"
#include <cstdint>

static const PACKAGE_FLOAT _ps_exp_hi = _SET1_PS(88.3762626647949f);
static const PACKAGE_FLOAT _ps_exp_lo = _SET1_PS(-88.3762626647949f);
static const PACKAGE_FLOAT _ps_cephes_LOG2EF = _SET1_PS(1.44269504088896341f);
static const PACKAGE_FLOAT _ps_cephes_exp_C1 = _SET1_PS(0.693359375f);
static const PACKAGE_FLOAT _ps_cephes_exp_C2 = _SET1_PS(-2.12194440e-4f);
static const PACKAGE_FLOAT _ps_cephes_exp_p0 = _SET1_PS(1.9875691500E-4f);
static const PACKAGE_FLOAT _ps_cephes_exp_p1 = _SET1_PS(1.3981999507E-3f);
static const PACKAGE_FLOAT _ps_cephes_exp_p2 = _SET1_PS(8.3334519073E-3f);
static const PACKAGE_FLOAT _ps_cephes_exp_p3 = _SET1_PS(4.1665795894E-2f);
static const PACKAGE_FLOAT _ps_cephes_exp_p4 = _SET1_PS(1.6666665459E-1f);
static const PACKAGE_FLOAT _ps_cephes_exp_p5 = _SET1_PS(5.0000001201E-1f);
static const PACKAGE_FLOAT _ps_0p5 = _SET1_PS(0.5f);
static const PACKAGE_INT _pi32_0x7f = _SET1_EPI32(0x7f);

inline PACKAGE_FLOAT exp_ps(PACKAGE_FLOAT x) {
    PACKAGE_FLOAT tmp = _SETZERO(), fx, mask, y, z;
    PACKAGE_FLOAT pow2n;
    #ifdef USE_SSE2
    PACKAGE_INT emm0;
    #else
    PACKAGE_INT mm0, mm1;
    #endif
    PACKAGE_FLOAT one = _SET1_PS(1.0f);

    x = _MIN_PS(x, _ps_exp_hi);
    x = _MAX_PS(x, _ps_exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    fx = _MUL(x, _ps_cephes_LOG2EF);
    fx = _ADD_PS(fx, _ps_0p5);

    /* how to perform a floorf with SSE: just below */
    #ifndef USE_SSE2
        /* step 1 : cast to int */
        tmp = _mm_movehl_ps(tmp, fx);
        mm0 = _mm_cvttps_pi32(fx);
        mm1 = _mm_cvttps_pi32(tmp);
        /* step 2 : cast back to float */
        tmp = _mm_cvtpi32x2_ps(mm0, mm1);
    #else
        emm0 = _CVT_PS_EPI32(fx);
        tmp  = _CVT_EPI32_PS(emm0);
    #endif
    /* if greater, subtract 1 */
    mask = _CMPGT_PS(tmp, fx);
    mask = _AND(mask, one);
    fx = _SUB_PS(tmp, mask);

    tmp = _MUL(fx, _ps_cephes_exp_C1);
    z = _MUL(fx, _ps_cephes_exp_C2);
    x = _SUB_PS(x, tmp);
    x = _SUB_PS(x, z);

    z = _MUL(x, x);

    y = _ps_cephes_exp_p0;
    y = _MUL(y, x);
    y = _ADD_PS(y, _ps_cephes_exp_p1);
    y = _MUL(y, x);
    y = _ADD_PS(y, _ps_cephes_exp_p2);
    y = _MUL(y, x);
    y = _ADD_PS(y, _ps_cephes_exp_p3);
    y = _MUL(y, x);
    y = _ADD_PS(y, _ps_cephes_exp_p4);
    y = _MUL(y, x);
    y = _ADD_PS(y, _ps_cephes_exp_p5);
    y = _MUL(y, z);
    y = _ADD_PS(y, x);
    y = _ADD_PS(y, one);

    /* build 2^n */
    #ifndef USE_SSE2
        z = _mm_movehl_ps(z, fx);
        mm0 = _mm_cvttps_pi32(fx);
        mm1 = _mm_cvttps_pi32(z);
        mm0 = _mm_add_pi32(mm0, *(__m64*)_pi32_0x7f);
        mm1 = _mm_add_pi32(mm1, *(__m64*)_pi32_0x7f);
        mm0 = _mm_slli_pi32(mm0, 23);
        mm1 = _mm_slli_pi32(mm1, 23);

        COPY_MM_TO_XMM(mm0, mm1, pow2n);
        _mm_empty();
    #else
        emm0 = _CVT_EPI32_PS(fx);
        emm0 = _ADD_EPI32(emm0, _pi32_0x7f);
        emm0 = _SLLI_EPI32(emm0, 23);
        pow2n = _CASTSI_PS(emm0);
    #endif

    y = _MUL(y, pow2n);
    return y;
}

static const PACKAGE_FLOAT _ps_min_norm_pos = _SET1_PS(1.17549435e-38f); // Smallest positive normal number
static const PACKAGE_INT _ps_inv_mant_mask = _SET1_EPI32(~0x7f800000); // Inverse of the mantissa mask
static const PACKAGE_FLOAT _ps_cephes_SQRTHF = _SET1_PS(0.707106781186547524f); // sqrt(0.5)
static const PACKAGE_FLOAT _ps_cephes_log_p0 = _SET1_PS(7.0376836292E-2f);
static const PACKAGE_FLOAT _ps_cephes_log_p1 = _SET1_PS(-1.1514610310E-1f);
static const PACKAGE_FLOAT _ps_cephes_log_p2 = _SET1_PS(1.1676998740E-1f);
static const PACKAGE_FLOAT _ps_cephes_log_p3 = _SET1_PS(-1.2420140846E-1f);
static const PACKAGE_FLOAT _ps_cephes_log_p4 = _SET1_PS(1.4249322787E-1f);
static const PACKAGE_FLOAT _ps_cephes_log_p5 = _SET1_PS(-1.6668057665E-1f);
static const PACKAGE_FLOAT _ps_cephes_log_p6 = _SET1_PS(2.0000714765E-1f);
static const PACKAGE_FLOAT _ps_cephes_log_p7 = _SET1_PS(-2.4999993993E-1f);
static const PACKAGE_FLOAT _ps_cephes_log_p8 = _SET1_PS(3.3333331174E-1f);
static const PACKAGE_FLOAT _ps_cephes_log_q1 = _SET1_PS(-2.12194440e-4f);
static const PACKAGE_FLOAT _ps_cephes_log_q2 = _SET1_PS(0.693359375f);

inline PACKAGE_FLOAT log_ps(PACKAGE_FLOAT x) {
    PACKAGE_FLOAT e;
    #ifdef USE_SSE2
    PACKAGE_INT emm0;
    #else
    PACKAGE_INT mm0, mm1;
    #endif
    PACKAGE_FLOAT one = _SET1_PS(1.0f);
    PACKAGE_FLOAT invalid_mask = _CMPLE_PS(x, _SETZERO());

    x = _MAX_PS(x, _ps_min_norm_pos);  /* cut off denormalized stuff */

    #ifndef USE_SSE2
        /* part 1: x = frexpf(x, &e); */
        COPY_XMM_TO_MM(x, mm0, mm1);
        mm0 = _mm_srli_pi32(mm0, 23);
        mm1 = _mm_srli_pi32(mm1, 23);
    #else
        emm0 = _SRLI_EPI32(_CASTPS_SI(x), 23);
    #endif
    /* keep only the fractional part */
    x = _AND(x, _ps_inv_mant_mask);
    x = _OR_PS(x, _ps_0p5);

 #ifndef USE_SSE2
        /* now e=mm0:mm1 contain the really base-2 exponent */
        mm0 = _mm_sub_pi32(mm0, _pi32_0x7f);
        mm1 = _mm_sub_pi32(mm1, _pi32_0x7f);
        e = _mm_cvtpi32x2_ps(mm0, mm1);
        _mm_empty(); /* bye bye mmx */
    #else
        emm0 = _SUB_EPI32(emm0, _pi32_0x7f);
        e = _CVT_EPI32_PS(emm0);
    #endif

    e = _ADD_PS(e, one);

    /* part2:
       if( x < SQRTHF ) {
         e -= 1;
         x = x + x - 1.0;
       } else { x = x - 1.0; }
    */
    {
        PACKAGE_FLOAT z, y;
        PACKAGE_FLOAT mask = _CMPLT_PS(x, _ps_cephes_SQRTHF);
        PACKAGE_FLOAT tmp = _AND(x, mask);
        x = _SUB_PS(x, one);
        e = _SUB_PS(e, _AND(one, mask));
        x = _ADD_PS(x, tmp);

        z = _MUL(x, x);

        y = _ps_cephes_log_p0;
        y = _MUL(y, x);
        y = _ADD_PS(y, _ps_cephes_log_p1);
        y = _MUL(y, x);
        y = _ADD_PS(y, _ps_cephes_log_p2);
        y = _MUL(y, x);
        y = _ADD_PS(y, _ps_cephes_log_p3);
        y = _MUL(y, x);
        y = _ADD_PS(y, _ps_cephes_log_p4);
        y = _MUL(y, x);
	    y = _ADD_PS(y, _ps_cephes_log_p5);
        y = _MUL(y, x);
        y = _ADD_PS(y, _ps_cephes_log_p6);
        y = _MUL(y, x);
        y = _ADD_PS(y, _ps_cephes_log_p7);
        y = _MUL(y, x);
        y = _ADD_PS(y, _ps_cephes_log_p8);
        y = _MUL(y, x);

        y = _MUL(y, z);

        tmp = _MUL(e, _ps_cephes_log_q1);
        y = _ADD_PS(y, tmp);

        tmp = _MUL(z, _ps_0p5);
        y = _SUB_PS(y, tmp);

        tmp = _MUL(e, _ps_cephes_log_q2);
        x = _ADD_PS(x, y);
        x = _ADD_PS(x, tmp);
        x = _OR_PS(x, invalid_mask); // negative arg will be NAN
    }
    return x;
}




#ifdef __AVX512F__

// Unified function to generate random values
template <typename Generator,typename Distribution>
inline PACKAGE_TYPE _RAND(Generator& generator, Distribution& distribution) 
{
	return _SET(distribution(generator), distribution(generator), distribution(generator), distribution(generator),
                distribution(generator), distribution(generator), distribution(generator), distribution(generator),
                distribution(generator), distribution(generator), distribution(generator), distribution(generator),
                distribution(generator), distribution(generator), distribution(generator), distribution(generator));
}

// Compute the sum of a vector of 16 floats
// Ref : https://stackoverflow.com/a/26905830
inline float _HSUM(__m512 a) {
    __m256 low  = _mm512_castps512_ps256(zmm);
	__m256 high = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(zmm),1));

	__m256 lh = _mm256_add_ps(low,high);

	__m256 t1 = _mm256_hadd_ps(lh,lh);
    __m256 t2 = _mm256_hadd_ps(t1,t1);
    __m128 t3 = _mm256_extractf128_ps(t2,1);
    __m128 t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);

	return _mm_cvtss_f32(t4); 
}

// Get the max value of a vector of 16 floats
inline float _HMAX(__m512 a) {
    __m256 low  = _mm512_castps512_ps256(zmm);
	__m256 high = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(zmm),1));
    
    __m256 max256 = _mm256_max_ps(lower, upper);
    
    __m256 perm0 = _mm256_permute_ps(max256, 0b01001110);
    __m256 max1 = _mm256_max_ps(max256, perm0);
    
    __m256 perm1 = _mm256_permute_ps(max1, 0b10110001);
    __m256 max2 = _mm256_max_ps(max1, perm1);
    
    return _mm_cvtss_f32(max2);
}

inline void _TRANSPOSE(__m512 (&rows)[16]) {
    __m512 t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;
    __m512 r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;

    int64_t idx1[8] __attribute__((aligned(64))) = {2, 3, 0, 1, 6, 7, 4, 5}; 
    int64_t idx2[8] __attribute__((aligned(64))) = {1, 0, 3, 2, 5, 4, 7, 6}; 
    int32_t idx3[16] __attribute__((aligned(64))) = {1, 0, 3, 2, 5 ,4 ,7 ,6 ,9 ,8 , 11, 10, 13, 12 ,15, 14};
    
	__m512i vidx1 = _mm512_load_epi64(idx1);
    __m512i vidx2 = _mm512_load_epi64(idx2);
    __m512i vidx3 = _mm512_load_epi32(idx3);

    t0 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&rows[0])), _mm256_load_si256((__m256i*)&rows[8]), 1);
    t1 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&rows[1])), _mm256_load_si256((__m256i*)&rows[9]), 1);
    t2 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&rows[2])), _mm256_load_si256((__m256i*)&rows[10]), 1);
    t3 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&rows[3])), _mm256_load_si256((__m256i*)&rows[11]), 1);
    t4 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&rows[4])), _mm256_load_si256((__m256i*)&rows[12]), 1);
    t5 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&rows[5])), _mm256_load_si256((__m256i*)&rows[13]), 1);
    t6 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&rows[6])), _mm256_load_si256((__m256i*)&rows[14]), 1);
    t7 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&rows[7])), _mm256_load_si256((__m256i*)&rows[15]), 1);

    t8 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&rows[0 + 8])), _mm256_load_si256((__m256i*)&rows[8 + 8]), 1);
    t9 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&rows[1 + 8])), _mm256_load_si256((__m256i*)&rows[9 + 8]), 1);
    ta = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&rows[2 + 8])), _mm256_load_si256((__m256i*)&rows[10 + 8]), 1);
    tb = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&rows[3 + 8])), _mm256_load_si256((__m256i*)&rows[11 + 8]), 1);
    tc = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&rows[4 + 8])), _mm256_load_si256((__m256i*)&rows[12 + 8]), 1);
    td = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&rows[5 + 8])), _mm256_load_si256((__m256i*)&rows[13 + 8]), 1);
    te = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&rows[6 + 8])), _mm256_load_si256((__m256i*)&rows[14 + 8]), 1);
    tf = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&rows[7 + 8])), _mm256_load_si256((__m256i*)&rows[15 + 8]), 1);

    int mask = 0xcc;
    r0 = _mm512_mask_permutexvar_epi64(t0, (__mmask8)mask, vidx1, t4);
    r1 = _mm512_mask_permutexvar_epi64(t1, (__mmask8)mask, vidx1, t5);
    r2 = _mm512_mask_permutexvar_epi64(t2, (__mmask8)mask, vidx1, t6);
    r3 = _mm512_mask_permutexvar_epi64(t3, (__mmask8)mask, vidx1, t7);
    r8 = _mm512_mask_permutexvar_epi64(t8, (__mmask8)mask, vidx1, tc);
    r9 = _mm512_mask_permutexvar_epi64(t9, (__mmask8)mask, vidx1, td);
    ra = _mm512_mask_permutexvar_epi64(ta, (__mmask8)mask, vidx1, te);
    rb = _mm512_mask_permutexvar_epi64(tb, (__mmask8)mask, vidx1, tf);

    mask = 0x33;
    r4 = _mm512_mask_permutexvar_epi64(t4, (__mmask8)mask, vidx1, t0);
    r5 = _mm512_mask_permutexvar_epi64(t5, (__mmask8)mask, vidx1, t1);
    r6 = _mm512_mask_permutexvar_epi64(t6, (__mmask8)mask, vidx1, t2);
    r7 = _mm512_mask_permutexvar_epi64(t7, (__mmask8)mask, vidx1, t3);
    rc = _mm512_mask_permutexvar_epi64(tc, (__mmask8)mask, vidx1, t8);
    rd = _mm512_mask_permutexvar_epi64(td, (__mmask8)mask, vidx1, t9);
    re = _mm512_mask_permutexvar_epi64(te, (__mmask8)mask, vidx1, ta);
    rf = _mm512_mask_permutexvar_epi64(tf, (__mmask8)mask, vidx1, tb);

    mask = 0xaa;
    t0 = _mm512_mask_permutexvar_epi64(r0, (__mmask8)mask, vidx2, r2);
    t1 = _mm512_mask_permutexvar_epi64(r1, (__mmask8)mask, vidx2, r3);
    t4 = _mm512_mask_permutexvar_epi64(r4, (__mmask8)mask, vidx2, r6);
    t5 = _mm512_mask_permutexvar_epi64(r5, (__mmask8)mask, vidx2, r7);
    t8 = _mm512_mask_permutexvar_epi64(r8, (__mmask8)mask, vidx2, ra);
    t9 = _mm512_mask_permutexvar_epi64(r9, (__mmask8)mask, vidx2, rb);
    tc = _mm512_mask_permutexvar_epi64(rc, (__mmask8)mask, vidx2, re);
    td = _mm512_mask_permutexvar_epi64(rd, (__mmask8)mask, vidx2, rf);

    mask = 0x55;
    t2 = _mm512_mask_permutexvar_epi64(r2, (__mmask8)mask, vidx2, r0);
    t3 = _mm512_mask_permutexvar_epi64(r3, (__mmask8)mask, vidx2, r1);
    t6 = _mm512_mask_permutexvar_epi64(r6, (__mmask8)mask, vidx2, r4);
    t7 = _mm512_mask_permutexvar_epi64(r7, (__mmask8)mask, vidx2, r5);
    ta = _mm512_mask_permutexvar_epi64(ra, (__mmask8)mask, vidx2, r8);
    tb = _mm512_mask_permutexvar_epi64(rb, (__mmask8)mask, vidx2, r9);
    te = _mm512_mask_permutexvar_epi64(re, (__mmask8)mask, vidx2, rc);
    tf = _mm512_mask_permutexvar_epi64(rf, (__mmask8)mask, vidx2, rd);

    mask = 0xaaaa;
    r0 = _mm512_mask_permutexvar_epi32(t0, (__mmask16)mask, vidx3, t1);
    r2 = _mm512_mask_permutexvar_epi32(t2, (__mmask16)mask, vidx3, t3);
    r4 = _mm512_mask_permutexvar_epi32(t4, (__mmask16)mask, vidx3, t5);
    r6 = _mm512_mask_permutexvar_epi32(t6, (__mmask16)mask, vidx3, t7);
    r8 = _mm512_mask_permutexvar_epi32(t8, (__mmask16)mask, vidx3, t9);
    ra = _mm512_mask_permutexvar_epi32(ta, (__mmask16)mask, vidx3, tb);
    rc = _mm512_mask_permutexvar_epi32(tc, (__mmask16)mask, vidx3, td);
    re = _mm512_mask_permutexvar_epi32(te, (__mmask16)mask, vidx3, tf);    

    mask = 0x5555;
    r1 = _mm512_mask_permutexvar_epi32(t1, (__mmask16)mask, vidx3, t0);
    r3 = _mm512_mask_permutexvar_epi32(t3, (__mmask16)mask, vidx3, t2);
    r5 = _mm512_mask_permutexvar_epi32(t5, (__mmask16)mask, vidx3, t4);
    r7 = _mm512_mask_permutexvar_epi32(t7, (__mmask16)mask, vidx3, t6);
    r9 = _mm512_mask_permutexvar_epi32(t9, (__mmask16)mask, vidx3, t8);  
    rb = _mm512_mask_permutexvar_epi32(tb, (__mmask16)mask, vidx3, ta);  
    rd = _mm512_mask_permutexvar_epi32(td, (__mmask16)mask, vidx3, tc);
    rf = _mm512_mask_permutexvar_epi32(tf, (__mmask16)mask, vidx3, te);

    rows[0] = r0;
    rows[1] = r1;
    rows[2] = r2;
    rows[3] = r3;
    rows[4] = r4;
    rows[5] = r5;
    rows[6] = r6;
    rows[7] = r7;
    rows[8] = r8;
    rows[9] = r9;
    rows[10] = ra;
    rows[11] = rb;
    rows[12] = rc;
    rows[13] = rd;
    rows[14] = re;
    rows[15] = rf;
}

#elif defined __AVX__

// Unified function to generate random values
template <typename Generator,typename Distribution>
inline PACKAGE_FLOAT _RAND(Generator& generator, Distribution& distribution) 
{
	return _SET(distribution(generator), distribution(generator), distribution(generator), distribution(generator),
                distribution(generator), distribution(generator), distribution(generator), distribution(generator));
}

// Compute the sum of a vector of 8 floats
// Ref : https://stackoverflow.com/a/18616679
inline float _HSUM (__m256 a) {
    __m256 t1 = _mm256_hadd_ps(a,a);
    __m256 t2 = _mm256_hadd_ps(t1,t1);

    __m128 t3 = _mm256_extractf128_ps(t2,1);

    __m128 t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);

    return _mm_cvtss_f32(t4);        
}

// Get the max value of a vector of 8 floats
inline float _HMAX(__m256 a){
    const __m256 permHalves = _mm256_permute2f128_ps(a, a, 1);
    const __m256 m0 = _mm256_max_ps(permHalves, a);

    const __m256 perm0 = _mm256_permute_ps(m0, 0b01001110);
    const __m256 m1 = _mm256_max_ps(m0, perm0);

    const __m256 perm1 = _mm256_permute_ps(m1, 0b10110001);
    const __m256 m2 = _mm256_max_ps(perm1, m1);

    return _mm256_cvtss_f32(m2);
}

// Transpose a 8x8 matrix
// Ref : https://stackoverflow.com/a/25627536/22910685
inline void _TRANSPOSE(__m256 (&rows)[8]) {
	__m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
    __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
    __t0 = _mm256_unpacklo_ps(rows[0], rows[1]);
    __t1 = _mm256_unpackhi_ps(rows[0], rows[1]);
    __t2 = _mm256_unpacklo_ps(rows[2], rows[3]);
    __t3 = _mm256_unpackhi_ps(rows[2], rows[3]);
    __t4 = _mm256_unpacklo_ps(rows[4], rows[5]);
    __t5 = _mm256_unpackhi_ps(rows[4], rows[5]);
    __t6 = _mm256_unpacklo_ps(rows[6], rows[7]);
    __t7 = _mm256_unpackhi_ps(rows[6], rows[7]);
    __tt0 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0));
    __tt1 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2));
    __tt2 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0));
    __tt3 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2));
    __tt4 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(1, 0, 1, 0));
    __tt5 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(3, 2, 3, 2));
    __tt6 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(1, 0, 1, 0));
    __tt7 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(3, 2, 3, 2));
    rows[0] = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
    rows[1] = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
    rows[2] = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
    rows[3] = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
    rows[4] = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
    rows[5] = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
    rows[6] = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
    rows[7] = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}


#else

// Unified function to generate random values
template <typename Generator,typename Distribution>
inline PACKAGE_FLOAT _RAND(Generator& generator, Distribution& distribution) 
{
	return _SET(distribution(generator), distribution(generator), distribution(generator), distribution(generator));
}

// Compute the sum of a vector of 4 floats
// Ref : https://stackoverflow.com/a/35270026/22910685
inline float _HSUM(__m128 v) 
{
    #ifdef USE_SSE3
	__m128 shuf = _mm_movehdup_ps(v);
    #else
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 1, 1));
    #endif
	__m128 sums = _mm_add_ps(v, shuf);
	shuf = _mm_movehl_ps(shuf, sums);
	sums = _mm_add_ss(sums, shuf);
	return        _mm_cvtss_f32(sums);
}

// Get the max value of a vector of 4 floats
// Ref: https://stackoverflow.com/a/46126018
inline float _HMAX(__m128 x) {
    __m128 max1 = _mm_shuffle_ps(x, x, _MM_SHUFFLE(0,0,3,2));
    __m128 max2 = _mm_max_ps(x, max1);
    __m128 max3 = _mm_shuffle_ps(max2, max2, _MM_SHUFFLE(0,0,0,1));
    __m128 max4 = _mm_max_ps(max2, max3);
    return _mm_cvtss_f32(max4);
}

// Tranpose a 4x4 matrix
inline void _TRANSPOSE(__m128 (&rows)[4])
{
	_MM_TRANSPOSE4_PS(rows[0], rows[1], rows[2], rows[3]);
}

#endif

#ifdef __AVX2__

// Return a mask based on a given offset
// Only the offset values in the returned mask are set to 1, the others are set to 0
template<uint16_t offset>
inline PACKAGE_INT remainderMaskSI()
{
    if (offset == 0) {
        return _mm256_setzero_si256();
    }

    // Make a mask of 8 bytes
    // No need to clip for missingLanes <= 8 because the shift is already good, results in zero
    uint64_t mask = ~(uint64_t) 0;
    mask >>= (PACKAGE_LENGTH - offset) * 8;
    // Sign extend these bytes into int32 lanes in AVX vector
    __m128 tmp = _mm_cvtsi64_si128((int64_t)mask);

    #if defined(__AVX512F__) || defined(__AVX__)
        return _CVT_EPI8_EPI32(tmp);
    #else
        return tmp;
    #endif
}

#else
    #ifdef __AVX512F__
        alignas(128) static const int s_remainderLoadMask[32] = {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    #elif defined __AVX__
        alignas(64) static const int s_remainderLoadMask[16] = {
        -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0 };
    #else
        alignas(32) static const int s_remainderLoadMask[8] = {
        -1, -1, -1, -1, 0, 0, 0, 0 };
    #endif


    template<uint16_t offset>
    inline PACKAGE_INT remainderMaskSI()
    {
        // Unaligned load from a constant array
        const int* rsi = &s_remainderLoadMask[offset];
        return _LOADU_SI((const PACKAGE_INT*)rsi);
    }   

#endif

template<uint16_t offset>
inline PACKAGE_FLOAT remainderMask()
{
    return _CASTSI_PS(remainderMaskSI<offset>());
}


inline int testz_si(PACKAGE_INT a, PACKAGE_INT b)
{
    PACKAGE_INT and_result = _ANDSI(a, b);
    // Compare the result with zero
    PACKAGE_INT zero = _SETZERO_SI();
    PACKAGE_INT cmp_result = _CMPEQ_EPI32(and_result, zero);
    // Check if all elements are zero
    return _MOVEMASK_EPI8(cmp_result) == 0xFFFF;
}


inline PACKAGE_FLOAT blend_ps(PACKAGE_FLOAT a, PACKAGE_FLOAT b, const int imm8) {
    #ifdef __AVX512F__
        __m512 mask = _mm512_set_ps(
        (imm8 & 0x80000000) ? -1.0f : 0.0f,
        (imm8 & 0x40000000) ? -1.0f : 0.0f,
        (imm8 & 0x20000000) ? -1.0f : 0.0f,
        (imm8 & 0x10000000) ? -1.0f : 0.0f,
        (imm8 & 0x8000000) ? -1.0f : 0.0f,
        (imm8 & 0x4000000) ? -1.0f : 0.0f,
        (imm8 & 0x2000000) ? -1.0f : 0.0f,
        (imm8 & 0x1000000) ? -1.0f : 0.0f,
        (imm8 & 0x800000) ? -1.0f : 0.0f,
        (imm8 & 0x400000) ? -1.0f : 0.0f,
        (imm8 & 0x200000) ? -1.0f : 0.0f,
        (imm8 & 0x100000) ? -1.0f : 0.0f,
        (imm8 & 0x80000) ? -1.0f : 0.0f,
        (imm8 & 0x40000) ? -1.0f : 0.0f,
        (imm8 & 0x20000) ? -1.0f : 0.0f,
        (imm8 & 0x10000) ? -1.0f : 0.0f
    );
    #elif defined __AVX__
    __m256 mask = _mm256_set_ps(
        (imm8 & 0x80) ? -1.0f : 0.0f,
        (imm8 & 0x40) ? -1.0f : 0.0f,
        (imm8 & 0x20) ? -1.0f : 0.0f,
        (imm8 & 0x10) ? -1.0f : 0.0f,
        (imm8 & 0x8) ? -1.0f : 0.0f,
        (imm8 & 0x4) ? -1.0f : 0.0f,
        (imm8 & 0x2) ? -1.0f : 0.0f,
        (imm8 & 0x1) ? -1.0f : 0.0f
    );
    #else
    __m128 mask = _mm_set_ps(
        (imm8 & 0x8) ? -1.0f : 0.0f,
        (imm8 & 0x4) ? -1.0f : 0.0f,
        (imm8 & 0x2) ? -1.0f : 0.0f,
        (imm8 & 0x1) ? -1.0f : 0.0f
    );
    #endif
    
    PACKAGE_FLOAT not_mask = _ANDNOT(mask, a);
    PACKAGE_FLOAT masked_b = _AND(mask, b);
    return _OR_PS(not_mask, masked_b);
}

inline PACKAGE_FLOAT load1_ps(const float* p) {
    #ifdef __AVX__
        return _LOAD1(p);
    #else
        PACKAGE_FLOAT value = _LOAD_SS(p); // Load the single float value
        return _SHUFFLE(value, value, _MM_SHUFFLE(0, 0, 0, 0)); // Replicate the value across all elements
    #endif
}

inline void maskstore_ps(float *a, PACKAGE_INT mask, PACKAGE_FLOAT b) {
    // Load the current values from memory
    PACKAGE_FLOAT current = _LOAD(a);

    // Apply the mask
    PACKAGE_FLOAT masked_b = _AND(_CASTSI_PS(mask), b);
    PACKAGE_FLOAT masked_current = _ANDNOT(_CASTSI_PS(mask), current);

    // Combine the masked values
    PACKAGE_FLOAT result = _OR_PS(masked_b, masked_current);

    // Store the result back to memory
    _STORE(a, result);
}