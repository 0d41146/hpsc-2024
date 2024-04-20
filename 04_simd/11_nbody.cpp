#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

#define EPS 1e-9

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  for(int i=0; i<N; i++) {
    int j[] = {0, 1, 2, 3, 4, 5, 6, 7};
    __m512 ivec = _mm512_set1_ps(i);
    __m512 jvec = _mm512_load_ps(j);
    __mmask16 mask = _mm512_cmp_ps_mask(ivec, jvec, _MM_CMPINT_NE);
   
    __m512 xvec = _mm512_load_ps(x);
    __m512 xivec = _mm512_set1_ps(x[i]);
    __m512 yvec = _mm512_load_ps(y);
    __m512 yivec = _mm512_set1_ps(y[i]);
    //__m512 fxvec = _mm512_load_ps(fx); <- fy returns random value when i = 0. Why?
    __m512 fxvec = _mm512_setzero_ps();
    //__m512 fyvec = _mm512_load_ps(fy);
    __m512 fyvec = _mm512_setzero_ps();
    __m512 mvec = _mm512_load_ps(m);
  
    __m512 rxvec = _mm512_sub_ps(xivec, xvec);
    __m512 ryvec = _mm512_sub_ps(yivec, yvec);
    __m512 rx2vec = _mm512_mul_ps(rxvec, rxvec);
    __m512 ry2vec = _mm512_mul_ps(ryvec, ryvec);
    __m512 r2vec = _mm512_add_ps(rx2vec, ry2vec);

    __mmask16 eps_mask = _mm512_cmp_ps_mask(r2vec, _mm512_set1_ps(EPS), _CMP_LT_OQ);
    __m512 invrvec = _mm512_rsqrt14_ps(r2vec);
    invrvec = _mm512_mask_blend_ps(eps_mask, invrvec, _mm512_setzero_ps());

    __m512 inv3rvec = _mm512_mul_ps(_mm512_mul_ps(invrvec, invrvec), invrvec);

    __m512 fxvec_tmp = _mm512_mul_ps(_mm512_mul_ps(rxvec, mvec), inv3rvec);
    fxvec = _mm512_mask_blend_ps(mask & ~eps_mask, fxvec, fxvec_tmp);
  
    __m512 fyvec_tmp = _mm512_mul_ps(_mm512_mul_ps(ryvec, mvec), inv3rvec);
    fyvec = _mm512_mask_blend_ps(mask & ~eps_mask, fyvec, fyvec_tmp);
   
    fx[i] -= _mm512_reduce_add_ps(fxvec);
    fy[i] -= _mm512_reduce_add_ps(fyvec);
  
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
