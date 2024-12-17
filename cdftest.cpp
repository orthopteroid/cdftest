// cdftest.cpp : Defines the entry point for the application.
// (c 2024) orthopteroid@gmail.com, released under MIT license

#include <assert.h>
#include <stdint.h>
#include <chrono>

#include "cdftest.h"

#ifdef _MSVC_LANG

#include <intrin.h> // msvc
#include <immintrin.h>
#include <cstdlib>

#define BREAK do { __debugbreak(); }while(false)
#define ALIGNED16
#define szPlatform "win"

template<size_t A, class T>
T* aligned_malloc(size_t n)
{
  return static_cast<T*>(_aligned_malloc(n,A));
}

template<class T>
void aligned_free(T *p)
{
	_aligned_free(p);
}

#if defined(_M_IX86)
#define ARCH_X86
#define szArch "x86"
#else
#define ARCH_X64
#define szArch "x64"
#endif

#else // _MSVC_LANG

#include <x86intrin.h> // gcc
#include <mmintrin.h>

#define BREAK __asm("int3")
#define ALIGNED16 __attribute__((aligned(16)))
#define ALIGNED8 __attribute__((aligned(8)))
#define ALIGNED4 __attribute__((aligned(4)))
#define szPlatform "lin"

template<size_t A, class T>
T* aligned_malloc(size_t n)
{
  return static_cast<T*>(std::aligned_malloc(A,n));
}

template<class T>
void aligned_free(T *p)
{
	std::free(p);
}

#if defined(__i386__)
#define ARCH_X86
#else
#define ARCH_X64
#endif

#endif // _MSVC_LANG

/////////////////////////

// http://supercomputingblog.com/optimization/getting-started-with-sse-programming/

// 8 categories, each at 16 discretization levels
// impl1: return largest index s.t. cdf[index] is not > sample, otherwise 8
// impl2: return largest index s.t. cdf[index] is <= sample

// ?? assume [-1] is 0 and [8] is 0x0F ??

// cdf values are interval-ending values. ie: cdf[0] is the ending value of interval 0. therefore:
// cdf[-1] == 0x00, implicitly
// cdf[-1] <= cdf[0] <= cdf[1] <= ... 2, 3, 4, 5 ... <= cdf[6] <= cdf[7] <= cdf[8] - which is 10 intervals!
// cdf[8] == 0x0F, implicitly
//
// therefore:
// - cdf[0] is the ending prob for interval 0
// - cdf[7] deosn't need to be 0x0F because cdf[8] implicitly has the value of 0x0F
//
// 0 < s (implied)
// case 1. when [0] > s then c := 0
// case 2. when [7] < s then c := 9
// case 3. c := minimum value of c+1 satistfying [c] = s
// case 4. c := minimum value of c+1 satistfying [c] > s

//////////////////////////////

void calc_category_naive_x1(const uint8_t sample, const size_t n, const uint32_t* cdf, uint8_t* cat)
{
    static const uint32_t m[8] =
	{
	  0x0000000F, 0x000000F0, 0x00000F00, 0x0000F000,
	  0x000F0000, 0x00F00000, 0x0F000000, 0xF0000000
	};

	uint32_t s = 0;
	for (int i = 0; i < 8; i++) s |= sample << (i * 4);

	for (size_t i = 0; i < n; i++)
	{
		uint8_t c;
		if ((s & m[0]) == 0)  c = 0; // case 1
		else if ((cdf[i] & m[0]) >= (s & m[0]))  c = 1;
		else if ((cdf[i] & m[1]) >= (s & m[1]))  c = 2;
		else if ((cdf[i] & m[2]) >= (s & m[2]))  c = 3;
		else if ((cdf[i] & m[3]) >= (s & m[3]))  c = 4;
		else if ((cdf[i] & m[4]) >= (s & m[4]))  c = 5;
		else if ((cdf[i] & m[5]) >= (s & m[5]))  c = 6;
		else if ((cdf[i] & m[6]) >= (s & m[6]))  c = 7;
		else if ((cdf[i] & m[7]) >= (s & m[7]))  c = 8;
		else c = 9; // case 2
		cat[i] = c;
	}
}

void calc_category_naive_x8(const uint8_t sample, const size_t n, const uint32_t* cdf, uint32_t* cat)
{
    static const uint32_t m[8] =
	{
	  0x0000000F, 0x000000F0, 0x00000F00, 0x0000F000,
	  0x000F0000, 0x00F00000, 0x0F000000, 0xF0000000
	};
	static const uint8_t shift[8] = {0, 4, 8, 12, 16, 20, 24, 28};

	uint32_t s = 0;
	for (int i = 0; i < 8; i++) s |= sample << (i * 4);

	for (size_t i = 0; i < n; i+=8)
	{
	  uint32_t c32 = 0;
	  for (size_t j = 0; j < 8; j++)
		{
		  const size_t k = i+j;
		  uint8_t c;
		  if ((s & m[0]) == 0)  c = 0; // case 1
		  else if ((cdf[k] & m[0]) >= (s & m[0]))  c = 1;
		  else if ((cdf[k] & m[1]) >= (s & m[1]))  c = 2;
		  else if ((cdf[k] & m[2]) >= (s & m[2]))  c = 3;
		  else if ((cdf[k] & m[3]) >= (s & m[3]))  c = 4;
		  else if ((cdf[k] & m[4]) >= (s & m[4]))  c = 5;
		  else if ((cdf[k] & m[5]) >= (s & m[5]))  c = 6;
		  else if ((cdf[k] & m[6]) >= (s & m[6]))  c = 7;
		  else if ((cdf[k] & m[7]) >= (s & m[7]))  c = 8;
		  else c = 9; // case 2
		  c32 = (uint32_t)c | (uint32_t)(c32 << shift[j]); // makes low bits into low j index
		}
	  cat[i]=c32;
	}
}

#if defined(ARCH_X86)

void calc_category_sse_x1(const uint8_t sample, const size_t n, const uint32_t* cdf32, uint8_t* cat)
{
	static const __m64 lns = _mm_set_pi8(-1, +3, -1, +2, -1, +1, -1, +0);
	static const __m64 uns = _mm_set_pi8(+3, -1, +2, -1, +1, -1, +0, -1);
	static const __m64 ord = _mm_set_pi8(+8, +7, +6, +5, +4, +3, +2, +1);
	static const __m64 eqr = _mm_set_pi8(-1, -1, -1, -1, +7, +6, +5, +4);
	static const __m64 hr1 = _mm_set_pi8(-1, +7, -1, +5, -1, +3, -1, +1);
	static const __m64 hr2 = _mm_set_pi8(-1, -1, -1, +6, -1, -1, -1, +2);
	static const __m64 hr3 = _mm_set_pi8(-1, -1, -1, -1, -1, -1, -1, +4);
	static const __m64 fff = _mm_set1_pi8(0xFF);

	const uint32_t sample_cat9 = sample << 28; // shift into upper nibble
	const __m64 sample64 = _mm_set1_pi8(sample);
	for (size_t i = 0; i < n; i++)
	{
	  uint8_t c;
	  if (sample == 0) c = 0;
	  else if ((cdf32[i] & 0xF0000000) < sample_cat9) c = 9;
	  else
		{
		  // uncompress cdf nibbles
		  __m64 cdf64 = _mm_or_si64(
		    _mm_shuffle_pi8(_mm_cvtsi32_si64(cdf32[i] & 0x0F0F0F0F), lns),
		    _mm_shuffle_pi8(_mm_cvtsi32_si64((cdf32[i] >> 4) & 0x0F0F0F0F), uns)
		  );
		  // scan it
		  __m64 cmp = _mm_or_si64(
	        _mm_cmpeq_pi8(cdf64, sample64), // case 3
		    _mm_cmpgt_pi8(cdf64, sample64) // case 4
          );
		  // set category ordinals, merged with negated mask so min works right
		  __m64 reduce_me = _mm_or_si64(_mm_and_si64(cmp, ord), _m_pandn(cmp, fff));
		  // horizontal min reduction
		  reduce_me = _mm_min_pu8(reduce_me, _mm_shuffle_pi8(reduce_me, hr1));
		  reduce_me = _mm_min_pu8(reduce_me, _mm_shuffle_pi8(reduce_me, hr2));
		  reduce_me = _mm_min_pu8(reduce_me, _mm_shuffle_pi8(reduce_me, hr3));
		  c = _mm_cvtsi64_si32(reduce_me);
		}
	  cat[i] = c;
	}
	_mm_empty();
}

void calc_category_sse_x8(const uint8_t sample, const size_t n, const uint32_t* cdf32, uint32_t* cat)
{
	static const __m64 lns = _mm_set_pi8(-1, +3, -1, +2, -1, +1, -1, +0);
	static const __m64 uns = _mm_set_pi8(+3, -1, +2, -1, +1, -1, +0, -1);
	static const __m64 ord = _mm_set_pi8(+8, +7, +6, +5, +4, +3, +2, +1);
	static const __m64 eqr = _mm_set_pi8(-1, -1, -1, -1, +7, +6, +5, +4);
	static const __m64 hr1 = _mm_set_pi8(-1, +7, -1, +5, -1, +3, -1, +1);
	static const __m64 hr2 = _mm_set_pi8(-1, -1, -1, +6, -1, -1, -1, +2);
	static const __m64 hr3 = _mm_set_pi8(-1, -1, -1, -1, -1, -1, -1, +4);
	static const __m64 fff = _mm_set1_pi8(0xFF);
	static const uint8_t shift[8] = {0, 4, 8, 12, 16, 20, 24, 28};

	const uint32_t sample_cat9 = sample << 28; // shift into upper nibble
	const __m64 sample64 = _mm_set1_pi8(sample);
	for (size_t i = 0; i < n; i+=8)
	{
	  uint32_t c32 = 0;
	  for (size_t j = 0; j < 8; j++)
		{
		  const size_t k = i+j;
		  uint8_t c;
		  if (sample == 0) c = 0;
		  else if ((cdf32[k] & 0xF0000000) < sample_cat9) c = 9;
		  else
			{
			  // uncompress cdf nibbles
			  __m64 cdf64 = _mm_or_si64(
			    _mm_shuffle_pi8(_mm_cvtsi32_si64(cdf32[k] & 0x0F0F0F0F), lns),
			    _mm_shuffle_pi8(_mm_cvtsi32_si64((cdf32[k] >> 4) & 0x0F0F0F0F), uns)
		      );
			  // scan it
			  __m64 cmp = _mm_or_si64(
		        _mm_cmpeq_pi8(cdf64, sample64), // case 3
			    _mm_cmpgt_pi8(cdf64, sample64) // case 4
	          );
			  // set category ordinals, merged with negated mask so min works right
			  __m64 reduce_me = _mm_or_si64(_mm_and_si64(cmp, ord), _m_pandn(cmp, fff));
			  // horizontal min reduction
			  reduce_me = _mm_min_pu8(reduce_me, _mm_shuffle_pi8(reduce_me, hr1));
			  reduce_me = _mm_min_pu8(reduce_me, _mm_shuffle_pi8(reduce_me, hr2));
			  reduce_me = _mm_min_pu8(reduce_me, _mm_shuffle_pi8(reduce_me, hr3));
			  c = _mm_cvtsi64_si32(reduce_me);
			}
		  c32 = (uint32_t)c | (uint32_t)(c32 << shift[j]); // makes low bits into low j index
      }
	  cat[i] = c32;
	}
	_mm_empty();
}

#elif defined(ARCH_X64)

void calc_category_sse_x1(const uint8_t sample, const size_t n, const uint32_t* cdf32, uint8_t* cat)
{
	static const __m128i lns = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, +3, -1, +2, -1, +1, -1, +0, -1);
	static const __m128i uns = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, +3, -1, +2, -1, +1, -1, +0, -1, -1);
	static const __m128i bds = _mm_set_epi8(+0, +0, +0, +0, +0, +0, 15, +0, +0, +0, +0, +0, +0, +0, +0, +0);
	static const __m128i ord = _mm_set_epi8(+0, +0, +0, +0, +0, +0, +9, +8, +7, +6, +5, +4, +3, +2, +1, +0);
	static const __m128i hr1 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, +9, -1, +7, -1, +5, -1, +3, -1, +1);
	static const __m128i hr2 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, +8, -1, +6, -1, -1, -1, +2);
	static const __m128i hr3 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +6, -1, -1, -1, +4);
	static const __m128i hr4 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +4);
	static const __m128i fff = _mm_set1_epi8(0xFF);

	const __m128i sample128 = _mm_set1_epi8(sample);
	for (size_t i = 0; i < n; i++)
	{
		// uncompress cdf nibbles
		__m128i cdf80 = _mm_or_si128(
			_mm_shuffle_epi8(_mm_set1_epi32(cdf32[i] & 0x0F0F0F0F), lns),
			_mm_shuffle_epi8(_mm_set1_epi32((cdf32[i] >> 4) & 0x0F0F0F0F), uns)
		);
		cdf80 = _mm_or_si128(cdf80, bds);
		// lte scan the cdf and set category ordinals
		__m128i reduce_me = _mm_or_si128(
			_mm_cmpeq_epi8(cdf80, sample128),
			_mm_cmpgt_epi8(cdf80, sample128)
		);
		// set category ordinals, merged with negated mask so min works right
		reduce_me = _mm_or_si128(_mm_and_si128(reduce_me, ord), _mm_andnot_si128(reduce_me, fff));
		// horizontal min reduction
		reduce_me = _mm_min_epu8(reduce_me, _mm_shuffle_epi8(reduce_me, hr1));
		reduce_me = _mm_min_epu8(reduce_me, _mm_shuffle_epi8(reduce_me, hr2));
		reduce_me = _mm_min_epu8(reduce_me, _mm_shuffle_epi8(reduce_me, hr3));
		reduce_me = _mm_min_epu8(reduce_me, _mm_shuffle_epi8(reduce_me, hr4));
		cat[i] = _mm_cvtsi128_si32(reduce_me);
	}
}

void calc_category_sse_x8(const uint8_t sample, const size_t n, const uint32_t* cdf32, uint32_t* cat)
{
	static const __m128i lns = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, +3, -1, +2, -1, +1, -1, +0, -1);
	static const __m128i uns = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, +3, -1, +2, -1, +1, -1, +0, -1, -1);
	static const __m128i bds = _mm_set_epi8(+0, +0, +0, +0, +0, +0, 15, +0, +0, +0, +0, +0, +0, +0, +0, +0);
	static const __m128i ord = _mm_set_epi8(+0, +0, +0, +0, +0, +0, +9, +8, +7, +6, +5, +4, +3, +2, +1, +0);
	static const __m128i hr1 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, +9, -1, +7, -1, +5, -1, +3, -1, +1);
	static const __m128i hr2 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, +8, -1, +6, -1, -1, -1, +2);
	static const __m128i hr3 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +6, -1, -1, -1, +4);
	static const __m128i hr4 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +4);
	static const __m128i fff = _mm_set1_epi8(0xFF);
	static const uint8_t shift[8] = { 0, 4, 8, 12, 16, 20, 24, 28 };

	const __m128i sample128 = _mm_set1_epi8(sample);
	for (size_t i = 0; i < n; i+=8)
	{
		uint32_t c32 = 0;
		for (size_t j = 0; j < 8; j++)
		{
			const size_t k = i + j;
			// uncompress cdf nibbles
			__m128i cdf80 = _mm_or_si128(
				_mm_shuffle_epi8(_mm_set1_epi32(cdf32[k] & 0x0F0F0F0F), lns),
				_mm_shuffle_epi8(_mm_set1_epi32((cdf32[k] >> 4) & 0x0F0F0F0F), uns)
			);
			cdf80 = _mm_or_si128(cdf80, bds);
			// lte scan the cdf and set category ordinals
			__m128i reduce_me = _mm_or_si128(
				_mm_cmpeq_epi8(cdf80, sample128),
				_mm_cmpgt_epi8(cdf80, sample128)
			);
			// set category ordinals, merged with negated mask so min works right
			reduce_me = _mm_or_si128(_mm_and_si128(reduce_me, ord), _mm_andnot_si128(reduce_me, fff));
			// horizontal min reduction
			reduce_me = _mm_min_epu8(reduce_me, _mm_shuffle_epi8(reduce_me, hr1));
			reduce_me = _mm_min_epu8(reduce_me, _mm_shuffle_epi8(reduce_me, hr2));
			reduce_me = _mm_min_epu8(reduce_me, _mm_shuffle_epi8(reduce_me, hr3));
			reduce_me = _mm_min_epu8(reduce_me, _mm_shuffle_epi8(reduce_me, hr4));
			uint32_t c = _mm_cvtsi128_si32(reduce_me);
			c32 = (uint32_t)c | (uint32_t)(c32 << shift[j]); // makes low bits into low j index
		}
		cat[i] = c32;
	}
}

#endif

///////////////////////

void validate_category()
{
	uint8_t cata, catb;
	{
		// test when cdf has a value of sample
		uint32_t cdf = 0x76543210;
		calc_category_naive_x1(4, 1, &cdf, &cata);
		calc_category_sse_x1(4, 1, &cdf, &catb);
		if (cata != 5) BREAK;
		if (cata != catb) BREAK;
	}
	{
		// test when cdf doesn't have a value of sample
		uint32_t cdf = 0xFEA76510;
		calc_category_naive_x1(4, 1, &cdf, &cata);
		calc_category_sse_x1(4, 1, &cdf, &catb);
		if (cata != 3) BREAK;
		if (cata != catb) BREAK;
	}
	{
		// test when cdf has multiple values of sample
		uint32_t cdf = 0x76444210;
		calc_category_naive_x1(4, 1, &cdf, &cata);
		calc_category_sse_x1(4, 1, &cdf, &catb);
		if (cata != 4) BREAK;
		if (cata != catb) BREAK;
	}
	{
		// test when sample exceeds cdf
		uint32_t cdf = 0x33333210;
		calc_category_naive_x1(4, 1, &cdf, &cata);
		calc_category_sse_x1(4, 1, &cdf, &catb);
		if (cata != 9) BREAK;
		if (cata != catb) BREAK;
	}
	{
		// test when sample underflows cdf
		uint32_t cdf = 0x55443322;
		calc_category_naive_x1(1, 1, &cdf, &cata);
		calc_category_sse_x1(1, 1, &cdf, &catb);
		if (cata != 1) BREAK;
		if (cata != catb) BREAK;
	}
	{
		// test when sample is zero
		uint32_t cdf = 0x55443300;
		calc_category_naive_x1(0, 1, &cdf, &cata);
		calc_category_sse_x1(0, 1, &cdf, &catb);
		if (cata != 0) BREAK;
		if (cata != catb) BREAK;
	}
	{
		// test when sample is zero and underflows
		uint32_t cdf = 0x55443322;
		calc_category_naive_x1(0, 1, &cdf, &cata);
		calc_category_sse_x1(0, 1, &cdf, &catb);
		if (cata != 0) BREAK;
		if (cata != catb) BREAK;
	}
}

///////////////////////

enum ECategoryType
{
	ECTError, ECTBaseX1, ECTBaseX8, ECTFastX1, ECTFastX8
};

ECategoryType benchmark_category()
{
	size_t BUF = 16 * 64;
	size_t N = BUF / 16;
	auto cdf = aligned_malloc<16, uint32_t>(BUF);
	auto cata = aligned_malloc<16, uint8_t>(BUF);
	auto catb = aligned_malloc<16, uint8_t>(BUF);

	//std::cout << "reinitializing." << std::endl;
	for (size_t i = 0; i < N; ++i)
		cdf[i] = 0;
	for (size_t i = 0; i < N; ++i)
		for (uint32_t digit = rand() % 3, j = 0; j < 8; j++, digit += rand() % 3)
			cdf[i] = (digit << 28) | (cdf[i] >> 4);

	uint8_t s = 6;

	//std::cout << "checking x1." << std::endl;
	calc_category_naive_x1(s, N, cdf, cata);
	calc_category_sse_x1(s, N, cdf, catb);
	//for(size_t i=0; i<16; ++i)
	//std::cout << 1+cata[i] << " " << 1+catb[i] << std::endl;
	for (size_t i = 0; i < N; ++i)
		if (cata[i] != catb[i]) BREAK;

	//std::cout << "checking x8" << std::endl;
	calc_category_naive_x8(s, N, cdf, (uint32_t*)cata);
	calc_category_sse_x8(s, N, cdf, (uint32_t*)catb);
	//for(size_t i=0; i<16; ++i)
	//std::cout << 1+cata[i] << " " << 1+catb[i] << std::endl;
	for (size_t i = 0; i < N; ++i)
		if (cata[i] != catb[i]) BREAK;

	const int M = 1024 * 1024;
	double best_cps = 0;
	ECategoryType best_type = ECTError;

	{
		auto start = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i != M; i++)
			calc_category_naive_x1(s, N, cdf, cata);
		auto finish = std::chrono::high_resolution_clock::now();
		auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
		double perf = (double)1e+9 * (double)M / (double)dur;
		std::cout << "1. scalar unbatched " << perf << std::endl;
		if (perf > best_cps) { best_cps = perf; best_type = ECTBaseX1; }
	}
	{
		auto start = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i != M; i++)
			calc_category_naive_x8(s, N, cdf, (uint32_t*)cata);
		auto finish = std::chrono::high_resolution_clock::now();
		auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
		double perf = (double)1e+9 * (double)M / (double)dur;
		std::cout << "2. scalar batched " << perf << std::endl;
		if (perf > best_cps) { best_cps = perf; best_type = ECTBaseX8; }
	}
	{
		auto start = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i != M; i++)
			calc_category_sse_x1(s, N, cdf, catb);
		auto finish = std::chrono::high_resolution_clock::now();
		auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
		double perf = (double)1e+9 * (double)M / (double)dur;
		std::cout << "3. vectorized unbatched " << perf << std::endl;
		if (perf > best_cps) { best_cps = perf; best_type = ECTFastX1; }
	}
	{
		auto start = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i != M; i++)
			calc_category_sse_x8(s, N, cdf, (uint32_t*)catb);
		auto finish = std::chrono::high_resolution_clock::now();
		auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
		double perf = (double)1e+9 * (double)M / (double)dur;
		std::cout << "4. vectorized batched " << perf << std::endl;
		if (perf > best_cps) { best_cps = perf; best_type = ECTFastX8; }
	}

	aligned_free(cdf);
	aligned_free(cata);
	aligned_free(catb);

	return best_type;
}

///////////////////////////////

int main()
{
	std::cout << "validating." << std::endl;
	validate_category();
	std::cout << "benchmarking on architecture " szPlatform "-" szArch "." << std::endl;
	auto best_type = benchmark_category();
	std::cout << "best is " << best_type << std::endl;
	return 0;
}



















