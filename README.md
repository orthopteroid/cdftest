# cdftest
x86/MMX &amp; x64/SSE code to test small Cumulative Distributions Functions

Small cumulative distributions functions can be used in a variety of ways
to perform random selection either over uniform or weighted distributions. Even
discontinious distributions.

This little project was a test to determine if an interesting way of packing
small distributions into a small binary format was viable and what kind of performance
benifits there might be in such an approach.

The CDF used in this project conatins 8 sample points, each with 16 quantizations. This
should sound very familiar - it can be stored as 8 nibbles in an unsigned 32bit value.

The code to test evaluation performance is broken down by:
- platform to test the compiler: windows or linux
- architecture to test various possible benifits of using a uint32 packing format
- batching evaluations individually or in grouped a cachline-friendly size

# RESULTS

Tests were conducted on a windows x64 host and a linux x86 host.
Performance measured in evaluations per second.

```
validating.
benchmarking on architecture win-x86.
1. scalar unbatched 2.89846e+06
2. scalar batched 2.69774e+06
3. vectorized unbatched 4.62584e+06
4. vectorized batched 4.03884e+06
best is 3
```

```
validating.
benchmarking on architecture win-x64.
1. scalar unbatched 2.68311e+06
2. scalar batched 6.52668e+06
3. vectorized unbatched 4.0696e+06
4. vectorized batched 4.44681e+06
best is 2
```

I'm sort of surprised by these results: neither shows the vectorized
evalutaors as better than the scalar ones. Perhaps this is because I hand-wrote
the vector code and the compiler is simply better than me. Very likely.
