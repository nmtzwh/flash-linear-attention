# GatedDeltaNet Pure PyTorch Implementation

This document outlines the recent modifications made to the `GatedDeltaNet` layer, which refactor it to run entirely on pure PyTorch tensor primitives without depending on the `triton` backend. This is particularly useful for environments where `triton` is unsupported (e.g., standard CPU instances, some AMD/Mac devices) or for easier debugging and numerical verification.

## Summary of Changes

1. **Standalone Sub-Modules**
   - Eliminated the `triton`-accelerated `ShortConvolution` from `fla.modules` and replaced it with an inline equivalent constructed purely via `torch.nn.Conv1d` and native padding/concatenation.
   - Replaced `RMSNorm` and `FusedRMSNormGated` with simple, mathematically equivalent native Pytorch module classes.

2. **Native Delta Rule Ops**
   - The layer previously dispatched to the highly-optimized Triton implementations (`chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule`).
   - We redirected these to the reference implementation counterparts from `fla.ops.gated_delta_rule.naive` (`naive_chunk_gated_delta_rule`, `naive_recurrent_gated_delta_rule`).

3. **L2 Normalization**
   - The original kernel utilized `use_qk_l2norm_in_kernel=True` for performance.
   - We decoupled this step and implemented explicit pure Pytorch L2 normalization (`F.normalize(..., p=2.0, dim=-1, eps=1e-6)`) applied right before calculating the delta rule.

4. **Parameters Cleanup**
   - Keyword arguments specific to triton optimizations (such as `cu_seqlens`) were stripped from the operation calls, as the naive kernels rely entirely on dense sequence padding and native PyTorch loops/einsums.

## How to Run the Numerical Correctness Test

To ensure that the refactored pure PyTorch chunk and recurrent modes still yield mathematically equivalent results to each other, a test script `test_gated_deltanet_correctness.py` has been provided in the root directory.

Because the wider `fla` package natively asserts for `triton` during initialization (specifically deep inside attention operation structures), the script uses an elaborate mocking mechanism to intercept `triton` imports seamlessly allowing the pure CPU testing of the module.

**To run the test:**
```bash
python test_gated_deltanet_correctness.py
```

**Expected Output:**
```
Max difference between chunk and recurrent modes: 0.000000
Success! Numerical correctness verified.
```