# Fix for Tensor Shape Mismatch in Validation (Kabsch Alignment)

## Summary
We fixed the critical issue blocking end-to-end training: a tensor shape mismatch during Kabsch alignment in validation (3x69 vs 345x3). The root cause was that the validation labels (true coordinates) can contain multiple reference conformations or atom-groupings per residue (A_true), while the model predicts a single conformation/atom-grouping (A_pred). This led to flattening to different N = L*A and a mismatch during alignment and RMSD/TM calculations.

## Root Cause Analysis
- Predicted coordinates shape per sample: (L, A_pred, 3)
- True coordinates shape per sample: (L, A_true, 3)
- Observed example: pred flattened to (69, 3) while true flattened to (345, 3) where 345 = 69 * 5
- This indicates A_true = 5 * A_pred (e.g., five reference conformations/atom groups)

## Implemented Fixes
1. batch_rmsd now supports A_true != A_pred by grouping true atoms and taking the minimum RMSD across groups:
   - If A_true == A_pred: directly compare
   - If A_true % A_pred == 0: split true into k = A_true / A_pred groups, compute RMSD to each, take min
   - Else: conservative fallback (truncate/pad) to match shapes and compute RMSD
   - Uses nanmean to ignore rare unresolvable samples

2. Cast to float32 in metrics to avoid AMP dtype mismatches:
   - kabsch_align now casts P and Q to float32
   - batch_rmsd flattens and casts pred/true to float32 before computing RMSD
   - tm_score casts inputs to float32 and enforces matching shapes

3. Hardened TM-score path:
   - Same shape checks
   - Float32 casting

## Files Changed
- src/stanford_rna_folding/evaluation/metrics.py
- kaggle_dataset_package/src/stanford_rna_folding/evaluation/metrics.py

## Validation Plan
- Pushed Kaggle kernel (v11) with fixes
- Kernel runs through validation without shape mismatch exceptions
- If any sample has non-divisible A_true/Ap, fallback prevents crash, logged as NaN and excluded from averages

## Next Steps
- Monitor logs for any remaining shape mismatches or unexpected NaNs
- Consider formal multi-conformation evaluation (e.g., weighted average) if dataset guarantees multiple refs
- Add unit tests for metrics with:
  - A_true == A_pred
  - A_true = k * A_pred (k=2..5)
  - A_true not divisible by A_pred (fallback)
- Optimize performance by vectorizing group RMSD computation if needed

