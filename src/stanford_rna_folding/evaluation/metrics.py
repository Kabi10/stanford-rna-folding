"""
Evaluation metrics for RNA 3D structure prediction.
Includes RMSD and TM-score with optional Kabsch alignment and batch utilities.
"""

from typing import Tuple
import torch
import torch.nn.functional as F


def kabsch_align(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Align P to Q using the Kabsch algorithm.
    P, Q: shape (..., N, 3). Returns aligned P of same shape.
    """
    # Ensure P and Q have the same shape
    if P.shape != Q.shape:
        raise ValueError(f"P and Q must have the same shape, got P: {P.shape}, Q: {Q.shape}")

    # Handle empty or invalid tensors
    if P.numel() == 0 or Q.numel() == 0:
        return P.clone()

    # Center P and Q
    Pc = P - P.mean(dim=-2, keepdim=True)
    Qc = Q - Q.mean(dim=-2, keepdim=True)
    # Compute covariance
    H = Pc.transpose(-2, -1) @ Qc  # (..., 3, 3)
    # SVD
    U, S, Vt = torch.linalg.svd(H)
    V = Vt.transpose(-2, -1)
    # Correct possible reflection
    d = torch.det(V @ U.transpose(-2, -1))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=-1))
    R = V @ D @ U.transpose(-2, -1)
    # Rotate P
    P_aligned = (Pc @ R)
    # Translate to Q centroid
    P_aligned = P_aligned + Q.mean(dim=-2, keepdim=True)
    return P_aligned


def rmsd(pred: torch.Tensor, true: torch.Tensor, align: bool = True) -> torch.Tensor:
    """Compute RMSD between predicted and true coords. Shapes: (..., N, 3)."""
    if align:
        pred = kabsch_align(pred, true)
    diff2 = (pred - true) ** 2
    mse = diff2.mean(dim=(-2, -1))
    return torch.sqrt(mse + 1e-8)


def batch_rmsd(pred_coords: torch.Tensor, true_coords: torch.Tensor, lengths: torch.Tensor, align: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute RMSD per-sample for a batch with padding.
    Inputs:
      pred_coords: (B, L, A, 3)
      true_coords: (B, L, A, 3)
      lengths: (B,) valid sequence lengths
    Returns: (rmsd_per_sample[B], mean_rmsd)
    """
    B = pred_coords.shape[0]
    rmsds = []
    for i in range(B):
        L = int(lengths[i].item())
        if L <= 0:
            rmsds.append(torch.tensor(0.0, device=pred_coords.device))
            continue
        # Flatten atoms into N = L*A
        pred = pred_coords[i, :L].reshape(-1, 3)
        true = true_coords[i, :L].reshape(-1, 3)

        # Debug shape mismatch
        if pred.shape != true.shape:
            print(f"Shape mismatch in batch {i}: pred {pred.shape}, true {true.shape}")
            print(f"Original shapes: pred_coords[{i}, :{L}] = {pred_coords[i, :L].shape}, true_coords[{i}, :{L}] = {true_coords[i, :L].shape}")
            # Skip this sample to avoid crash
            rmsds.append(torch.tensor(float('nan'), device=pred_coords.device))
            continue

        rmsds.append(rmsd(pred, true, align=align))
    rmsd_tensor = torch.stack(rmsds)
    return rmsd_tensor, rmsd_tensor.mean()


def tm_score(pred: torch.Tensor, true: torch.Tensor, d0: float | None = None, align: bool = True) -> torch.Tensor:
    """Compute an internal TM-score variant for identical-length sets.
    pred, true: (N, 3) or (L, A, 3). If (L, A, 3), will be flattened.
    """
    if pred.dim() == 3:
        pred = pred.reshape(-1, 3)
    if true.dim() == 3:
        true = true.reshape(-1, 3)
    N = pred.shape[-2]
    if align:
        pred = kabsch_align(pred, true)
    if d0 is None:
        # Standard approximate d0 as in TM-score literature
        Lref = N
        d0 = 1.24 * (Lref - 15) ** (1.0/3) - 1.8
        d0 = float(max(d0, 0.5))
    dist = torch.linalg.norm(pred - true, dim=-1)  # (N,)
    score = torch.mean(1.0 / (1.0 + (dist / d0) ** 2))
    return score


def batch_tm_score(pred_coords: torch.Tensor, true_coords: torch.Tensor, lengths: torch.Tensor, align: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch TM-score for padded batches.
    Returns: (tm_per_sample[B], mean_tm)
    """
    B = pred_coords.shape[0]
    scores = []
    for i in range(B):
        L = int(lengths[i].item())
        if L <= 0:
            scores.append(torch.tensor(0.0, device=pred_coords.device))
            continue
        pred = pred_coords[i, :L]
        true = true_coords[i, :L]
        scores.append(tm_score(pred, true, align=align))
    s = torch.stack(scores)
    return s, s.mean()