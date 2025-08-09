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

    # Cast to float32 to avoid mixed precision dtype mismatch
    dtype = torch.float32
    P = P.to(dtype)
    Q = Q.to(dtype)

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
      pred_coords: (B, L, A_pred, 3)
      true_coords: (B, L, A_true, 3)  where A_true can be >= A_pred (multiple refs)
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
        # Shapes
        P = pred_coords[i, :L]    # (L, A_pred, 3)
        T = true_coords[i, :L]    # (L, A_true, 3)
        Lp, Ap = P.shape[0], P.shape[1]
        Lt, At = T.shape[0], T.shape[1]

        # Flatten predicted (cast to float32 to avoid half/float mismatch)
        pred = P.reshape(-1, 3).to(torch.float32)   # (L*Ap, 3)

        # If true has multiple atom sets (e.g., multiple conformations), compute min RMSD
        if At == Ap:
            true = T.reshape(-1, 3).to(torch.float32)
            rmsds.append(rmsd(pred, true, align=align))
        elif At % Ap == 0:
            k = At // Ap
            # Split true atoms into k groups along atom dimension
            cand = []
            for g in range(k):
                true_g = T[:, g*Ap:(g+1)*Ap, :].reshape(-1, 3).to(torch.float32)
                if true_g.shape == pred.shape:
                    cand.append(rmsd(pred, true_g, align=align))
            if cand:
                rmsds.append(torch.stack(cand).min())
            else:
                print(f"No valid true groups matching pred atoms in batch {i}: pred Ap={Ap}, true At={At}")
                rmsds.append(torch.tensor(float('nan'), device=pred_coords.device))
        else:
            # Fallback: truncate/pad true to match pred
            if At > Ap:
                true = T[:, :Ap, :].reshape(-1, 3).to(torch.float32)
            else:
                # Pad true with repeats to match
                pad_repeat = (Ap + At - 1) // At
                T_rep = T.repeat(1, pad_repeat, 1)[:, :Ap, :]
                true = T_rep.reshape(-1, 3)
            if pred.shape == true.shape:
                rmsds.append(rmsd(pred, true, align=align))
            else:
                print(f"Shape mismatch unresolved in batch {i}: pred {pred.shape}, true {true.shape}")
                rmsds.append(torch.tensor(float('nan'), device=pred_coords.device))
    rmsd_tensor = torch.stack(rmsds)
    return rmsd_tensor, rmsd_tensor.nanmean()


def tm_score(pred: torch.Tensor, true: torch.Tensor, d0: float | None = None, align: bool = True) -> torch.Tensor:
    """Compute an internal TM-score variant.
    pred, true: (N, 3) tensors. Casts to float32 and optionally aligns.
    """
    pred = pred.to(torch.float32)
    true = true.to(torch.float32)
    if pred.dim() == 3:
        pred = pred.reshape(-1, 3)
    if true.dim() == 3:
        true = true.reshape(-1, 3)
    if pred.shape != true.shape:
        raise ValueError(f"tm_score expects matching shapes, got pred {pred.shape}, true {true.shape}")
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
    """Batch TM-score for padded batches with support for multiple true conformations.
    Inputs:
      pred_coords: (B, L, A_pred, 3)
      true_coords: (B, L, A_true, 3)
      lengths: (B,)
    Returns: (tm_per_sample[B], mean_tm)
    """
    B = pred_coords.shape[0]
    scores = []
    for i in range(B):
        L = int(lengths[i].item())
        if L <= 0:
            scores.append(torch.tensor(0.0, device=pred_coords.device))
            continue
        P = pred_coords[i, :L]    # (L, A_pred, 3)
        T = true_coords[i, :L]    # (L, A_true, 3)
        Ap = P.shape[1]
        At = T.shape[1]
        pred = P.reshape(-1, 3).to(torch.float32)
        if At == Ap:
            true = T.reshape(-1, 3).to(torch.float32)
            scores.append(tm_score(pred, true, align=align))
        elif At % Ap == 0:
            k = At // Ap
            cand = []
            for g in range(k):
                true_g = T[:, g*Ap:(g+1)*Ap, :].reshape(-1, 3).to(torch.float32)
                if true_g.shape == pred.shape:
                    cand.append(tm_score(pred, true_g, align=align))
            if cand:
                # TM-score: higher is better
                scores.append(torch.stack(cand).max())
            else:
                scores.append(torch.tensor(float('nan'), device=pred_coords.device))
        else:
            # Fallback match
            if At > Ap:
                true = T[:, :Ap, :].reshape(-1, 3).to(torch.float32)
            else:
                pad_repeat = (Ap + At - 1) // At
                T_rep = T.repeat(1, pad_repeat, 1)[:, :Ap, :]
                true = T_rep.reshape(-1, 3).to(torch.float32)
            if pred.shape == true.shape:
                scores.append(tm_score(pred, true, align=align))
            else:
                scores.append(torch.tensor(float('nan'), device=pred_coords.device))
            continue
        pred = pred_coords[i, :L]
        true = true_coords[i, :L]
        scores.append(tm_score(pred, true, align=align))
    s = torch.stack(scores)
    return s, s.mean()