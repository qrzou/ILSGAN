import torch
import numpy as np

#----------------------------------------------------------------------------

def iou_binary(mask_A, mask_B, debug=False):
    if debug:
        assert mask_A.shape == mask_B.shape
        assert mask_A.dtype == torch.bool
        assert mask_B.dtype == torch.bool
    intersection = (mask_A * mask_B).sum((1, 2, 3))
    union = (mask_A + mask_B).sum((1, 2, 3))
    # Return -100 if union is zero, else return IOU
    return torch.where(union == 0, torch.tensor(-100.0),
                       intersection.float() / union.float())

#----------------------------------------------------------------------------

def average_segcover(segA, segB, ignore_background=False):
    """
    Covering of segA by segB
    segA.shape = [batch size, 1, img_dim1, img_dim2]
    segB.shape = [batch size, 1, img_dim1, img_dim2]

    scale: If true, take weighted mean over IOU values proportional to the
           the number of pixels of the mask being covered.

    Assumes labels in segA and segB are non-negative integers.
    Negative labels will be ignored.
    """

    assert segA.shape == segB.shape, f"{segA.shape} - {segB.shape}"
    assert segA.shape[1] == 1 and segB.shape[1] == 1
    bsz = segA.shape[0]
    nonignore = (segA >= 0)

    mean_scores = torch.tensor(bsz*[0.0])
    N = torch.tensor(bsz*[0])
    scaled_scores = torch.tensor(bsz*[0.0])
    scaling_sum = torch.tensor(bsz*[0])

    # Find unique label indices to iterate over
    if ignore_background:
        iter_segA = torch.unique(segA[segA > 0]).tolist()
    else:
        iter_segA = torch.unique(segA[segA >= 0]).tolist()
    iter_segB = torch.unique(segB[segB >= 0]).tolist()
    # Loop over segA
    for i in iter_segA:
        binaryA = segA == i
        if not binaryA.any():
            continue
        max_iou = torch.tensor(bsz*[0.0])
        # Loop over segB to find max IOU
        for j in iter_segB:
            # Do not penalise pixels that are in ignore regions
            binaryB = (segB == j) * nonignore
            if not binaryB.any():
                continue
            iou = iou_binary(binaryA, binaryB)
            max_iou = torch.where(iou > max_iou, iou, max_iou)
        # Accumulate scores
        mean_scores += max_iou
        N = torch.where(binaryA.sum((1, 2, 3)) > 0, N+1, N)
        scaled_scores += binaryA.sum((1, 2, 3)).float() * max_iou
        scaling_sum += binaryA.sum((1, 2, 3))

    # Compute coverage
    mean_sc = mean_scores / torch.max(N, torch.tensor(1)).float()
    scaled_sc = scaled_scores / torch.max(scaling_sum, torch.tensor(1)).float()

    # Sanity check
    assert (mean_sc >= 0).all() and (mean_sc <= 1).all(), mean_sc
    assert (scaled_sc >= 0).all() and (scaled_sc <= 1).all(), scaled_sc
    assert (mean_scores[N == 0] == 0).all()
    assert (mean_scores[nonignore.sum((1, 2, 3)) == 0] == 0).all()
    assert (scaled_scores[N == 0] == 0).all()
    assert (scaled_scores[nonignore.sum((1, 2, 3)) == 0] == 0).all()

    # Return mean over batch dimension
    return mean_sc.mean(0), scaled_sc.mean(0)

#----------------------------------------------------------------------------

def np_iou_binary(mask_A, mask_B, debug=False):
    if debug:
        assert mask_A.shape == mask_B.shape
        assert mask_A.dtype == np.bool
        assert mask_B.dtype == np.bool
    intersection = (mask_A * mask_B).sum((1, 2, 3)).astype(np.float)
    union = (mask_A + mask_B).sum((1, 2, 3)).astype(np.float)
    # Return -100 if union is zero, else return IOU
    # return np.where(union == 0, -100., intersection / union)
    return intersection / (union + 1e-8)

#----------------------------------------------------------------------------

def np_average_segcover(segA, segB, ignore_background=False):
    """
    Covering of segA by segB
    segA.shape = [batch size, 1, img_dim1, img_dim2]
    segB.shape = [batch size, 1, img_dim1, img_dim2]

    scale: If true, take weighted mean over IOU values proportional to the
           the number of pixels of the mask being covered.

    Assumes labels in segA and segB are non-negative integers.
    Negative labels will be ignored.
    """

    assert segA.shape == segB.shape, f"{segA.shape} - {segB.shape}"
    assert segA.shape[1] == 1 and segB.shape[1] == 1
    bsz = segA.shape[0]
    nonignore = (segA >= 0)

    mean_scores = np.zeros((bsz,), dtype=np.float)
    N = np.zeros((bsz,), dtype=np.float)
    scaled_scores = np.zeros((bsz,), dtype=np.float)
    scaling_sum = np.zeros((bsz,), dtype=np.float)

    # Find unique label indices to iterate over
    if ignore_background:
        iter_segA = np.unique(segA[segA > 0]).tolist()
    else:
        iter_segA = np.unique(segA[segA >= 0]).tolist()
    iter_segB = np.unique(segB[segB >= 0]).tolist()
    # Loop over segA
    for i in iter_segA:
        binaryA = (segA == i)
        if not binaryA.any():
            continue
        max_iou = np.zeros((bsz,), dtype=np.float)
        # Loop over segB to find max IOU
        for j in iter_segB:
            # Do not penalise pixels that are in ignore regions
            binaryB = (segB == j) * nonignore
            if not binaryB.any():
                continue
            iou = np_iou_binary(binaryA, binaryB)
            max_iou = np.where(iou > max_iou, iou, max_iou)
        # Accumulate scores
        mean_scores += max_iou
        N = np.where(binaryA.sum((1, 2, 3)) > 0, N+1, N)
        scaled_scores += binaryA.sum((1, 2, 3)).astype(np.float) * max_iou
        scaling_sum += binaryA.sum((1, 2, 3))

    # Compute coverage
    mean_sc = mean_scores / np.maximum(N, 1).astype(np.float)
    scaled_sc = scaled_scores / np.maximum(scaling_sum, 1).astype(np.float)

    # Sanity check
    assert (mean_sc >= 0).all() and (mean_sc <= 1).all(), mean_sc
    assert (scaled_sc >= 0).all() and (scaled_sc <= 1).all(), scaled_sc
    assert (mean_scores[N == 0] == 0).all()
    assert (mean_scores[nonignore.sum((1, 2, 3)) == 0] == 0).all()
    assert (scaled_scores[N == 0] == 0).all()
    assert (scaled_scores[nonignore.sum((1, 2, 3)) == 0] == 0).all()

    # Return mean over batch dimension
    return mean_sc.mean(0), scaled_sc.mean(0)

#----------------------------------------------------------------------------
