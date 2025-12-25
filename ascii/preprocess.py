import torch


def expose_contrast(x: torch.Tensor, exposure: float = 1.0, contrast: float = 1.0, mid: float = 0.5, brightness=1.0):
        """
        x: float tensor in [0,1]
        exposure: gamma-like (your current behavior): x ** exposure
        contrast: >1 increases contrast, <1 decreases
        mid: pivot point for contrast (0.5 is typical)
        """
        if brightness != 1.0:
            x = x * brightness
        if contrast != 1.0:
            x = (x - mid) * contrast + mid
        if exposure != 1.0:
            x = x ** exposure
        return x.clamp(0, 1)


def autocontrast_percentile(x: torch.Tensor, lo=0.01, hi=0.99, eps=1e-6):
    """
    x: float tensor in [0,1]
    lo/hi: percentiles (e.g., 1% and 99%)
    """
    flat = x.flatten()
    v_lo = torch.quantile(flat, lo)
    v_hi = torch.quantile(flat, hi)
    x = (x - v_lo) / (v_hi - v_lo + eps)
    return x.clamp(0, 1)


def sigmoid_contrast(x: torch.Tensor, strength: float = 6.0, mid: float = 0.5):
    ## strength ~ 4–10; higher = more contrast around mid
    return torch.sigmoid((x - mid) * strength)

