import torch


def expose_contrast(
        x: torch.Tensor, 
        exposure: float = 1.0, 
        contrast: float = 1.0, 
        mid: float = 0.3, 
        brightness=0.0
    ):
    """
    x: float tensor in [0,1]
    exposure: gamma-like (your current behavior): x ** exposure
    contrast: >1 increases contrast, <1 decreases
    mid: pivot point for contrast (0.5 is typical)
    """
    if brightness != 0.0:
        x = x * brightness + (1-brightness)
        # x = x + brightness
    if contrast != 1.0:
        # x = (x - mid) * contrast + mid
        x = sigmoid_contrast(x, strength=contrast, mid=mid)
    if exposure != 1.0:
        x = x ** exposure
    return x.clamp(0, 1)


def lift_shadows(x, lift=0.3, knee=0.4):
    """
    x in [0,1]. lift>0 brightens shadows. knee is where effect fades out.
    """
    # normalize shadows into [0,1]
    t = (x / knee).clamp(0, 1)
    # smoothstep for smooth transition
    t = t * t * (3 - 2 * t)
    # add lift mostly below knee
    y = x + lift * (1 - t) * (knee - x).clamp(min=0) / max(knee, 1e-6)    
    return y.clamp(0, 1)


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

