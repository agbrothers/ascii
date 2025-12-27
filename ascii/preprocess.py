import torch
import numpy as np
from PIL import Image



def required_in_size(out_size: int, k: int, p: int, s: int) -> int:
    # minimal input that produces exactly out_size (for dilation=1)
    return (out_size - 1) * s - 2 * p + k


def reshape_image(
        img:Image, 
        chars_per_line:int,
        memory
    ):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img = img.convert("L")

    W0, H0 = img.size

    ## GET CONV PARAMs
    kH, kW = memory.conv.kernel_size  
    pH, pW = memory.conv.padding      
    sH, sW = memory.conv.stride       

    ## DESIRED NUMBER OF CHARACTER COLUMNS FROM CONV OUTPUT
    W_out = int(chars_per_line)

    ## CHOOSE ROWS COUNT TO PRESERVE ASPECT RATIO IN GLYPH SPACE
    H_out = int(round(W_out * (memory.glyph_w / memory.glyph_h) * (H0 / W0)))
    H_out = max(1, H_out)

    ## COMPUTE REQUIRED INPUT SIZES SO CONV OUTPUT IS (H_out, W_out) 
    W_in = required_in_size(W_out, kW, pW, sW)
    H_in = required_in_size(H_out, kH, pH, sH)

    ## RESIZE GRAYSCALE IMAGE TO CONV INPUT SIZE
    img = img.resize((W_in, H_in), resample=Image.BILINEAR)
    return np.asarray(img)


def crop_frame(
        target_size:tuple, 
        frame:np.ndarray,
        background_color:int,
    ) -> np.ndarray:
    # target_size is (w, h)
    h, w = frame.shape[:2]

    # PAD/CROP HEIGHT
    if h < target_size[1]:
        pad_dim = target_size[1] - h
        pad = np.ones((pad_dim, target_size[0]), dtype=np.uint8) * background_color
        frame = np.concatenate((frame, pad), axis=0)
    elif h > target_size[1]:
        frame = frame[: target_size[1]]

    # PAD/CROP WIDTH
    if w < target_size[0]:
        pad_dim = target_size[0] - w
        pad = np.ones((target_size[1], pad_dim), dtype=np.uint8) * background_color
        frame = np.concatenate((frame, pad), axis=1)
    elif w > target_size[0]:
        frame = frame[:, : target_size[0]]

    h2, w2 = frame.shape[:2]
    assert (w2, h2) == target_size
    return frame


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

