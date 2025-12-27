import os
import cv2
import torch
import numpy as np
from tqdm import trange

from ascii.preprocess import expose_contrast, reshape_image, crop_frame
from ascii.memory import AsciiMemory


def convert_video(
        path:str, 
        memory:AsciiMemory,
        exposure:float,
        contrast:float,
        brightness:float,
        chars_per_line:int
    ) -> None:
    output_path, extension = os.path.splitext(path)
    output_path = output_path + "-ascii" + extension
    input_path = path

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        fps = 30.0

    out = None
    size = None  # (w, h)

    for _ in trange(frame_count):
        ok, img = cap.read()
        if not ok or img is None:
            break

        img = reshape_image(img, chars_per_line, memory)
        frame = torch.Tensor(np.array(img) / 255)
        # frame = lift_shadows(frame)
        frame = expose_contrast(frame, exposure, contrast, brightness=brightness)
        ascii_frame = memory(frame)
        h, w = ascii_frame.shape[:2]

        if out is None:
            size = (w, h)
            out = cv2.VideoWriter(output_path, fourcc, fps, size)
            if not out.isOpened():
                cap.release()
                raise RuntimeError(
                    "Could not open VideoWriter. Your OpenCV build may lack an MP4 backend/codec."
                )

        if (w, h) != size:
            ascii_frame = crop_frame(size, ascii_frame)

        # most portable: write as BGR
        ascii_bgr = cv2.cvtColor(ascii_frame, cv2.COLOR_GRAY2BGR)
        out.write(np.ascontiguousarray(ascii_bgr))

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {output_path}")    
    return    