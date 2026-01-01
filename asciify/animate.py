import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import trange

from asciify.memory import AsciiMemory
from asciify.utils.sampler import MoldFrontierSampler
from asciify.utils.preprocess import expose_contrast, reshape_image 


def animate_image(
        path:str, 
        memory:AsciiMemory,
        exposure:float,
        contrast:float,
        brightness:float,
        chars_per_line:int
    ) -> None:
    output_path, extension = os.path.splitext(path)
    output_path = output_path + "-ascii.mp4"
    input_path = path

    img = Image.open(input_path)
    img = reshape_image(img, chars_per_line, memory)
    frame = torch.Tensor(np.array(img) / 255)
    frame = expose_contrast(frame, exposure, contrast, brightness=brightness)
    ascii_frame = memory(frame)

    ## GLYPHS
    char_idxs = memory.char_idxs
    patches = torch.Tensor(np.asarray(img).reshape(
        char_idxs.shape[0], memory.glyph_h,  
        char_idxs.shape[1], memory.glyph_w,        
    ).copy()).to(torch.uint8).transpose(2, 1)
    glyphs = memory.char_values[char_idxs]

    ## VIDEO WRITER
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    output_path = output_path.replace("png","mp4")
    fps = 30    
    h, w = frame.shape[:2]
    size = (w, h)
    writer = cv2.VideoWriter(output_path, fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(
            "Could not open VideoWriter. Your OpenCV build may lack an MP4 backend/codec."
        )

    ## ANIMATION
    # scan(patches, glyphs, char_idxs, memory, writer, fps)
    # splat(patches, glyphs, char_idxs, memory, writer, fps)
    spread(patches, glyphs, char_idxs, memory, writer, fps)

    print(f"Video saved to {output_path}")    
    return 


def scan(patches, glyphs, char_idxs, memory, writer, fps):

    for col in trange(char_idxs.shape[1]):

        patches[:,col] = glyphs[:,col]
        frame = patches.transpose(2, 1).reshape(
            char_idxs.shape[0] * memory.glyph_h,
            char_idxs.shape[1] * memory.glyph_w,
        ).numpy()

        ## WRITE FRAME
        ascii_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        writer.write(np.ascontiguousarray(ascii_bgr, dtype=np.uint8))

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    return


def splat(patches, glyphs, char_idxs, memory, writer, fps):
    ## RANDOMLY SAMPLE AND REPLACE `k` PATCHES PER STEP WITH ASCII
    frames = 150
    p = len(char_idxs.view(-1))
    k = p // frames
    m, n = char_idxs.shape
    flat = np.arange(p)
    rng = np.random.default_rng()
    rng.shuffle(flat)

    for i in trange(frames + 2*fps):

        ## PAUSE BEFORE AND AFTER CONVERSION
        if i > 0.5*fps and len(flat) > 0:
            ## SAMPLE RANDOM GLYPHS
            idx = min(k, len(flat))
            rows, cols = np.unravel_index(flat[:idx], (m, n))
            flat = flat[idx:]
            ## STITCH TILES -> (rows*gh, cols*gw)
            patches[rows,cols] = glyphs[rows,cols]

        ## BUILD FRAME
        frame = patches.transpose(2, 1).reshape(
            char_idxs.shape[0] * memory.glyph_h,
            char_idxs.shape[1] * memory.glyph_w,
        ).numpy()

        ## WRITE FRAME
        ascii_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        writer.write(np.ascontiguousarray(ascii_bgr, dtype=np.uint8))

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    return


def spread(patches, glyphs, char_idxs, memory, writer, fps):
    ## RANDOMLY SAMPLE AND REPLACE `k` PATCHES PER STEP WITH ASCII
    m, n = char_idxs.shape
    seed=(m//2, n//2)
    # seed=(1,1)

    sampler = MoldFrontierSampler(
        char_idxs.shape,
        seed_rc=seed,
        fraction=1/3,    
    )
    
    i = j = 0
    while not sampler.done() or j < fps:

        ## PAUSE BEFORE AND AFTER CONVERSION
        if i > 0.5*fps and not sampler.done():
            ## SAMPLE RANDOM GLYPHS
            ## STITCH TILES -> (rows*gh, cols*gw)
            idxs = sampler.step() 
            patches[idxs[:, 0], idxs[:, 1]] = glyphs[idxs[:, 0], idxs[:, 1]]
            patches[seed] = glyphs[seed]
        elif sampler.done():
            j+=1

        ## BUILD FRAME
        frame = patches.transpose(2, 1).reshape(
            char_idxs.shape[0] * memory.glyph_h,
            char_idxs.shape[1] * memory.glyph_w,
        ).numpy()

        ## WRITE FRAME
        ascii_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        writer.write(np.ascontiguousarray(ascii_bgr, dtype=np.uint8))

        i+=1

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    return

