import os
import torch
import numpy as np
from PIL import Image

from ascii.preprocess import expose_contrast, reshape_image
from ascii.memory import AsciiMemory


def convert_image(
        path:str, 
        memory:AsciiMemory,
        exposure:float,
        contrast:float,
        brightness:float,
        chars_per_line:int,
        output_mode:str="image"
    ) -> None:    
    output_path, extension = os.path.splitext(path)
    output_path = output_path + "-ascii" + extension
    output_mode = output_mode
    input_path = path

    img = Image.open(input_path)
    img = reshape_image(img, chars_per_line, memory)
    frame = torch.Tensor(np.array(img) / 255)
    # frame = lift_shadows(frame)
    frame = expose_contrast(
        frame, 
        exposure, 
        contrast, 
        0.3,
        brightness)
    ascii_frame = memory(frame)  # (rows, cols)
    # Image.fromarray((frame*255).to(torch.uint8).numpy()).save(output_path.replace("-ascii.","-base."))

    if output_mode == "text":
        # emit text file (optional: convert indices to chars)
        with open(output_path, "w") as f:
            for row in memory.char_idxs:
                f.write("".join(memory.palette[row]))
                f.write("\n")
        return

    print(f"Image saved to {output_path}")
    return Image.fromarray(ascii_frame).save(output_path)
    