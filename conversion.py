import os
import PIL
import numpy as np
from PIL import Image, ImageFont, ImageOps, ImageDraw

PIXEL_ON = 0  # PIL color to use for "on"
PIXEL_OFF = 255  # PIL color to use for "off"
color = (0, 255, 0)
color = PIXEL_ON


def text_image(lines, size):
    """
    Convert text file to a grayscale image 
      input_path: path to load text file
      output_path: path to save image
    """

    # Load font
    file_dir,_ = os.path.split(__file__)
    font_path = os.path.join(file_dir, "fonts", "Menlo-Regular.ttf")
    font = ImageFont.truetype(font_path,size=16)

    # Make the background image based on the combination of font and lines
    pt2px = lambda pt: int(round(pt * 96.0 / 72))  # convert points to pixels
    max_width_line = max(lines, key=lambda s: font.getsize(s)[0])

    # Max height is adjusted down because it's too large visually for spacing
    test_string = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    max_height = pt2px(font.getsize(test_string)[1])
    max_width = pt2px(font.getsize(max_width_line)[0])
    height = max_height * len(lines)  # perfect or a little oversized
    width = int(round(max_width + 40))  # a little oversized
    image = PIL.Image.new("L", (width, height), color=PIXEL_OFF)
    draw = PIL.ImageDraw.Draw(image)

    # Draw each line of text
    vertical_position = 5
    horizontal_position = 5
    line_spacing = int(round(max_height * 0.8))  # reduced spacing seems better
    for line in lines:
        draw.text((horizontal_position, vertical_position),
                  line, fill=color, font=font)
        vertical_position += line_spacing

    # Crop the text
    c_box = ImageOps.invert(image).getbbox()
    image = image.crop(c_box)
    image = image.resize(size)
    # print(width, height)
    # print(size)
    return np.array(image)


test = 0