import os
import cv2
import argparse

from PIL import Image, ImageFont, UnidentifiedImageError

from ascii.memory import AsciiMemory
from ascii.image import convert_image
from ascii.video import convert_video



FONTS = {
    "Menlo": "fonts/Menlo-Regular.ttf",
}

def main():

    # PARSE ARGUMENTS
    project_root = os.path.dirname(os.path.dirname(__file__))
    default_path = os.path.join(project_root, "content/test-image.png")

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-p', '--path', type=str, default=default_path, 
                        help='Path to the image or video to be converted')
    parser.add_argument('-ap', '--ascii_palette', type=str, default="default",
                        help='Name of a pre-defined ASCII characters palette to build the image from, in `ascii/palettes.py`.')
    parser.add_argument('-ac', '--ascii_chars', type=str, default=None,
                        help='A string of characters to build the image from. Note, this overrides the --ascii_palette selection.')
    parser.add_argument('-f', '--font', type=str, default="Menlo", 
                        help='Options: [`Menlo`]. Assuming a monospace font stored in ascii/fonts/ ')
    parser.add_argument('-c', '--color', type=str, default='b', 
                        help='Options: [`w`, `b`]. Color the characters `w` white or `b` black.')
    parser.add_argument('-m', '--output_mode', type=str, default="default",
                        help='Options: [`text`, `default`]. Save the result to a .txt file or default to the input file type.')
    parser.add_argument('-r', '--chars_per_line', type=int, default=200, 
                        help='Resolution of the output image in number of characters per line.')
    parser.add_argument('-b', '--brightness', type=float, default=0.0, 
                        help='Control image exposure prior to conversion, the closer to zero the brighter the image.')
    parser.add_argument('-e', '--exposure', type=float, default=1.0, #0.25, 
                        help='Control image exposure prior to conversion, the closer to zero the brighter the image.')
    parser.add_argument('-con', '--contrast', type=float, default=1.0, 
                        help='Control image exposure prior to conversion, the closer to zero the brighter the image.')
    parser.add_argument('-w', '--weight', type=float, default=10, 
                        help='Control the weighting of individual pixels vs surrounding neighbors.')
    parser.add_argument('-fs', '--font_size', type=int, default=12, 
                        help='Font size.')
    args = parser.parse_args()

    ## VALIDATE INPUT FILE
    assert os.path.exists(args.path), f"File not found: {args.path}"
    
    filetype = get_filetype(args.path)
    assert filetype != "unknown", \
        f"Invalid filetype {os.path.basename(args.path)}, could not be loaded as an image or video."

    font_path = os.path.join(project_root, "ascii", FONTS[args.font])
    font = ImageFont.truetype(font_path, size=args.font_size) 

    ## BUILD MEMORY FROM ASCII PALATE
    memory = AsciiMemory(
        ascii_palette=args.ascii_palette,
        ascii_chars=args.ascii_chars,
        weight=args.weight,
        color=args.color,
        font=font
    )
    conversion_args = {
        "path":args.path,
        "memory":memory,
        "exposure":args.exposure,
        "contrast":args.contrast,
        "brightness":args.brightness,
        "chars_per_line":args.chars_per_line,
    }

    ## CONVERT INPUT FILE
    if filetype == "image":
        convert_image(**conversion_args, output_mode=args.output_mode)
    elif filetype == "video":
        convert_video(**conversion_args)
    return


def get_filetype(path:str):
    ## IS THIS A VALID IMAGE FILE?
    try:
        with Image.open(path) as im:
            im.load()  
        return "image"
    except (UnidentifiedImageError, OSError, ValueError):
        pass

    ## IS THIS A VALID VIDEO FILE?
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return "invalid"
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return "video"
        return "invalid"
    finally:
        cap.release()



if __name__ == "__main__":
    main()
