import os
import cv2
import argparse
from PIL import Image, ImageFont, UnidentifiedImageError

from ascii.converter import AsciiConverter



FONTS = {
    "Menlo": "fonts/Menlo-Regular.ttf",
}

def get_filetype(path:str):
    ## IS THIS A VALID IMAGE FILE?
    try:
        with Image.open(path) as im:
            im.load()  # forces decoding; catches many corrupt/truncated files
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



def main():

    # PARSE ARGUMENTS
    project_root = os.path.dirname(os.path.dirname(__file__))
    # default_path = os.path.join(project_root, "content/einstein.png")
    # default_path = os.path.join(project_root, "content/obama.png")
    default_path = os.path.join(project_root, "content/test-image.png")
    # default_path = os.path.join(project_root, "content/test-video.mp4")
    # default_palatte = [
    #     ' ','.',"'",'-','\\','/',',','_',':','=','^','"','+','•','~',';','|','(',')',
    #     '<','>','%','?','c','s','{','}','!','I','[',']','i','t','v','x','z','1','r',
    #     'a','e','l','o','n','u','T','f','w','3','7','J','y','5','$','4','g','k','p',
    #     'q','F','P','b','d','h','G','O','V','X','E','Z','8','A','U','D','H','K','W',
    #     '&','@','R','B','Q','#','0','M','N'
    # ]

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-p', '--path', type=str, default=default_path, 
                        help='Path to the image or video to be converted')
    parser.add_argument('-ap', '--ascii_palatte', type=str, default="default",
                        help='Name of a pre-defined ASCII characters palatte to build the image from, in `ascii/palattes.py`.')
    parser.add_argument('-ac', '--ascii_chars', type=str, default=None,
                        help='A string of characters to build the image from. Note, this overrides the --ascii_palatte selection.')
    parser.add_argument('-f', '--font', type=str, default="Menlo", 
                        help='Options: [`Menlo`]. Assuming a monospace font stored in ascii/fonts/ ')
    parser.add_argument('-c', '--color', type=str, default='b', 
                        help='Options: [`w`, `b`]. Color the characters `w` white or `b` black.')
    parser.add_argument('-m', '--output_mode', type=str, default="default",
                        help='Options: [`text`, `default`]. Save the result to a .txt file or default to the input file type.')
    parser.add_argument('-d', '--chars_per_line', type=int, default=200, 
                        help='Width of the output image in number of characters.')
    parser.add_argument('-b', '--brightness', type=float, default=1.1, 
                        help='Control image exposure prior to conversion, the closer to zero the brighter the image.')
    parser.add_argument('-e', '--exposure', type=float, default=0.15, 
                        help='Control image exposure prior to conversion, the closer to zero the brighter the image.')
    parser.add_argument('-con', '--contrast', type=float, default=1.0, 
                        help='Control image exposure prior to conversion, the closer to zero the brighter the image.')
    parser.add_argument('-w', '--weight', type=float, default=20, 
                        help='Control the weighting of individual pixels vs surrounding neighbors.')
    # parser.add_argument('-fs', '--font_size', type=int, default=24, 
    #                     help='Font size.')
    args = parser.parse_args()

    ## VALIDATE INPUT FILE
    assert os.path.exists(args.path), f"File not found: {args.path}"
    
    filetype = get_filetype(args.path)
    assert filetype != "unknown", \
        f"Invalid filetype {os.path.basename(args.path)}, could not be loaded as an image or video."

    font_path = os.path.join(project_root, "ascii", FONTS[args.font])
    args.font = ImageFont.truetype(font_path, size=24)
    converter = AsciiConverter(**vars(args))

    ## CONVERT INPUT FILE
    if filetype == "image":
        converter.convert_image()
        print(f"Image saved to {converter.output_path}")
    elif filetype == "video":
        converter.convert_video()
        print(f"Video saved to {converter.output_path}")
    return


if __name__ == "__main__":
    main()
