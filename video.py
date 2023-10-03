import os
import cv2
import numpy as np
from PIL import Image
from tqdm import trange
from ascii import AsciiConverter



def process_frame(frame, chars_per_line):
    image = Image.fromarray(frame)
    width, height = image.size
    image = image.convert('L')
    k = image.size[0]/chars_per_line  
    image = image.resize((int(width/k), int(height/(2*k))))
    return np.array(image)


pth = 'b2a924472c4b43a1a70563637c6ad063.MOV'
pth = 'video_input/test.MOV'
pth = 'dolorian.MP4'
pth = 'video_input/korosh.MP4'
pth = 'video_input/watergate.mov'
# pth = 'fireworks2.MOV'
cap = cv2.VideoCapture(pth)

filename = os.path.splitext(pth)[0].split("/")[-1]
filename = f"video/ascii_{filename.split('.')[0]}"
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 30
chars_per_line = 165

converter = AsciiConverter(character_width=chars_per_line, color="w", weight=5, exposure=0.35)

for i in trange(frame_count):
    frames_left, frame = cap.read()

    if not frames_left:
        break

    grayscale_frame = process_frame(frame, chars_per_line)

    ## KEEP SIZES INTERNAL TO CLASS
    if i==0:
        ascii_frame = converter.pixels_to_characters(grayscale_frame, size)
        size = ascii_frame.shape
        out = cv2.VideoWriter(f"{filename}.mp4", fourcc, fps, size)
    else:
        ascii_frame = converter.pixels_to_characters(grayscale_frame, size)
    
    ## 
    ascii_frame = cv2.cvtColor(ascii_frame, cv2.COLOR_GRAY2BGR)


    # print(ascii_frame.shape)
    # print(size)
    out.write(ascii_frame)
    # print(f"{i+1} / {frame_count} Frames Rendered")

out.release()
cap.release()
print(f"done")

