import os
import PIL
import cv2
import time
import torch
import argparse
import numpy as np
from tqdm import trange
from PIL import Image, ImageDraw, ImageFont, ImageOps


class AsciiConverter:

    def __init__(self, characters_per_line=200, color='w', weight=5, exposure=0.35, font='fonts/Menlo-Regular.ttf'):
        # palate = [' ','.',"'",'-','\\','/',',','_',':','=','^','"','+','•','~',';','|','(',')','<','>','%','?','c','s','{','}','!','I','[',']','i','t','v','x','z','1','r','a','e','l','o','n','u','T','f','w','3','7','J','y','5','$','4','g','k','p','q','F','P','b','d','h','G','O','V','X','E','Z','8','A','U','D','H','K','W','&','@','R','B','Q','#','0','M','N']
        self.characters = np.array([' ','.',"'",'-','\\','/',',',':','=','^','"','+','•','~',';','|','(',')','<','>','%','?','c','s','{','}','!','I','[',']','i','t','v','x','z','1','r','a','e','l','o','n','u','T','f','w','3','7','J','y','5','$','4','g','k','p','q','F','P','b','d','h','G','O','V','X','E','Z','8','A','U','D','H','K','W','&','@','R','B','Q','#','0','M','N'])
        self.weight = weight
        self.exposure = exposure if color == 'w' else 1 / exposure
        self.font = ImageFont.truetype(font, size=24)
        # self.font = ImageFont.truetype(font, size=30)
        self.color = 1 if color == 'w' else 0
        self.background_color = self.color * 255
        self.character_color = (not self.color) * 255
        self.characters_per_line = characters_per_line

        ## CONSTANTS FOR RENDERING
        pt2px = lambda pt: int(round(pt * 96.0 / 72))  # convert points to pixels
        max_width_line = " " * characters_per_line
        self.max_width = pt2px(self.font.getsize(max_width_line)[0])
        self.max_height = pt2px(self.font.getsize(max_width_line)[1])
        self.width = int(round(self.max_width))  # a little oversized
        # self.height = self.max_height * len(lines)  # perfect or a little oversized
        self.height = None
        self.font = ImageFont.truetype(font, size=30)

        ## RENDER EACH CHARACTER AND EMBED THEM IN MANUALLY SPECIFIED FEATURES SPACE
        self.character_embeddings = self.get_character_embeddings()

        ## EXTRACT IMAGE FEATURES VIA HAND-CRAFTED CONVOLUTION, MAP TO NEAREST CHAR EMBEDDING
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=6, stride=1, padding=2)
        self.conv.weight = torch.nn.Parameter(torch.Tensor([[
            [
                [0,    0,    0,    0,    0],
                [0,    0,    1,    0,    0],
                [0,    0,    0,    0,    0],
                [0,    0,    0,    0,    0],
                [0,    0,    0,    0,    0],
            ],
            [
                [0,    0,    0,    0,    0],
                [0,    0,    0,    0,    0],
                [1,    0,    0,    0,    0],
                [0,    0,    0,    0,    0],
                [0,    0,    0,    0,    0],
            ],
            [
                [0,    0,    0,    0,    0],
                [0,    0,    0,    0,    0],
                [0,    0,    1,    0,    0],
                [0,    0,    0,    0,    0],
                [0,    0,    0,    0,    0],
            ],
            [
                [0,    0,    0,    0,    0],
                [0,    0,    0,    0,    0],
                [0,    0,    0,    0,    1],
                [0,    0,    0,    0,    0],
                [0,    0,    0,    0,    0],
            ],
            [
                [0,    0,    0,    0,    0],
                [0,    0,    0,    0,    0],
                [0,    0,    0,    0,    0],
                [0,    0,    1,    0,    0],
                [0,    0,    0,    0,    0],
            ],
            [
                [0,    0,    0,    0,    0],
                [0,    0,    0,    0,    0],
                [0,    0,  weight, 0,    0],
                [0,    0,    0,    0,    0],
                [0,    0,    0,    0,    0],
            ],
        ]]).transpose(1,0), requires_grad=False)


    def char_brightness(self, char_image: np.ndarray):
        '''
        Function:
        Map each character from a 18x22 char image to a length (5 + weight) vector by
        average pooling significant regions of the image

        Input
        char_image: np.ndarray -> a 22x18 pixel map of a char belonging to a given font
        weight: int -> how much to weight the darkness of the character

        Output:
        brightness_arr: list -> mapping of the character to a lower dimensional list to be 
                                compared with local pixel neighborhoods in the target image 

        '''
        brightness_arr = [
            np.mean(char_image[:7]),         # TOP MIDDLE
            np.mean(char_image[:, :5]),      # MID LEFT
            np.mean(char_image[6:16, 5:13]), # CENTER
            np.mean(char_image[:, 13:]),     # MID RIGHT
            np.mean(char_image[15:]),        # BOTTOM MIDDLE
            self.weight*np.mean(char_image),      # MEAN IMAGE BRIGHTNESS
        ]
        return brightness_arr


    def get_character_embeddings(self):
        """
        Function
        Render each character from the provided palate in the provided font
        Then map each rendered image to a lower dimensional vector
        
        """
        brightness = []
        for char in self.characters:
            image = Image.new(mode='1', size=(18, 22), color=(self.color))
            # img = Image.new(mode='1', size=(18, 26), color=(color))
            d = ImageDraw.Draw(image)
            d.text((0, -6), char, fill=(1-self.color), font=self.font)
            image = np.asarray(image)
            brightness.append(self.char_brightness(image))
        return torch.Tensor(brightness)


    def load_image(self, path):
        return  Image.open(path)
        
    def process_image(self, image:Image):
        ## RESIZE IMAGE FOR APPROPRIATE CHARACTER LINE WIDTH
        if not isinstance(image, Image.Image): image = Image.fromarray(image)
        image = image.convert('L')
        width, height = image.size
        ratio = width / self.characters_per_line
        image = image.resize((int(width/ratio), int(height/(2*ratio))))
        return np.array(image)


    def pixels_to_characters(self, image) -> np.ndarray:

        ## ADD EXPOSURE TO INPUT IMAGE
        image = (torch.Tensor(image)/255) ** self.exposure
        image_embedding = self.conv(image[None, None, ...])[0].permute(1,2,0)
        character_indices = torch.cdist(image_embedding, self.character_embeddings).argmin(dim=-1)
        output_characters = ["".join(self.characters[row]) for row in character_indices]

        return self.render(output_characters)    


    def render(self, lines):
        """
        Convert text file to a grayscale image 
        input_path: path to load text file
        output_path: path to save image
        """

        # Choose a font (you can see more detail in my library on github)
        # font = ImageFont.truetype('fonts/Menlo-Regular.ttf',size=24)#16)

        ## Format as UUID injection
        # numbers = "0123456789"
        # ids = [uuid.uuid1(random.randint(0,99999999999999)).urn[9:] for line in lines]
        # line_len = int(2*len(lines[0])/3)
        # for i in range(len(lines)):
        #   lines[i] = ids[i][:24] + "d" + lines[i][0:line_len] + f"{random.choice(numbers)}-b" + lines[i][line_len+3:] + f"{random.choice(numbers)}-" + ids[i][24:]
        #   if i==len(lines)-3:
        #     lines[i] = lines[i].replace(lines[i][-12:],"[g.brothers]")
        if self.height is None: 
            self.height = len(lines) * self.max_height
        ## Make the background image based on the combination of font and lines
        image = PIL.Image.new("L", (self.width, self.height), color=self.color)
        draw = PIL.ImageDraw.Draw(image)

        # Draw each line of text
        vertical_position = 5
        horizontal_position = 5
        line_spacing = int(round(self.max_height * 1.0))  # reduced spacing seems better
        for line in lines:
            draw.text((horizontal_position, vertical_position), line, fill=self.character_color, font=self.font)
            vertical_position += line_spacing

        # Crop the text
        c_box = PIL.ImageOps.invert(image).getbbox()
        image = image.crop(c_box)
        # if size != image.size: image = image.resize(size)
        # print(width, height)
        # print(size)
        return np.array(image)

    def convert_video(self, input_path="video_input/room.MOV"):
    # def convert_video(self, input_path="video_input/watergate.mov"):

        cap = cv2.VideoCapture(input_path)
        filename = os.path.splitext(input_path)[0].split("/")[-1]
        filename = f"video/ascii_{filename.split('.')[0]}"
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        fps = 30.0
        out = None

        for _ in trange(frame_count):
            frames_left, frame = cap.read()

            if not frames_left: 
                break

            grayscale_frame = self.process_image(frame)
            ascii_frame = converter.pixels_to_characters(grayscale_frame)
            ## ADD RGB CHANNELS FOR OUTPUT
            ascii_frame = cv2.cvtColor(ascii_frame, cv2.COLOR_GRAY2BGR)            
            # if ascii_frame.shape != frame.shape:
            #     ascii_frame = np.array(Image.fromarray(ascii_frame).resize(frame.transpose(1,0,2).shape[:-1]))
                # ascii_frame = ascii_frame.transpose(1,0,2)     
                # k=0      
            # print(ascii_frame.shape)
            # if out is None: out = cv2.VideoWriter(filename=f"{filename}.avi", apiPreference=0, fourcc=fourcc, fps=fps, frameSize=ascii_frame.shape[:2])
            # if out is None: out = cv2.VideoWriter(filename=f"{filename}.mp4", apiPreference=0, fourcc=fourcc, fps=fps, frameSize=ascii_frame.shape[:2])
            if out is None: out = cv2.VideoWriter(filename=f"{filename}.mp4", apiPreference=0, fourcc=fourcc, fps=fps, frameSize=(ascii_frame.shape[1], ascii_frame.shape[0]))

            # grayscale_frame = self.process_image(frame)
            # ascii_frame = converter.pixels_to_characters(grayscale_frame)

            out.write(ascii_frame)

        cap.release()        
        out.release()
        cv2.destroyAllWindows()
        out = None
        print(f"Video saved to {filename}.mp4")

    def convert_image(self, input_path):
        original_image = self.load_image(input_path)
        original_image = self.process_image(original_image)
        ascii_image = self.pixels_to_characters(original_image)
        image = Image.fromarray(ascii_image)
        output_path = "output/ascii_test_" + os.path.basename(input_path)
        image.save(output_path)


if __name__ == "__main__":

    char = [' ','.',"'",'-','\\','/',',','_',':','=','^','"','+','•','~',';','|','(',')','<','>','%','?','c','s','{','}','!','I','[',']','i','t','v','x','z','1','r','a','e','l','o','n','u','T','f','w','3','7','J','y','5','$','4','g','k','p','q','F','P','b','d','h','G','O','V','X','E','Z','8','A','U','D','H','K','W','&','@','R','B','Q','#','0','M','N']
    math_char = ['1','2','3','4','5','6','7','8','9','0',',','.','*','%','#','!','/','<','>','~','^','&','(',')','[',']','{','}','x','y','z','ε','θ','π','Σ','∂','∫','µ','≤','≥','≈','√','±','λ',"'",' ']

    parser = argparse.ArgumentParser(description='Cleaning audio data')
    parser.add_argument('--input_path', '-i', type=str, default='images/obama.jpg', #'images/kendall1.jpg',
                        help='Filepath to the desired image')
    # parser.add_argument('--palate', '-p', type=list, default=char,
    #                     help='Characters to build the image from, should be a python list')
    parser.add_argument('--dimension', '-d',type=int, default=100,#44,#150,
                        help='Number of characters to scale the width/resolution to')
    parser.add_argument('--exposure', '-e', type=float, default=0.35, #0.50,
                        help='How much exposure is added (default 0.50), the closer to zero the brighter the image')
    parser.add_argument('--weight', '-w', type=int, default=5, #15,
                        help='How much individual pixels are weighted, default 15')
    parser.add_argument('--color', '-c', type=str, default='b', #'b',
                        help='Color the characters w (white) or b (black)')
    args, _ = parser.parse_known_args()

    # main(**vars(args))

    converter = AsciiConverter(args.dimension, args.color, args.weight, args.exposure)
    # converter.convert_image(args.input_path)
    converter.convert_video()
