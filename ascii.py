import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# local imports
from conversion import text_image


def char_brightness(char_image: np.ndarray, weight: int):
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
    brightness_arr = []
    brightness_arr.append(np.mean(char_image[:7]))  # TOP MIDDLE
    brightness_arr.append(np.mean(char_image[:, :5]))   # MID LEFT
    brightness_arr.append(np.mean(char_image[6:16, 5:13]))  # CENTER
    brightness_arr.append(np.mean(char_image[:, 13:]))   # MID RIGHT
    brightness_arr.append(np.mean(char_image[15:]))  # BOTTOM MIDDLE
    brightness_arr += weight*[np.mean(char_image)]  # WHOLE IMAGE
    return brightness_arr


def char_palate(chars,weight=4, color=1):
    """
    Function
      Render each character from the provided palate in the provided font
      Then map each rendered image to a lower dimensional vector
    
    """
    brightness = []
    file_dir,_ = os.path.split(__file__)
    font_path = os.path.join(file_dir, "fonts", "Menlo-Regular.ttf")
    font = ImageFont.truetype(font_path,size=16)
    for char in chars:
        img = Image.new(mode='1', size=(18, 22), color=(color))
        d = ImageDraw.Draw(img)
        d.text((0, -6), char, fill=(1-color), font=font)
        pix_val = np.asarray(img)
        brightness.append(char_brightness(pix_val, weight))
    char_array = np.array(brightness)
    char_index = dict(zip(np.arange(len(brightness)),chars))
    # return dict(zip(chars,brightness))
    return char_array, char_index


def distance(x, y, p=1):
    '''
    Parameters
    ----------
    x : int, float, np.array (1-dim)
        Scalar or Vector quantity
    y : int, float, np.array (1-dim)
        Scalar or Vector quantity
    p : int (optional)
        Determines the order/type of distance metric to be returned.
        The default is 2 (Euclidean Distance)

    Returns
    -------
    dist : np.array
        The element-wise distance between the inputs

    '''
    # Convert inputs to np.arrays
    if type(x) != np.ndarray:
        x = np.array([x])
    if type(y) != np.ndarray:
        y = np.array([y])
    
    # Perform the distance computation
    temp = np.abs(y - x) ** p
    dist = np.sum(temp, axis=len(temp.shape)-1) ** (1/p)
    return dist


def get_neighbors(image: np.ndarray, x: int, y: int, weight: int=4):
    nb = [image[x-1, y], image[x, y-2],   image[x, y],   image[x, y+2], image[x+1, y]]
    nb += weight*[image[x, y]]
    return(nb)

def load_image(path, w=100):
    image = Image.open(path)
    width, height = image.size
    image = image.convert('L') # convert the image to monochrome
    k = image.size[0]/w  # /400
    image = image.resize((int(width/k), int(height/(2*k))))
    return np.array(image), (width, height)


def to_ascii(image: np.ndarray, weight=4, exposure=0.35, color='b', size=(750, 1334)) -> np.ndarray:
    palate = [' ','.',"'",'-','\\','/',',','_',':','=','^','"','+','•','~',';','|','(',')','<','>','%','?','c','s','{','}','!','I','[',']','i','t','v','x','z','1','r','a','e','l','o','n','u','T','f','w','3','7','J','y','5','$','4','g','k','p','q','F','P','b','d','h','G','O','V','X','E','Z','8','A','U','D','H','K','W','&','@','R','B','Q','#','0','M','N']
    c = 1 if color == 'b' else 0
    e = exposure if color == 'b' else 1/exposure
    pix_val = (np.asarray(image)/255)**e
    char_array, char_index = char_palate(palate, weight=weight, color=c)

    lines = []
    current_line = ''
    for x in range(2,image.shape[0]-1):
        for y in range(2,image.shape[1]-2):
            local_pixels = get_neighbors(pix_val,x,y, weight)
            current_char_index = np.argmin(distance(local_pixels, char_array))
            current_char = char_index[current_char_index]
            current_line += current_char
        lines.append(current_line)
        current_line = ''
    return text_image(lines, size)    


def main(input_path, output_path, dimension=100, weight=4, exposure=0.35, color='b'):

    # output_path = "output/ascii_test_" + os.path.basename(input_path)
    original_image, size = load_image(input_path, dimension)
    ascii_image = to_ascii(original_image, weight, exposure, color, size)
    image = Image.fromarray(ascii_image)
    image.save(output_path)



if __name__ == "__main__":

    char = [' ','.',"'",'-','\\','/',',','_',':','=','^','"','+','•','~',';','|','(',')','<','>','%','?','c','s','{','}','!','I','[',']','i','t','v','x','z','1','r','a','e','l','o','n','u','T','f','w','3','7','J','y','5','$','4','g','k','p','q','F','P','b','d','h','G','O','V','X','E','Z','8','A','U','D','H','K','W','&','@','R','B','Q','#','0','M','N']
    math_char = ['1','2','3','4','5','6','7','8','9','0',',','.','*','%','#','!','/','<','>','~','^','&','(',')','[',']','{','}','x','y','z','ε','θ','π','Σ','∂','∫','µ','≤','≥','≈','√','±','λ',"'",' ']

    parser = argparse.ArgumentParser(description='Cleaning audio data')
    parser.add_argument('--input_path', '-i', type=str, default='dataset/3D91EB5C-DD7E-47A8-856E-BD7EC8038FC9.JPG',
                        help='Filepath to the desired image')
    parser.add_argument('--dimension', '-d',type=int, default=150,
                        help='Number of characters to scale the width/resolution to')
    parser.add_argument('--exposure', '-e', type=float, default=0.35,
                        help='How much exposure is added (default 0.50), the closer to zero the brighter the image')
    parser.add_argument('--weight', '-w', type=int, default=15,
                        help='How much individual pixels are weighted, default 15')
    parser.add_argument('--color', '-c', type=str, default='b',
                        help='Color the characters w (white) or b (black)')
    args, _ = parser.parse_known_args()

    main(**vars(args))
