import os
import cv2
import torch
import numpy as np
from tqdm import trange
from PIL import Image, ImageDraw, ImageFont

from ascii.kernels import KERNELS
from ascii.palattes import PALATTES
from ascii.preprocess import expose_contrast


class AsciiConverter:
    """
    Same public behavior as your original class, but the slow PIL-per-frame text rendering
    is replaced with a glyph atlas + NumPy blitting (fast).
    """

    def __init__(
        self,
        path: str,
        ascii_palatte: str,
        ascii_chars: str,
        chars_per_line: int,
        color: str,
        weight: float,
        brightness: float,
        exposure: float,
        contrast: float,
        font: ImageFont.FreeTypeFont,
        output_mode: str,
    ):
        # PATHING
        output_path, extension = os.path.splitext(path)
        self.output_path = output_path + "-ascii" + extension
        self.output_mode = output_mode
        self.input_path = path

        # PARAMETERS
        self.chars_per_line = chars_per_line
        if ascii_chars is not None:
            self.palatte = np.array([*ascii_chars])
        else:
            self.palatte = np.array(PALATTES[ascii_palatte.lower()])

        is_white = 1 if color == "w" else 0
        self.brightness = brightness
        self.exposure = exposure
        self.contrast = contrast
        self.weight = weight
        self.font = font

        self.character_color = is_white * 255
        self.background_color = (not is_white) * 255

        ## FAST RENDERING: GLYPH ATLASS + LOOKUP
        W_mx = H_mx = 0
        for char in self.palatte:
            x0, y0, x1, y1 = font.getbbox(char)
            W_mx = max(W_mx, x1 - x0)
            H_mx = max(H_mx, y1 - y0)
        self.glyph_w, self.glyph_h = W_mx, H_mx

        # build atlas tensors:
        # - glyphs: (N, H, W) uint8
        # - byte_to_idx: (256,) int16 mapping ASCII code -> index in glyphs
        self.glyphs = self.get_glyph_atlas()

        ## EMBED ASCII CHARACTERS
        self.char_embd = self.embd_char(font)

        ## CONVLUTION
        kernel = KERNELS["default"](weight)
        self.conv = torch.nn.Conv2d(
            in_channels=kernel.shape[1], 
            out_channels=kernel.shape[0], 
            kernel_size=kernel.shape[-2:], 
            stride=kernel.shape[-2:], 
            padding=2,
        )
        with torch.no_grad():
            self.conv.weight[:] = kernel
        return


    def char_kernel(self, char_img: np.ndarray):
        w_third = round(0.333*self.glyph_w)
        h_third = round(0.333*self.glyph_h)
        brightness_conv = [
            np.mean(char_img[:h_third]),                              # TOP MIDDLE
            np.mean(char_img[:, :w_third]),                           # MID LEFT
            np.mean(char_img[h_third-1:2*h_third+1, w_third-1:2*w_third+1]),  # CENTER
            np.mean(char_img[:, 2*w_third:]),                         # MID RIGHT
            np.mean(char_img[2*h_third:]),                            # BOTTOM MIDDLE
            self.weight * np.mean(char_img)                           # MEAN IMAGE BRIGHTNESS
        ]
        return brightness_conv


    def draw_palatte(self, fontsize=24):
        font = ImageFont.truetype("ascii/fonts/Menlo-Regular.ttf", size=fontsize)
        img = Image.new(mode="1", size=(self.glyph_w*len(self.palatte), self.glyph_h), color=self.background_color)
        d = ImageDraw.Draw(img)
        d.text((0, 0), "".join(self.palatte), fill=self.character_color, font=font)
        img.show()
        return


    def embd_char(self, font):
        brightness = []
        for char in self.palatte:
            img = Image.new(mode="1", size=(self.glyph_w, self.glyph_h), color=self.background_color)
            d = ImageDraw.Draw(img)
            d.text((0, 0), str(char), fill=self.character_color, font=font)
            img = np.asarray(img)
            brightness.append(self.char_kernel(img))
        return torch.Tensor(brightness)


    def reshape_image(self, img:Image):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("L")

        W0, H0 = img.size

        ## GET CONV PARAMs
        kH, kW = self.conv.kernel_size  
        pH, pW = self.conv.padding      
        sH, sW = self.conv.stride       

        ## DESIRED NUMBER OF CHARACTER COLUMNS FROM CONV OUTPUT
        W_out = int(self.chars_per_line)

        ## CHOOSE ROWS COUNT TO PRESERVE ASPECT RATIO IN GLYPH SPACE
        H_out = int(round(W_out * (self.glyph_w / self.glyph_h) * (H0 / W0)))
        H_out = max(1, H_out)

        ## COMPUTE REQUIRED INPUT SIZES SO CONV OUTPUT IS (H_out, W_out) 
        W_in = required_in_size(W_out, kW, pW, sW)
        H_in = required_in_size(H_out, kH, pH, sH)

        ## RESIZE GRAYSCALE IMAGE TO CONV INPUT SIZE
        img = img.resize((W_in, H_in), resample=Image.BILINEAR)
        return np.asarray(img)


    def get_glyph_atlas(self) -> np.ndarray:
        ## MAP BYTE -> PALATTE INDEX (default to 0)
        self.byte_to_idx = np.zeros(256, dtype=np.int16)

        ## ENSURE CHARACTERS ARE LENGTH 1 STRINGS
        chars = [str(c) for c in self.palatte.tolist()]
        for i, ch in enumerate(chars):
            o = ord(ch)
            if 0 <= o < 256:
                self.byte_to_idx[o] = i

        ## RENDER EACH GLYPH ONCE INTO A SMALL TILE: (N, H, W)
        glyphs = np.empty((len(chars), self.glyph_h, self.glyph_w), dtype=np.uint8)
        for i, ch in enumerate(chars):
            im = Image.new("L", (self.glyph_w, self.glyph_h), color=self.background_color)
            d = ImageDraw.Draw(im)
            d.text((0,0), ch, fill=self.character_color, font=self.font)
            glyphs[i] = np.asarray(im, dtype=np.uint8)
        return glyphs  # (N, gh, gw)


    def render(self, char_idxs_2d: np.ndarray) -> np.ndarray:
        """
        Fast renderer: given (rows, cols) palette indices, return uint8 grayscale frame.
        """
        ## char_idxs_2d: (rows, cols) values in [0, N)
        ## (rows, cols, gh, gw)
        tiles = self.glyphs[char_idxs_2d]  
        ## STITCH TILES -> (rows*gh, cols*gw)
        frame = tiles.transpose(0, 2, 1, 3).reshape(
            char_idxs_2d.shape[0] * self.glyph_h,
            char_idxs_2d.shape[1] * self.glyph_w,
        )
        return frame

    def pixels_to_characters(self, img: torch.Tensor) -> np.ndarray:
        # img = (torch.Tensor(img) / 255) ** self.exposure
        img_embd = self.conv(img[None, None, ...])[0].permute(1, 2, 0)
        char_idxs = torch.cdist(img_embd, self.char_embd).argmin(dim=-1)  # (rows, cols)
        return char_idxs.cpu().numpy().astype(np.int32)

    def convert_image(self) -> None: 
        img = Image.open(self.input_path)
        img = self.reshape_image(img)
        img = torch.Tensor(np.array(img) / 255)
        img = expose_contrast(img, self.exposure, self.contrast, brightness=self.brightness)
        char_idxs = self.pixels_to_characters(img)  # (rows, cols)

        if self.output_mode == "text":
            # emit text file (optional: convert indices to chars)
            with open(self.output_path, "w") as f:
                for row in char_idxs:
                    f.write("".join(self.palatte[row]))
                    f.write("\n")
            return

        ascii_frame = self.render(char_idxs)
        Image.fromarray(ascii_frame).save(self.output_path)
        return
    

    def crop_frame(self, target_size:tuple, frame:np.ndarray) -> np.ndarray:
        # target_size is (w, h)
        h, w = frame.shape[:2]

        # PAD/CROP HEIGHT
        if h < target_size[1]:
            pad_dim = target_size[1] - h
            pad = np.ones((pad_dim, target_size[0]), dtype=np.uint8) * self.background_color
            frame = np.concatenate((frame, pad), axis=0)
        elif h > target_size[1]:
            frame = frame[: target_size[1]]

        # PAD/CROP WIDTH
        if w < target_size[0]:
            pad_dim = target_size[0] - w
            pad = np.ones((target_size[1], pad_dim), dtype=np.uint8) * self.background_color
            frame = np.concatenate((frame, pad), axis=1)
        elif w > target_size[0]:
            frame = frame[:, : target_size[0]]

        h2, w2 = frame.shape[:2]
        assert (w2, h2) == target_size
        return frame


    def convert_video(self) -> None:
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open input video: {self.input_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 0:
            fps = 30.0

        out = None
        size = None  # (w, h)

        for _ in trange(frame_count):
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            grayscale_frame = self.reshape_image(frame)
            char_idxs = self.pixels_to_characters(grayscale_frame)  # (rows, cols)
            ascii_frame = self.render(char_idxs)                    # (H, W) uint8

            h, w = ascii_frame.shape[:2]

            if out is None:
                size = (w, h)
                out = cv2.VideoWriter(self.output_path, fourcc, fps, size)
                if not out.isOpened():
                    cap.release()
                    raise RuntimeError(
                        "Could not open VideoWriter. Your OpenCV build may lack an MP4 backend/codec."
                    )

            if (w, h) != size:
                ascii_frame = self.crop_frame(size, ascii_frame)

            # most portable: write as BGR
            ascii_bgr = cv2.cvtColor(ascii_frame, cv2.COLOR_GRAY2BGR)
            out.write(np.ascontiguousarray(ascii_bgr))

        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        return


def required_in_size(out_size: int, k: int, p: int, s: int) -> int:
    # minimal input that produces exactly out_size (for dilation=1)
    return (out_size - 1) * s - 2 * p + k
