import torch
import numpy as np

from PIL import Image, ImageDraw, ImageFont

from asciify.kernels import KERNELS
from asciify.palettes import PALETTES


class AsciiMemory:
    """
    This class stores and recalls character glyphs
    via the chosen similarity function. 

    A 2D convolution is used to extract queries 
    from an image (2D Tensor). 
    """

    def __init__(
        self,
        ascii_palette:str,
        ascii_chars:str,
        weight:float,
        font:ImageFont.FreeTypeFont,
        color:str="b",
        kernel:str="default",
        sim:str="dist",
    ):
        ## PARSE ARGS
        is_white = {"b":0, "w":1}[color]
        self.character_color = is_white * 255
        self.background_color = (not is_white) * 255
        self.weight = weight
        self.font = font
        self.sim = sim
        self.char_idxs = None
        if ascii_chars is not None:
            self.palette = np.array([*ascii_chars])
        else:
            self.palette = np.array(PALETTES[ascii_palette.lower()])

        ## GET MAX CHARACTER WIDTH AND HEIGHT
        W_mx = H_mx = 0
        for char in self.palette:
            x0, y0, x1, y1 = font.getbbox(char)
            W_mx = max(W_mx, x1 - x0)
            H_mx = max(H_mx, y1 - y0)
        self.glyph_w, self.glyph_h = W_mx, H_mx

        ## CONVOLUTION MAPPING PIXELS TO QUERIES
        kernel = KERNELS[kernel](
            weight=weight,
            width=self.glyph_w, 
            height=self.glyph_h
        )
        self.conv = torch.nn.Conv2d(
            in_channels=kernel.shape[1], 
            out_channels=kernel.shape[0], 
            kernel_size=kernel.shape[-2:], 
            stride=kernel.shape[-2:], 
            padding=0,
        )
        with torch.no_grad():
            self.conv.weight[:] = kernel
            self.conv.bias[:] = 0

        ## THE RENDERED CHARACTERS ARE OUR MEMORY VALUES
        self.char_values = self.get_char_values(self.palette)

        ## DERIVE KEYS FROM EACH CHARACTER'S PIXELS
        self.char_keys = self.get_char_keys(self.char_values) 
        return


    def get_char_values(self, palette) -> np.ndarray:
        ## RENDER EACH CHARACTER GLYPH (ONCE) INTO A SMALL TILE: (N, H, W)
        glyphs = np.zeros((len(palette), self.glyph_h, self.glyph_w), dtype=np.uint8)
        for i,char in enumerate(palette):
            im = Image.new("L", size=(self.glyph_w, self.glyph_h), color=self.background_color)
            d = ImageDraw.Draw(im)
            d.text((0,0), char, fill=self.character_color, font=self.font)
            glyphs[i] = np.asarray(im, dtype=np.uint8)
        return torch.Tensor(glyphs).to(torch.uint8)


    def get_char_keys(self, glyphs:torch.Tensor):
        ## CONVOLVE EACH GLYPH TO OBTAIN THEIR CORRESPONDING KEYS
        glyphs = torch.clamp(glyphs.to(torch.float) / 255, 0, 1)        
        return self.conv(glyphs[:, None])[..., 0,0]


    def recall_char(self, queries:torch.Tensor) -> torch.IntTensor:
        ## HARD ATTENTION/ASSOCIATION: 
        ## Look up the character with the greatest similarity 
        if self.sim == "dist":      # (negative euclidean distance)
            char_sim = -torch.cdist(queries, self.char_keys) 
        elif self.sim == "dot":     # (dot product)
            char_sim = queries @ (self.char_keys).T        
        elif self.sim == "cosine":  # (cosine similarity)
            key_norm = self.char_keys.norm(dim=-1, keepdim=True)
            query_norm = queries.norm(dim=-1, keepdim=True)
            char_sim = (queries/query_norm) @ (self.char_keys/key_norm).T   
        ## RETURN IDX OF THE MOST SIMILAR CHARACTER FOR EACH PATCH
        return char_sim.argmax(dim=-1) 


    def render(self, char_idxs:torch.IntTensor) -> np.ndarray:
        ## GIVEN A CHAR IDX FOR EACH PATCH, REPLACE EACH PATCH 
        ## WITH THE CORRESPONDING CHARACTER GLYPH
        ## char_idxs: (rows, cols, idx) values in [0, N)
        ## (rows, cols, gh, gw)
        tiles = self.char_values[char_idxs]
        ## STITCH TILES -> (rows*gh, cols*gw)
        frame = tiles.transpose(2, 1).reshape(
            char_idxs.shape[0] * self.glyph_h,
            char_idxs.shape[1] * self.glyph_w,
        )
        return frame.numpy()
    
    def forward(self, frame:torch.Tensor) -> np.ndarray:
        ## FOR EACH PIXEL PATCH, GENERATE A QUERY VECTOR VIA CONVOLUTION
        queries = self.conv(frame[None, None, ...])[0].permute(1, 2, 0)        
        self.char_idxs = self.recall_char(queries)  
        return self.render(self.char_idxs) 

    def __call__(self, x):
        return self.forward(x)
    