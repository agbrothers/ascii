### Converting Images to ASCII using 
# Convolutions & Associative Memory

<p align="center">
  <img src="content/steve-jobs.png" width="45%" />
  &nbsp;&nbsp;&nbsp;
  <img src="content/steve-jobs-ascii.png" width="45%" />
</p>

### Install:

```
pip install "git+https://github.com/agbrothers/ascii.git"
```

### Run: 
Simply type the keyword `asciify` followed by the image/video filepath to convert it to an ascii representation. The parameters at the end of this README can be passed to tweak the result. 

```
asciify -p=/path/to/asciify/content/steve-jobs.png -r=200 -e=.35 -b=0.95 -w=10 -con=4
```

```
asciify -p=/path/to/asciify/content/einstein.png -e=0.23 -r=200 -w=10 
```

```
asciify -p=/path/to/asciify/content/obama.png -r=200 -e=.3 -b=0.95 -w=10 -con=5
```




### Implementation:
>
> 1. Render each character as a 2D array called a glyph, ~15x8 pixels with the default font size. 
> 
> 2. Apply a hand-crafted 2D convolution to each glyph to pull out a feature vector corresponding to the character's shape and brightness. *NOTE*: We use a kernel with the same dimensions and stride of each character glyph.  
> 
> 3. Store the set of convolutional feature vectors as keys and the glyph arrays as values in a hetero-associative memory. 
> 
> 4. Given an image, partition it into patches the size of our character glyphs. Apply the convolution to each patch to generate a query feature vector. 
> 
> 5. Compute the similarity between the queries and keys in the ASCII memory (negative distance by default), then retrieve the most similar glyph. 
> 
> 6. Insert the retrieved glyphs where each query patch used to be, yielding the converted image. 
> 
> 

<br/>
There is no learning going on with this algorithm. The implementation pursued here is nonetheless deeply connected with a number of core concepts from machine learning, including convolutional neural networks, associative memories, and even attention. [TODO] Below I will walk through these big ideas, how they connect to a simple ASCII image converter, and why this is actually a desireable use case. 


### Parameters: 
```
asciify -p=./asciify/content/einstein.png -e=0.20 -r=150 -c=b
```

  `-p` `--path`, `[type=str]` 
  <br/> Path to the image or video to be converted

  `-ap` `--ascii_palette` `[type=str]` 
  <br/> Name of a pre-defined ASCII characters palette to build the image from, in `asciify/palettes.py`

  `-ac` `--ascii_chars` `[type=str]` 
  <br/> A string of characters to build the image from. Note, this overrides the --ascii_palette selection.

  `-f` `--font` `[type=str]` 
  <br/> Options: [`Menlo`]. Assuming a monospace font stored in asciify/fonts/ 

  `-c` `--color` `[type=str]` 
  <br/> Options: [`w`, `b`]. Color the characters `w` white or `b` black.

  `-m` `--output_mode` `[type=str]` 
  <br/> Options: [`text`, `default`]. Save the result to a .txt file or default to the input file type

  `-d` `--chars_per_line` `[type=int]` 
  <br/> Width of the output image in number of characters.

  `-b` `--brightness` `[type=float,` 
  <br/> default] Control image exposure prior to conversion, the closer to zero the brighter the image.

  `-e` `--exposure` `[type=float]` 
  <br/> Control image exposure prior to conversion, the closer to zero the brighter the image.

  `-con` `--contrast` `[type=float]` 
  <br/> Control image exposure prior to conversion, the closer to zero the brighter the image.

  `-w` `--weight` `[type=float]` 
  <br/> Control the weighting of individual pixels vs surrounding neighbors.

  `-fs` `--font_size` `[type=int]` 
  <br/> Font size.

  `-sim` `--similarity` `[type=str]`
  <br/> Options: [`dist`, `dot`, `cosine`]. Similarity metric to use for association.

