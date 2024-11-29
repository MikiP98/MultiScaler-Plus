# Nearest Neighbour

[Back to algorithm list](../README.md)

## Description

The Nearest Neighbour interpolation is the simplest algorithm that duplicates or keeps only the nearest pixel to the middle of the new pixel.  
It supports both upscaling and downscaling.  
<br> 
*Small downscale and upscale "how it works" images*  
<br>
Upscaling with this algorithm is pointless as the only thing you will achieve is a bigger file size.  
It is recommended that if you want to make a pixel art image bigger, you should use some internal scaling of the program you use, e.g.: website CSS: 
```css
image {
    image-rendering: pixelated;
    width: 200%;
    height: auto;
}
```
As for downscaling, you can use it to create some interesting artistic effects, but it is not recommended for general use.

[*Wikipedia Article*](https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation)

## Images:

Test pattern images (d) + shell (ud) + pixel art (ud)  
In:
- Original
- Nearest Neighbour
- Area Average
- Bilinear
- PIL Lanczos
- ...

### [Back to algorithm list](../README.md)