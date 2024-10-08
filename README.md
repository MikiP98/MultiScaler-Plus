# MultiScaler Plus

## Universal app for scaling images

**MultiScaler Plus** is a universal app for scaling images using various algorithms.  
It can be used as a command line tool, a webUI, or as a console application.

This app has **2** active versions [**Main (release)**]() and [**Dev (beta)**]()  
If you use the **Main** branch and see on roadmap some feature you would like to use,  
check the **Dev** branch to see if it's already implemented!  
To switch between branches, use the `git checkout {main/dev}` command  
If feature you are interested in is not in either branch's roadmap, feel free to create a **Feature Request** issue!

## Supported algorithms
- **Classical Algorithms** <sup>(Up-scaling and downscaling)</sup>:
  - **Area** *(Averages pixels into smaller ones)* *<sup>(Only down-scaling)</sup>*
  - **Bicubic** *(Better than bilinear, less blur, more detail, higher contrast)*
  - **Bilinear** *(Second-simplest algorithm, most common and most blurry)*
  - **Lanczos** *(Better than bicubic, less blur, higher contrast)*
  - **Nearest neighbor** *(Simplest algorithm, duplicates or keeps only the nearest pixel)*
- **AI-based Algorithms** <sup>(Only up-scaling)</sup>:
  - **A2N**
  - **AWSRN-BAM**
  - **CARN**
  - **CARN-BAM**
  - **DRLN** *(recommended)*
  - **DRLN-BAM** *(recommended)*
  - **EDSR**
  - **EDSR-base**
  - **ESPCN**
  - **FSRCNN** *(normal and small)*
  - **HAN**
  - **LapSRN**
  - **MDSR**
  - **MDSR-BAM**
  - **MSRN**
  - **MSRN-BAM**
  - **PAN**
  - **PAN-BAM**
  - **RCAN-BAM**
  - **RealESRGAN** *(improved ESRGAN)* *(recommended)*
  - **Anime4K** *(recommended)*
  - **HSDBTRE** *(hybrid of DRLN and RealESRGAN AIs)* *(recommended)*
- **Edge Detection Algorithms** <sup>(Only up-scaling)</sup>:
  - **hqx** *(edge detection algorithm, simple, not so great)*
  - **NEDI** *(New Edge-Directed Interpolation, can be better than hqx, but probably won't)*
  - **Super xBR** *(edge detection algorithm, based on xBR, more angles but more blur)*
  - **xBRZ** *(edge detection algorithm, based on xBR, better at preserving small details)*
- **Smart Algorithms** <sup>(Only up-scaling)</sup>:
  - **FSR** *(FidelityFX Super Resolution 1.1, made by AMD)*
  - **CAS** *(Contrast Adaptive Sharpening, made by AMD)*

### [More detailed algorithms descriptions](content/resources/README_extensions/algorithm_descriptions/README.md)

<br/>

## Algorithm Recommendations:

### Downscaling:

Downscaling is simpler that's why it's first  
In theory the best algorithm to use, supported by this APP is **PIL**'s implementation of `Lanchos` algorithm  
Second best in theory is **PIL**'s implementation of `Bicubic` algorithm  
In practice though the differance is that **Lanchos** will have **sharper** and **contrastier** look, 
  but sometimes it looks like it has some over-sharping artifacts  
If you are looking for even softer look try `Area Average` implementation by **CV2**.  
The rest of algorithms might be used as an artistic choice, sometimes with cool and interesting results

### Upscaling:

If you want to quickly scale some images with anything at 
  least a bit better than a default bilinear scaler present in most application, 
  chose either: `Bicubic` or `Lanchos`  
Though **Lanchos** is in theory better it sometimes looks over sharpened and over contrasted 
  in comparison to **Bicubic**  
`FSR` can also result in better image as it better preserves the overall shape of the object in the image, 
  but it will add some noise and grain to the output

If you are willing to use AI, EDSR implementation by CV2 offers the least blur while not adding any visible artifacts.
The image will still be noticeably blurry though
The result might also get worse the bigger the scaling factor.<sup>(main testing is done with the factor of 4)</sup>

If you wish to get the best possible results from the upscaling you can choose 1 of 3 paths:

- **AI for realistic images:**  
  There are a lot of AI algorithms to chose from but here are the best overall for realistic images:
  - `DRLN` implemented by **SI**, *or `DRLN-BAM` if your scaling factor is less than 4*  
    **DRLN** is in theore the bst of the simple scaling AI's that do not add detail to the image.
      The image will most likely look better than when scaled with classic or smart algorithms,
      but at larger scales the lack of detail becomes visible :/
  - `RealESRGAN`  
    **RealESRGAN** adds more detail to the upscaled image and when it works, it works great!
      But not so rarely it tends to over smooth the image creating flat surfaces where previously was detail
      and has big tendencies to hallucinate if the input image was too small or there was not enough detail in it
      **RealESRGAN** can also be used to remove the JPEG artifacts from the image :)
  - `HSDBTRE`  
    **HSDBTRE** is a simple hybrid of the 2 algorithms above.
      It starts with applying 2x **DRLN** after which comes 2x **RealESRGAN**.

- **AI for Anime or similar contrast art-style:**
  - `Anime4K`  
    Designed to upscale old Anime in realtime during playback. Easy to run with mostly good results.
  - `RealESRGAN`  
    Offers a bit better contrast on the edges, but a bit worse antialiasing, while being a lot slower.
     Also tends to over smooth the background, removing small detail as e.g. fences or pattern on shirts.  
  - `DRLN`
    As it focuses on the best upscaling without adding detail, it won't destroy the visuals with many artifacts, 
      while still being better than **Bilinear** scaling in most playback software.
      Will be the softest of all 3.

- **Edge Detection for pixel art or Anime/similar contrast art-style:**  
  Most edge detection algorithms are really unique, it is really hard to choose the best ones but here we go!
  - `xBRZ`  
    Personally one of my favourite algorithms, this is the one that inspired me to make this APP :)  
    Creates a palette effect when there are gradients of high frequency detail,
      usually not visible on anime or similar styles. 
      The simplest way to describe it is that it adds 45 deg lines where there are edges, instead of blurring them.
  - `Super xBR`  
    Works for more angles than xBRZ, but produces more blurry output
  - `NEDI`  
    The overall bluriness and shape are similar to **Super xBR**, but it adds detail in artistic way. 
      May produce some visible artifacts. 
      The edge detection radius can be fine-tuned with `Nedi_m` config option. *(Default and recommended minimum is 4)*


<br/>

## Installation:
1. Make sure you have installed on your system:
   - **[Python](https://www.python.org/downloads/) 3.12** <sup>(minor version does not matter)</sup>
   - ~~**[*OPTIONAL*] [Node.js](https://nodejs.org/en/download/prebuilt-installer)** *(16.0.0 or newer)*~~
   - ~~**[*OPTIONAL*] [Docker](https://docs.docker.com/get-docker/)** *(for **Waifu2x** & **Supir**)*~~
2. Clone this repository `git clone "https://github.com/MikiP98/MultiScaler-Plus"`
3. Run the included `install.bat` script

## Usage:
- **Command line tool**:
  - Run the included `run_console.ps1` script
    - Right-click on the script and select `Run with PowerShell`
  - Or run the python script manually: `python main.py`
    - Make sure you are inside the folder `content/src` *(for now)*
    - ~~You can also pass arguments to the script. Add `--help` to see the list of available arguments~~ *(will be back soon!)*:
- ~~**Web GUI**~~ *(will be back soon!)*:
  - ~~Run the included `run_webui.bat` script~~

<br/>

## Examples:
### Example - Wiki Shell:

Scaled down image *(40px)*: <br>
![Wiki Example Shell - Small](content/resources/example_images/input/example_shell_40px.png)

[//]: # (TODO: Make the Original use the local file)

A summary of best and most unique results of up-scaling the image *(40px -> 160px)*:

|                                                      Original                                                      |                                                       Nearest Neighbour <sup>*(CV2)*</sup>                                                       |                                                    Hamming                                                    |                                                    Bicubic *(PIL)*                                                    |
|:------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------:|
| ![Original](https://upload.wikimedia.org/wikipedia/commons/a/a6/160_by_160_thumbnail_of_%27Green_Sea_Shell%27.png) | ![Nearest Neighbour <sup>*(CV2)*</sup>](content/resources/example_images/output/example_shell_40px/CV2_INTER_NEAREST_example_shell_40px_4x.webp) | ![Hamming](content/resources/example_images/output/example_shell_40px/PIL_HAMMING_example_shell_40px_4x.webp) | ![Bicubic *(PIL)*](content/resources/example_images/output/example_shell_40px/PIL_BICUBIC_example_shell_40px_4x.webp) |

|                                                    Lanczos *(PIL)*                                                    |                                                  EDSR *(CV2)*                                                   |                                                  DRLN<sup>*(-BAM if <4x)*</sup> *(SI)*                                                  |                                                   RealESRGAN                                                    |
|:---------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:|
| ![Lanczos *(PIL)*](content/resources/example_images/output/example_shell_40px/PIL_LANCZOS_example_shell_40px_4x.webp) | ![EDSR *(CV2)*](content/resources/example_images/output/example_shell_40px/CV2_EDSR_example_shell_40px_4x.webp) | ![DRLN<sup>*(-BAM if <4x)*</sup> *(SI)*](content/resources/example_images/output/example_shell_40px/SI_drln_example_shell_40px_4x.webp) | ![RealESRGAN](content/resources/example_images/output/example_shell_40px/RealESRGAN_example_shell_40px_4x.webp) |

|                                                  Anime4K                                                  |                                                  HSDBTRE                                                  |                                                NEDI <sup>*(m = 4)*</sup>                                                 |                                                   Super xBR                                                   |
|:---------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------:|
| ![Anime4K](content/resources/example_images/output/example_shell_40px/Anime4K_example_shell_40px_4x.webp) | ![HSDBTRE](content/resources/example_images/output/example_shell_40px/HSDBTRE_example_shell_40px_4x.webp) | ![NEDI <sup>*(m = 4)*</sup>](content/resources/example_images/output/example_shell_40px/NEDI_example_shell_40px_4x.webp) | ![Super xBR](content/resources/example_images/output/example_shell_40px/Super_xBR_example_shell_40px_4x.webp) |

|                                                xBRZ                                                 |                                                FSR 1.1                                                |                                                   Repetition                                                    |
|:---------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:|
| ![xBRZ](content/resources/example_images/output/example_shell_40px/xBRZ_example_shell_40px_4x.webp) | ![FSR 1.1](content/resources/example_images/output/example_shell_40px/FSR_example_shell_40px_4x.webp) | ![Repetition](content/resources/example_images/output/example_shell_40px/Repetition_example_shell_40px_4x.webp) |

### [More Detailed Comparisons](content/resources/README_extensions/quality_comparison/README.md)

*For recommendations look just below the algorithm list*

<br/>

## Supported file formats:

### Tested working:

- **Write:**
  - **PNG** *(Widely used, popular, lossless format)*
  - **QOI** *(A bit worse compression then **PNG**, but a lot, lot faster to save and load)*
  - **WEBP** *(Comparable, lossless and lossy compression, to **JPEG XL** (a bit worse on average), but with better overall support)*
  - **JPEG XL** *(New advanced compression format, better lossless compression compared to **PNG** and better lossy compared to **JPEG**)* <br> <sup>*(see [this plugin](https://github.com/saschanaz/jxl-winthumb) for Windows Support)*</sup>
  - **AVIF** *(New advanced compression format, much, much slower and with worse lossless compression then **WEBP** and **JPEG XL**, currently no transparency because of a bug, pretty wide support)*  
    *<sup> See benchmarks below for more detail </sup>*

- **Read:**
  - **JPEG** *(.jpg, .jpeg)*
  - **PNG** *(.png)*
  - **WEBP** *(.webp)*

### Should work:

- **Read:**
  <table>
    <tr>
      <th>
- 
    - **APNG** *(.apng, .png2)*
    - **BLP** *(.blp, .blp2, .tex)*
    - **BMP** *(.bmp, .rle)*
    - **CUR** *(.cur)*
    - **DCX** *(.dcx)*
    - **DDS** *(.dds, .dds2)*
    - **DIB** *(.dib, .dib2)*
    - **EMF** *(.emf)*
    - **EPS** *(.eps, .eps2, .epsf, .epsi)*
    - **FITS** *(.fits)*
    - **FLC** *(.flc)*
    - **FLI** *(.fli)*
    - **FPX** *(.fpx)*
    - **FTEX** *(.ftex)*
    - **GBR** *(.gbr)*
    - **GD** *(.gd)*
    - **GIF** *(.gif, .giff)*
    - **ICNS** *(.icns, .icon)*
    - **ICO** *(.ico, .cur)*
    - **IM** *(.im, .im2)*
    - **IMT** *(.imt)*
    - **IPTC** *(.iptc)*
    - **JPEG** *(.jpg, .jpeg, .jpe)*
    - **JPEG 2000** *(.jp2, .j2k, .jpf, .jpx, .jpm, .j2c, .j2r, .jpx)*
      </th>
      <th>
-
    - **MCIDAS** *(.mcidas)*
    - **MIC** *(.mic)*
    - **MPO** *(.mpo)*
    - **MSP** *(.msp, .msp2)*
    - **NAA** *(.naa)*
    - **PCD** *(.pcd)*
    - **PCX** *(.pcx, .pcx2)*
    - **PFM** *(.pfm, .pfm2)*
    - **PIXAR** *(.pixar)*
    - **PNG** *(.png, .pns)*
    - **PPM** *(.ppm, .ppm2)*
    - **PSD** *(.psd)*
    - **QOI** *(.qoi)*
    - **SGI** *(.sgi, .rgb, .bw)*
    - **SPIDER** *(.spi, .spider2)*
    - **SUN** *(.sun)*
    - **TGA** *(.tga, .targa)*
    - **TIFF** *(.tif, .tiff, .tiff2)*
    - **WAL** *(.wal)*
    - **WMF** *(.wmf)*
    - **WebP** *(.webp, .webp2)*
    - **XBM** *(.xbm, .xbm2)*
    - **XPM** *(.xpm)*
      </th>
    </tr>
  </table>

<br/>

## Performance:

[//]: # (TODO: Rerun the benchmarks with the new version of the app + add a version with no additional lossless compression)

File size and time needed to save the image using different formats with lossless+ compression.  
Tested on the [xBRZ Retexture v1.2 64x](https://modrinth.com/resourcepack/xbrz-retexture/version/1.2) Minecraft resourcepack + example shell:

| File format | Size *(B)*   | Time *(~s)* |
|:------------|:-------------|:------------|
| **PNG**     | *19 963 489* | *37.685-*   |
| **QOI**     | *30 006 495* | *2.017-*    |
| **WEBP**    | *11 396 360* | *19.904-*   |
| **JPEG XL** | *11 947 953* | *56.468-*   |
| **AVIF***   | *17 282 612* | *691.370+*  |

Different test on random collection of smaller files:

| File format | Size *(B)* |
|:------------|:-----------|
| **PNG**     | *675 397*  |
| **QOI**     | *790 448*  |
| **WEBP**    | *444 538*  |
| **JPEG XL** | *450 085*  |
| **AVIF***   | *507 384*  |

<sup>*AVIF does not have transparency for some unknown reason</sup>

### How to speed up AI algorithms:

By default `pytorch`, library used by most AI algorithms, installs without GPU acceleration.  
It is that way because it required <300 MB to download and install, while the GPU version requires almost 2 GB to download.  
To get the GPU accelerated version, first please run the following command to uninstall current pytorch version:
```bash
pip uninstall torch torchvision torchaudio
```
After that you need to reinstall pytorch with GPU support, to find the correct command for your system, please visit the [pytorch website](https://pytorch.org/get-started/locally/) and select the correct options.  
*Example command for Windows with CUDA 12.4 with stable torch release 2.4.0:*
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Roadmap:

- Rewrite and update the **WebUI** *(in progress...)*
- Add support for **Waifu2x** and **Supir** AIs via **Docker**
  - Add lambda GPU *(or other)* connection support for **Supir** and others
- **Fix** and **improve** standalone console application experience:
  - Smarter Algorithms print with descriptions and categories
  - Smarter config editing with descriptions ~~and incorrect input handling~~
  - Saving user config settings *(multiple **presets**)*
  - *(add console **buttons**?)*
- Add support for **stacked** and **animated** images
- Add **image tracing** scaling algorithm and support for **SVG** format
- Add proper **HDR** support <sup> *(I think **JPEG XL**, **WEBP** and **AVIF** may have some already)* </sup>
- Add better image quality comparison:
  - ~~Summary~~
  - Extended summary
  - ~~note with recommendations~~
  - Downscaling comparison
- Create a **C++ python extension** for:
  - More optimizations and better performance
  - **ScaleFX** scaling shader
  - NVIDIAs **DLSS** and **NIS** support
  - support for **WEBP2** format *(both reading and writing)*
- Add support for **ZIP** and **7z** archives as input and output
- Add **filters** and **effects** support: *(in progress...)*
  - Blur
  - Brightness
  - CAS *(Contrast Adaptive Sharpening)*
  - Color correction
  - Color grading
  - Contrast
  - [DeOldify](https://github.com/jantic/DeOldify)
  - Noise reduction
  - Saturation
  - Sharpen
  - Exposure
  - Motion blur *(for animated and stacked images)* *(temporal data and optical flow)*
  - [Diffraction hologram](https://www.youtube.com/watch?v=atg1PN1sZfg)
  - Negative
  - ~~Normal map strength~~
- Add basic **cropping** and **rotating** support
- Add **intelligent masking** *(to e.g. not mask the minecraft bat wing on the edge, but in a box)*
- Make my own scaling algorithm or AI for fun :) <sup>*(HSDBTRE deos not count)*</sup>
- Add an option to blend all algorithms together instead of saving them separately
- Add some conversions:
  - Old SEUS to labPBR 1.3
  - ~~Old Continuum to labPBR 1.3~~
  - PPR+Emissive (old BSL) to labPBR 1.3
  - Gray to labPBR 1.3 (most likely won't be great)
  - More?
- Add DP DSC image format?
- ~~Covert classes into typed dictionaries to increase performance~~
- Add image merger: multiple images into one stacked or animated image
- ~~Add big 160px example shell image to example images~~
- Librarify this app...
- Add a markdown page(s) with detailed algorithms descriptions *(in progress...)*

<br/>

## Credits:
- **WebUI**, **Scaling App** and **HSDBTRE** AI hybrid created by [***Mikołaj Pokora***](https://github.com/MikiP98)
- **API backend** and **xBRZ wheel** by [***Piotr Przetacki***](https://github.com/PiotrPrzetacki)
- [**Anime4K**](https://github.com/TianZerL/pyanime4k) implementation by [TianZer (TianZerL)](https://github.com/TianZerL)
- [**RealESRGAN**](https://github.com/ai-forever/Real-ESRGAN) implementation by [ai-forever](https://github.com/ai-forever)
- [**NEDI**](https://github.com/Kirstihly/Edge-Directed_Interpolation) implementation by [Ley (Kirstihly)](https://github.com/Kirstihly)
- [**hqx**](https://pypi.org/project/hqx/) implementation by [whoatemybutter](https://pypi.org/user/whoatemybutter/)
- [**xBRZ**](https://github.com/ioistired/xbrz.py) implementation by [ioistired](https://github.com/ioistired)
- [**Super xBR**](https://github.com/MikiP98/py-super-xbr) implementation originally created by [Matt Schwartz (n0spaces)](https://github.com/n0spaces) corrected by [Mikołaj Pokora](https://github.com/MikiP98)
- [**FSR**](https://gpuopen.com/fidelityfx-superresolution/) and [**CAS**](https://gpuopen.com/fidelityfx-cas/) are implemented using [FidelityFX-CLI](https://github.com/GPUOpen-Effects/FidelityFX-CLI) by [GPUOpen-Effects](https://github.com/GPUOpen-Effects) and [AMD](https://www.amd.com/) <sup>*([licence](content/src/FidelityFX_CLI/FidelityFX-CLI-v1.0.3/license.txt) in "content/src/FidelityFX-CLI/FidelityFX-CLI-v1.0.3")*</sup>
- ***Bicubic***, ***Bilinear***, ***Box***, ***Hamming***, ***Lanchos*** and ***Nearest neighbor*** algorithms are implemented using [Pillow library](https://pillow.readthedocs.io/en/stable/)
- ***Area*** as well as ***Bicubic***, ***Bilinear***, ***Lanchos*** and ***Nearest neighbor*** algorithms are implemented using [OpenCV](https://opencv.org)
- ***EDSR***, ***ESPCN***, ***FSRCNN***, ***FSRCNN-small***, ***LapSRN*** AI algorithms are also implemented using [OpenCV](https://opencv.org)
- ***A2N***, ***AWSRN-BAM***, ***CARN***, ***CARN-BAM***, ***DRLN***, ***DRLN-BAM***, ***EDSR***, ***EDSR-base***, ***HAN***, ***MDSR***, *...gasssp...*  
  ***MDSR-BAM***, ***MSRN***, ***MSRN-BAM***, ***PAN***, ***PAN-BAM***, ***RCAN-BAM*** AI algorithms are implemented using:  
  [super-image](https://pypi.org/project/super-image/) by [eugenesiow (Eugene Siow)](https://pypi.org/user/eugenesiow/) and [Freed Wu](https://pypi.org/user/Freed-Wu/)
- [**QOI file format support library**](https://github.com/kodonnell/qoi) by [kodonnell](https://github.com/kodonnell)
- [**AVIF PIL plugin**](https://pypi.org/project/pillow-avif-plugin/) by [fdintino](https://pypi.org/user/fdintino/)
- [**JPEG XL PIL plugin**](https://pypi.org/project/pillow-jxl-plugin/) by [Isotr0py](https://pypi.org/user/Isotr0py/)
- **Example Shell** images: 
  - [**Green Sea Shell 160 thumbnail**](https://commons.wikimedia.org/wiki/File:160_by_160_thumbnail_of_%27Green_Sea_Shell%27.png) by [James Petts](https://www.flickr.com/people/14730981@N08) / shaddim *and*  
  [**Green Sea Shell 40 thumbnail**](https://commons.wikimedia.org/wiki/File:40_by_40_thumbnail_of_%27Green_Sea_Shell%27.png) by [James Petts](https://www.flickr.com/people/14730981@N08)  
  Under: [CC BY-SA 2.5](https://creativecommons.org/licenses/by-sa/2.5), via Wikimedia Commons
  - Original [**Green sea shell**](https://commons.wikimedia.org/wiki/File:Green_sea_shell_(11985932994).jpg) by [James Petts](https://www.flickr.com/people/14730981@N08)  
  Under: [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0), via Wikimedia Commons

<br>

*<sup>
This file contains shell images that are derived from works licensed under Creative Commons Attribution-ShareAlike 2.5 and 2.0.<br>
These images, including any modifications, are licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
</sup>*

[//]: # (<sup>)

[//]: # ()
[//]: # (- **Download** icon: <a href="https://www.flaticon.com/free-icons/install" title="install icons">Install icons created by NajmunNahar - Flaticon</a>)

[//]: # ()
[//]: # (- **Web GUI** icon: <a href="https://www.flaticon.com/free-icons/interface" title="interface icons">Interface icons created by Freepik - Flaticon</a>)

[//]: # ()
[//]: # (- **Console** icon: <a target="_blank" href="https://icons8.com/icon/nRH1nzeThlgk/console">Console</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a>)

[//]: # ()
[//]: # (</sup>)
