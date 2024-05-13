# AutoUpscale

## Universal app for scaling images

AutoUpscale is a universal app for scaling images. It can be used as a command line tool, a web GUI, or a console application.

### Supported algorithms
- **Classical Algorithms** <sup>(Up-scaling and downscaling)</sup>:
  - **Bicubic** *(Better than bilinear, less blur, more detail, higher contrast)*
  - **Bilinear** *(Second-simplest algorithm, most common and most blurry)*
  - **Nearest neighbor** *(Simplest algorithm, duplicates or keeps only the nearest pixel)*
  - **Area** *(Averages pixels into smaller ones)* *<sup>(Only down-scaling)</sup>*
  - **Lanczos** *(Better than bicubic, less blur, higher contrast)*
- **AI-based Algorithms** <sup>(Only up-scaling)</sup>:
  - **EDSR**
  - **ESPCN**
  - **FSRCNN** *(normal and small)*
  - **LapSRN**
  - **RealESRGAN** *(improved ESRGAN)*
- **Edge Detection Algorithms** <sup>(Only up-scaling)</sup>:
  - **hqx** *(edge detection algorithm, simple, not so great)*
  - **NEDI** *(New Edge-Directed Interpolation, can be better than hqx, but probably won't)*
  - **Super xBR** *(edge detection algorithm, based on xBR, more angles but more blur)*
  - **xBRZ** *(edge detection algorithm, based on xBR, better at preserving small details)*
- **Smart Algorithms** <sup>(Only up-scaling)</sup>:
  - **FSR** *(FidelityFX Super Resolution 1.1, made by AMD)*
  - **CAS** *(Contrast Adaptive Sharpening, made by AMD)*

<br/>

### Installation:
1. Make sure you have installed on your system:
   - **Python 3.12**
   - **Node.js** *(16.0.0 or newer)*
2. Clone this repository
3. Run the included `install.bat` script

### Usage:
- **Command line tool**:
  - Run the included `run_console.bat` script
  - Run the python script manually: `python src/standalone.py`
    - You can also pass arguments to the script. Add `--help` to see the list of available arguments
- **Web GUI** *(currently lacks some functionality)*:
  - Run the included `run_webui.bat` script

<br/>

### Examples:
**Example - Wiki Shell:**

Scaled down image *(40px)*:

![Wiki Example Shell - Small](./src/example_images/input/example_shell_40px.png)

Results of up-scaling the image *(40px -> 160px)*:

[//]: # (|                                                      Original                                                      |                                       Nearest Neighbour                                       |                                      Bilinear                                       |                                      Bicubic                                      |                                       Lanczos                                        |)

[//]: # (|:------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------:|)

[//]: # (| ![Original]&#40;https://upload.wikimedia.org/wikipedia/commons/a/a6/160_by_160_thumbnail_of_%27Green_Sea_Shell%27.png&#41; | ![Nearest Neighbour]&#40;./src/example_images/output/CV2_INTER_NEAREST_example_shell_40px_4x.png&#41; | ![Bilinear]&#40;./src/example_images/output/CV2_INTER_LINEAR_example_shell_40px_4x.png&#41; | ![Bicubic]&#40;./src/example_images/output/CV2_INTER_CUBIC_example_shell_40px_4x.png&#41; | ![Lanczos]&#40;./src/example_images/output/CV2_INTER_LANCZOS4_example_shell_40px_4x.png&#41; |)

[//]: # ()
[//]: # (|                                  EDSR                                   |                                   ESPCN                                   |                                   FSRCNN                                    |                                      FSRCNN-small                                       |                                   LapSRN                                    |                                   RealESRGAN                                    |)

[//]: # (|:-----------------------------------------------------------------------:|:-------------------------------------------------------------------------:|:---------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------:|:-------------------------------------------------------------------------------:|)

[//]: # (| ![EDSR]&#40;./src/example_images/output/CV2_EDSR_example_shell_40px_4x.png&#41; | ![ESPCN]&#40;./src/example_images/output/CV2_ESPCN_example_shell_40px_4x.png&#41; | ![FSRCNN]&#40;./src/example_images/output/CV2_FSRCNN_example_shell_40px_4x.png&#41; | ![FSRCNN-small]&#40;./src/example_images/output/CV2_FSRCNN_small_example_shell_40px_4x.png&#41; | ![LapSRN]&#40;./src/example_images/output/CV2_LapSRN_example_shell_40px_4x.png&#41; | ![RealESRGAN]&#40;./src/example_images/output/RealESRGAN_example_shell_40px_4x.png&#41; |)

[//]: # ()
[//]: # (|                                hqx                                |                      NEDI <sup>*&#40;m = 4&#41;*</sup>                      |                                   Super xBR                                   |                                xBRZ                                 |)

[//]: # (|:-----------------------------------------------------------------:|:-------------------------------------------------------------------:|:-----------------------------------------------------------------------------:|:-------------------------------------------------------------------:|)

[//]: # (| ![hqx]&#40;./src/example_images/output/hqx_example_shell_40px_4x.png&#41; | ![NEDI]&#40;./src/example_images/output/NEDI_example_shell_40px_4x.png&#41; | ![Super xBR]&#40;./src/example_images/output/Super_xBR_example_shell_40px_4x.png&#41; | ![xBRZ]&#40;./src/example_images/output/xBRZ_example_shell_40px_4x.png&#41; |)

[//]: # ()
[//]: # (|                              FSR                               |               CAS <sup>*&#40;sharpness = 0.5&#41;*</sup>               |)

[//]: # (|:--------------------------------------------------------------:|:--------------------------------------------------------------:|)

[//]: # (| ![FSR]&#40;./src/example_images/output/example_shell_40px_FSR.png&#41; | ![CAS]&#40;./src/example_images/output/example_shell_40px_CAS.png&#41; |)


| Original | Nearest Neighbour | Bilinear | Bicubic |
| :---: | :---: | :---: | :---: |
| ![Original](https://upload.wikimedia.org/wikipedia/commons/a/a6/160_by_160_thumbnail_of_%27Green_Sea_Shell%27.png) | ![Nearest Neighbour](./src/example_images/output/CV2_INTER_NEAREST_example_shell_40px_4x.png) | ![Bilinear](./src/example_images/output/CV2_INTER_LINEAR_example_shell_40px_4x.png) | ![Bicubic](./src/example_images/output/CV2_INTER_CUBIC_example_shell_40px_4x.png) |
| Lanczos | EDSR | ESPCN | FSRCNN |
| ![Lanczos](./src/example_images/output/CV2_INTER_LANCZOS4_example_shell_40px_4x.png) | ![EDSR](./src/example_images/output/CV2_EDSR_example_shell_40px_4x.png) | ![ESPCN](./src/example_images/output/CV2_ESPCN_example_shell_40px_4x.png) | ![FSRCNN](./src/example_images/output/CV2_FSRCNN_example_shell_40px_4x.png) |
| FSRCNN-small | LapSRN | RealESRGAN | hqx |
| ![FSRCNN-small](./src/example_images/output/CV2_FSRCNN_small_example_shell_40px_4x.png) | ![LapSRN](./src/example_images/output/CV2_LapSRN_example_shell_40px_4x.png) | ![RealESRGAN](./src/example_images/output/RealESRGAN_example_shell_40px_4x.png) | ![hqx](./src/example_images/output/hqx_example_shell_40px_4x.png) |
| NEDI <sup>*(m = 4)*</sup> | Super xBR | xBRZ | FSR |
| ![NEDI <sup>*(m = 4)*</sup>](./src/example_images/output/NEDI_example_shell_40px_4x.png) | ![Super xBR](./src/example_images/output/Super_xBR_example_shell_40px_4x.png) | ![xBRZ](./src/example_images/output/xBRZ_example_shell_40px_4x.png) | ![FSR](./src/example_images/output/example_shell_40px_FSR.png) |
| CAS <sup>*(sharpness = 0.5)*</sup> |
| ![CAS <sup>*(sharpness = 0.5)*</sup>](./src/example_images/output/example_shell_40px_CAS.png) |

<br/>

### Supported file formats:
**Tested working:**
- **Write:**
  - **PNG**
- **Read:**
  - **PNG** *(.png)*
  - **JPEG** *(.jpg, .jpeg)*

**Should work:**
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

### Credits:
- **WebUI** and **Scaling App** created by [***Mikołaj Pokora***](https://github.com/MikiP98)
- **API backend** and **xBRZ wheel** by [***Piotr Przetacki***](https://github.com/PiotrPrzetacki)
- [**RealESRGAN**](https://github.com/ai-forever/Real-ESRGAN) implementation by [ai-forever](https://github.com/ai-forever)
- [**NEDI**](https://github.com/Kirstihly/Edge-Directed_Interpolation) implementation by [Ley (Kirstihly)](https://github.com/Kirstihly)
- [**hqx**](https://pypi.org/project/hqx/) implementation by [whoatemybutter](https://pypi.org/user/whoatemybutter/)
- [**xBRZ**](https://github.com/ioistired/xbrz.py) implementation by [ioistired](https://github.com/ioistired)
- [**Super xBR**](https://github.com/MikiP98/py-super-xbr) implementation originally created by [Matt Schwartz (n0spaces)](https://github.com/n0spaces) corrected by [Mikołaj Pokora](https://github.com/MikiP98)
- [**FSR**](https://gpuopen.com/fidelityfx-superresolution/) and [**CAS**](https://gpuopen.com/fidelityfx-cas/) are implemented using [FidelityFX-CLI](https://github.com/GPUOpen-Effects/FidelityFX-CLI) by [GPUOpen-Effects](https://github.com/GPUOpen-Effects) and [AMD](https://www.amd.com/) <sup>*(licence in "src/FidelityFX-CLI-v1.0.3")*</sup>
- ***Area***, ***Bicubic***, ***Bilinear***, ***Lanchos*** and ***Nearest neighbor*** algorithms are implemented using [OpenCV](https://opencv.org)
- ***Nearest neighbor***, ***Bilinear***, ***Bicubic*** and ***Lanchos*** algorithms are also implemented using [Pillow library](https://pillow.readthedocs.io/en/stable/)

<sup>

- **Download** icon: <a href="https://www.flaticon.com/free-icons/install" title="install icons">Install icons created by NajmunNahar - Flaticon</a>

- **Web GUI** icon: <a href="https://www.flaticon.com/free-icons/interface" title="interface icons">Interface icons created by Freepik - Flaticon</a>

- **Console** icon: <a target="_blank" href="https://icons8.com/icon/nRH1nzeThlgk/console">Console</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a>

</sup>
