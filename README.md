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
  - **Anime4K**
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

| Original | Nearest Neighbour | Bilinear | Bicubic |
| :---: | :---: | :---: | :---: |
| ![Original](https://upload.wikimedia.org/wikipedia/commons/a/a6/160_by_160_thumbnail_of_%27Green_Sea_Shell%27.png) | ![Nearest Neighbour](./src/example_images/output/CV2_INTER_NEAREST_example_shell_40px_4x.png) | ![Bilinear](./src/example_images/output/CV2_INTER_LINEAR_example_shell_40px_4x.png) | ![Bicubic](./src/example_images/output/CV2_INTER_CUBIC_example_shell_40px_4x.png) |

| Lanczos | EDSR *(CV2)* | ESPCN *(CV2)* | FSRCNN *(CV2)* |
| :---: | :---: | :---: | :---: |
| ![Lanczos](./src/example_images/output/CV2_INTER_LANCZOS4_example_shell_40px_4x.png) | ![EDSR *(CV2)*](./src/example_images/output/CV2_EDSR_example_shell_40px_4x.png) | ![ESPCN *(CV2)*](./src/example_images/output/CV2_ESPCN_example_shell_40px_4x.png) | ![FSRCNN *(CV2)*](./src/example_images/output/CV2_FSRCNN_example_shell_40px_4x.png) |

| FSRCNN-small *(CV2)* | LapSRN *(CV2)* | A2N *(SI)* | AWSRN-BAM *(SI)* |
| :---: | :---: | :---: | :---: |
| ![FSRCNN-small *(CV2)*](./src/example_images/output/CV2_FSRCNN_small_example_shell_40px_4x.png) | ![LapSRN *(CV2)*](./src/example_images/output/CV2_LapSRN_example_shell_40px_4x.png) | ![A2N *(SI)*](./src/example_images/output/SI_a2n_example_shell_40px_4x.png) | ![AWSRN-BAM *(SI)*](./src/example_images/output/SI_awsrn_bam_example_shell_40px_4x.png) |

| CARN *(SI)* | CARN-BAM *(SI)* | DRLN *(SI)* | DRLN-BAM *(SI)* |
| :---: | :---: | :---: | :---: |
| ![CARN *(SI)*](./src/example_images/output/SI_carn_example_shell_40px_4x.png) | ![CARN-BAM *(SI)*](./src/example_images/output/SI_carn_bam_example_shell_40px_4x.png) | ![DRLN *(SI)*](./src/example_images/output/SI_drln_example_shell_40px_4x.png) | ![DRLN-BAM *(SI)*](./src/example_images/output/SI_drln_bam_example_shell_40px_4x.png) |

| EDSR *(SI)* | EDSR-base *(SI)* | HAN *(SI)* | MDSR *(SI)* |
| :---: | :---: | :---: | :---: |
| ![EDSR *(SI)*](./src/example_images/output/SI_edsr_example_shell_40px_4x.png) | ![EDSR-base *(SI)*](./src/example_images/output/SI_edsr_base_example_shell_40px_4x.png) | ![HAN *(SI)*](./src/example_images/output/SI_han_example_shell_40px_4x.png) | ![MDSR *(SI)*](./src/example_images/output/SI_mdsr_example_shell_40px_4x.png) |

| MDSR-BAM *(SI)* | MSRN *(SI)* | MSRN-BAM *(SI)* | PAN *(SI)* |
| :---: | :---: | :---: | :---: |
| ![MDSR-BAM *(SI)*](./src/example_images/output/SI_mdsr_bam_example_shell_40px_4x.png) | ![MSRN *(SI)*](./src/example_images/output/SI_msrn_example_shell_40px_4x.png) | ![MSRN-BAM *(SI)*](./src/example_images/output/SI_msrn_bam_example_shell_40px_4x.png) | ![PAN *(SI)*](./src/example_images/output/SI_pan_example_shell_40px_4x.png) |

| PAN-BAM *(SI)* | RCAN-BAM *(SI)* | RealESRGAN | Anime4K |
| :---: | :---: | :---: | :---: |
| ![PAN-BAM *(SI)*](./src/example_images/output/SI_pan_bam_example_shell_40px_4x.png) | ![RCAN-BAM *(SI)*](./src/example_images/output/SI_rcan_bam_example_shell_40px_4x.png) | ![RealESRGAN](./src/example_images/output/RealESRGAN_example_shell_40px_4x.png) | ![Anime4K](./src/example_images/output/Anime4K_example_shell_40px_4x.png) |

| HSDBTRE | hqx | NEDI <sup>*(m = 4)*</sup> | Super xBR |
| :---: | :---: | :---: | :---: |
| ![HSDBTRE](./src/example_images/output/HSDBTRE_example_shell_40px_4x.png) | ![hqx](./src/example_images/output/hqx_example_shell_40px_4x.png) | ![NEDI <sup>*(m = 4)*</sup>](./src/example_images/output/NEDI_example_shell_40px_4x.png) | ![Super xBR](./src/example_images/output/Super_xBR_example_shell_40px_4x.png) |

| xBRZ | FSR | CAS <sup>*(sharpness = 0.5)*</sup> |
| :---: | :---: | :---: |
| ![xBRZ](./src/example_images/output/xBRZ_example_shell_40px_4x.png) | ![FSR](./src/example_images/output/example_shell_40px_FSR.png) | ![CAS <sup>*(sharpness = 0.5)*</sup>](./src/example_images/output/example_shell_40px_CAS.png) |

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
- **WebUI**, **Scaling App** and **HSDBTRE** AI hybrid created by [***Mikołaj Pokora***](https://github.com/MikiP98)
- **API backend** and **xBRZ wheel** by [***Piotr Przetacki***](https://github.com/PiotrPrzetacki)
- [**Anime4K**](https://github.com/TianZerL/pyanime4k) implementation by [TianZer (TianZerL)](https://github.com/TianZerL)
- [**RealESRGAN**](https://github.com/ai-forever/Real-ESRGAN) implementation by [ai-forever](https://github.com/ai-forever)
- [**NEDI**](https://github.com/Kirstihly/Edge-Directed_Interpolation) implementation by [Ley (Kirstihly)](https://github.com/Kirstihly)
- [**hqx**](https://pypi.org/project/hqx/) implementation by [whoatemybutter](https://pypi.org/user/whoatemybutter/)
- [**xBRZ**](https://github.com/ioistired/xbrz.py) implementation by [ioistired](https://github.com/ioistired)
- [**Super xBR**](https://github.com/MikiP98/py-super-xbr) implementation originally created by [Matt Schwartz (n0spaces)](https://github.com/n0spaces) corrected by [Mikołaj Pokora](https://github.com/MikiP98)
- [**FSR**](https://gpuopen.com/fidelityfx-superresolution/) and [**CAS**](https://gpuopen.com/fidelityfx-cas/) are implemented using [FidelityFX-CLI](https://github.com/GPUOpen-Effects/FidelityFX-CLI) by [GPUOpen-Effects](https://github.com/GPUOpen-Effects) and [AMD](https://www.amd.com/) <sup>*(licence in "src/FidelityFX-CLI-v1.0.3")*</sup>
- ***Area***, ***Bicubic***, ***Bilinear***, ***Lanchos*** and ***Nearest neighbor*** algorithms are implemented using [OpenCV](https://opencv.org)
- ***EDSR***, ***ESPCN***, ***FSRCNN***, ***FSRCNN-small***, ***LapSRN*** AI algorithms are also implemented using [OpenCV](https://opencv.org)
- ***Nearest neighbor***, ***Bilinear***, ***Bicubic*** and ***Lanchos*** algorithms are also implemented using [Pillow library](https://pillow.readthedocs.io/en/stable/)
- ***A2N***, ***AWSRN-BAM***, ***CARN***, ***CARN-BAM***, ***DRLN***, ***DRLN-BAM***, ***EDSR***, ***EDSR-base***, ***HAN***, ***MDSR***, ***MDSR-BAM***, .*..hyyy*
- ***MSRN***, ***MSRN-BAM***, ***PAN***, ***PAN-BAM***, ***RCAN-BAM*** AI algorithms are implemented using [super-image](https://pypi.org/project/super-image/) by [eugenesiow (Eugene Siow)](https://pypi.org/user/eugenesiow/) and [Freed Wu](https://pypi.org/user/Freed-Wu/)

<sup>

- **Download** icon: <a href="https://www.flaticon.com/free-icons/install" title="install icons">Install icons created by NajmunNahar - Flaticon</a>

- **Web GUI** icon: <a href="https://www.flaticon.com/free-icons/interface" title="interface icons">Interface icons created by Freepik - Flaticon</a>

- **Console** icon: <a target="_blank" href="https://icons8.com/icon/nRH1nzeThlgk/console">Console</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a>

</sup>
