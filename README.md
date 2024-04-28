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

- Thanks to ***Dinkar Kamat*** for the **Bat To Exe Converter** tool
- **Download** icon: <a href="https://www.flaticon.com/free-icons/install" title="install icons">Install icons created by NajmunNahar - Flaticon</a>
- **Web GUI** icon: <a href="https://www.flaticon.com/free-icons/interface" title="interface icons">Interface icons created by Freepik - Flaticon</a>
- **Console** icon: <a target="_blank" href="https://icons8.com/icon/nRH1nzeThlgk/console">Console</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a>

</sup>
