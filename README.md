# AutoUpscale

## Universal app for scaling images

AutoUpscale is a universal app for scaling images. It can be used as a command line tool, a web GUI, or a console application.

### Supported algorithms
- **Classical Algorithms** <sup>(Up-scaling and downscaling)</sup>:
  - **Bicubic** *(Better than bilinear, less blur, more detail)*
  - **Bilinear** *(Second-simplest algorithm, most common and most blurry)*
  - **Nearest neighbor** *(Simplest algorithm, duplicates or keeps only the nearest pixel)*
  - **Lanczos** *(Similar to bicubic, but with a higher contrast)*
- **AI-based algorithms** <sup>(Only up-scaling)</sup>:
  - **RealESRGAN** *(AI-based algorithm, based on ESRGAN)*
- **Edge detection algorithms** <sup>(Only up-scaling)</sup>:
  - **xBRZ** *(edge detection algorithm, based on xBR, better at preserving small details)*

<br/>

### Additional Credits:
- [**RealESRGAN**](https://github.com/ai-forever/Real-ESRGAN) implementation by [ai-forever](https://github.com/ai-forever)
- [**xBRZ**](https://github.com/ioistired/xbrz.py) implementation by [ioistired](https://github.com/ioistired)
- ***Nearest neighbor***, ***Bilinear***, ***Bicubic*** and ***Lanchos*** algorithms are implemented using [Pillow library](https://pillow.readthedocs.io/en/stable/)

<sup>

- Thanks to ***Dinkar Kamat*** for the **Bat To Exe Converter** tool
- **Download** icon: <a href="https://www.flaticon.com/free-icons/install" title="install icons">Install icons created by NajmunNahar - Flaticon</a>
- **Web GUI** icon: <a href="https://www.flaticon.com/free-icons/interface" title="interface icons">Interface icons created by Freepik - Flaticon</a>
- **Console** icon: <a target="_blank" href="https://icons8.com/icon/nRH1nzeThlgk/console">Console</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a>

</sup>
