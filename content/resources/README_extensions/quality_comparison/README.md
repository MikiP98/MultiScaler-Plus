# Quality comparisons between algorithms

[Back to main README](../../../../README.md)

## Summary:

*(best algorithms from every category plus unique ones)*

<br>

### Wiki Example Shell *(40px -> 160px)*:

Scaled down image *(40px)*: <br>
![Wiki Example Shell - Small](../../example_images/input/example_shell_40px.png)

|                                                      Original                                                      |                                                 Nearest Neighbour                                                 |                                          Bicubic *(PIL)*                                          |                                          Lanczos *(PIL)*                                          |
|:------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------:|
| ![Original](https://upload.wikimedia.org/wikipedia/commons/a/a6/160_by_160_thumbnail_of_%27Green_Sea_Shell%27.png) | ![Nearest Neighbour](../../example_images/output/example_shell_40px/CV2_INTER_NEAREST_example_shell_40px_4x.webp) | ![Bicubic](../../example_images/output/example_shell_40px/PIL_BICUBIC_example_shell_40px_4x.webp) | ![Lanczos](../../example_images/output/example_shell_40px/PIL_LANCZOS_example_shell_40px_4x.webp) |


|                           DRLN<sup>*(-BAM if <4x)*</sup> *(SI)*                            |                                             RealESRGAN                                              |                                            Anime4K                                            |                                            HSDBTRE                                            |
|:------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------:|
| ![DRLN](../../example_images/output/example_shell_40px/SI_drln_example_shell_40px_4x.webp) | ![RealESRGAN](../../example_images/output/example_shell_40px/RealESRGAN_example_shell_40px_4x.webp) | ![Anime4K](../../example_images/output/example_shell_40px/Anime4K_example_shell_40px_4x.webp) | ![HSDBTRE](../../example_images/output/example_shell_40px/HSDBTRE_example_shell_40px_4x.webp) |


|                                NEDI <sup>*(m = 4)*</sup>                                |                                             Super xBR                                             |                                          xBRZ                                           |                                       FSR *1.1*                                       |
|:---------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------:|
| ![NEDI](../../example_images/output/example_shell_40px/NEDI_example_shell_40px_4x.webp) | ![Super xBR](../../example_images/output/example_shell_40px/Super_xBR_example_shell_40px_4x.webp) | ![xBRZ](../../example_images/output/example_shell_40px/xBRZ_example_shell_40px_4x.webp) | ![FSR](../../example_images/output/example_shell_40px/FSR_example_shell_40px_4x.webp) |

<br>

### Wiki Example Shell *(160px -> 640px)*:

|                                    Original                                    |                                                Nearest Neighbour                                                |
|:------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:|
| ![Original](../../example_images/other/Green_sea_shell_original_crop_640.webp) | ![Nearest Neighbour *(CV2)*](../../example_images/output/160_Sea_Shell/CV2_INTER_NEAREST_160_Sea_Shell_4x.webp) |

|                                         Bicubic *(PIL)*                                         |                                         Lanchos *(PIL)*                                         |
|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| ![Bicubic *(PIL)*](../../example_images/output/160_Sea_Shell/PIL_BICUBIC_160_Sea_Shell_4x.webp) | ![Lanchos *(PIL)*](../../example_images/output/160_Sea_Shell/PIL_LANCZOS_160_Sea_Shell_4x.webp) |

|                                       DRLN<sup>*(-BAM if <4x)*</sup> *(SI)*                                       |                                        RealESRGAN                                         |
|:-----------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------:|
| ![DRLN<sup>*(-BAM if <4x)*</sup> *(SI)*](../../example_images/output/160_Sea_Shell/SI_drln_160_Sea_Shell_4x.webp) | ![RealESRGAN](../../example_images/output/160_Sea_Shell/RealESRGAN_160_Sea_Shell_4x.webp) |

|                                       Anime4K                                       |                                       HSDBTRE                                       |
|:-----------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------:|
| ![Anime4K](../../example_images/output/160_Sea_Shell/Anime4K_160_Sea_Shell_4x.webp) | ![HSDBTRE](../../example_images/output/160_Sea_Shell/HSDBTRE_160_Sea_Shell_4x.webp) |

|                                     NEDI <sup>*(m = 4)*</sup>                                      |                                        Super xBR                                        |
|:--------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------:|
| ![NEDI <sup>*(m = 4)*</sup>](../../example_images/output/160_Sea_Shell/NEDI_160_Sea_Shell_4x.webp) | ![Super xBR](../../example_images/output/160_Sea_Shell/Super_xBR_160_Sea_Shell_4x.webp) |

|                                     xBRZ                                      |                                     FSR *1.1*                                     |
|:-----------------------------------------------------------------------------:|:---------------------------------------------------------------------------------:|
| ![xBRZ](../../example_images/output/160_Sea_Shell/xBRZ_160_Sea_Shell_4x.webp) | ![FSR *1.1*](../../example_images/output/160_Sea_Shell/FSR_160_Sea_Shell_4x.webp) |

<br>

### Wiki example text *(40x109 -> 160x436)*:

images coming soon

<br> <br>

## All:
*(All algorithms one after another)*

<br>

### Wiki Example Shell (40 -> 160):

|                                                      Original                                                      |                                                     Nearest Neighbour                                                     |                                              Bilinear *(PIL)*                                               |                                              Bicubic *(PIL)*                                              |
|:------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|
| ![Original](https://upload.wikimedia.org/wikipedia/commons/a/a6/160_by_160_thumbnail_of_%27Green_Sea_Shell%27.png) | ![Nearest Neighbour *(CV2)*](../../example_images/output/example_shell_40px/CV2_INTER_NEAREST_example_shell_40px_4x.webp) | ![Bilinear *(PIL)*](../../example_images/output/example_shell_40px/PIL_BILINEAR_example_shell_40px_4x.webp) | ![Bicubic *(PIL)*](../../example_images/output/example_shell_40px/PIL_BICUBIC_example_shell_40px_4x.webp) |

|                                              Lanczos *(PIL)*                                              |                                              Hamming *(PIL)*                                              |                                                Bilinear *(CV2)*                                                 |                                                Bicubic *(CV2)*                                                |
|:---------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------:|
| ![Lanczos *(PIL)*](../../example_images/output/example_shell_40px/PIL_LANCZOS_example_shell_40px_4x.webp) | ![Hamming *(PIL)*](../../example_images/output/example_shell_40px/PIL_HAMMING_example_shell_40px_4x.webp) | ![Bilinear *(CV2)*](../../example_images/output/example_shell_40px/CV2_INTER_LINEAR_example_shell_40px_4x.webp) | ![Bicubic *(CV2)*](../../example_images/output/example_shell_40px/CV2_INTER_CUBIC_example_shell_40px_4x.webp) |

|                                                 Lanczos *(CV2)*                                                  |                                            EDSR *(CV2)*                                             |                                             ESPCN *(CV2)*                                             |                                             FSRCNN *(CV2)*                                              |
|:----------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------:|
| ![Lanczos *(CV2)*](../../example_images/output/example_shell_40px/CV2_INTER_LANCZOS4_example_shell_40px_4x.webp) | ![EDSR *(CV2)*](../../example_images/output/example_shell_40px/CV2_EDSR_example_shell_40px_4x.webp) | ![ESPCN *(CV2)*](../../example_images/output/example_shell_40px/CV2_ESPCN_example_shell_40px_4x.webp) | ![FSRCNN *(CV2)*](../../example_images/output/example_shell_40px/CV2_FSRCNN_example_shell_40px_4x.webp) |

|                                                FSRCNN-small *(CV2)*                                                 |                                             LapSRN *(CV2)*                                              |                                           A2N *(SI)*                                            |                                              AWSRN-BAM *(SI)*                                               |
|:-------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------:|
| ![FSRCNN-small *(CV2)*](../../example_images/output/example_shell_40px/CV2_FSRCNN_small_example_shell_40px_4x.webp) | ![LapSRN *(CV2)*](../../example_images/output/example_shell_40px/CV2_LapSRN_example_shell_40px_4x.webp) | ![A2N *(SI)*](../../example_images/output/example_shell_40px/SI_a2n_example_shell_40px_4x.webp) | ![AWSRN-BAM *(SI)*](../../example_images/output/example_shell_40px/SI_awsrn_bam_example_shell_40px_4x.webp) |

|                                            CARN *(SI)*                                            |                                              CARN-BAM *(SI)*                                              |                                            DRLN *(SI)*                                            |                                              DRLN-BAM *(SI)*                                              |
|:-------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|
| ![CARN *(SI)*](../../example_images/output/example_shell_40px/SI_carn_example_shell_40px_4x.webp) | ![CARN-BAM *(SI)*](../../example_images/output/example_shell_40px/SI_carn_bam_example_shell_40px_4x.webp) | ![DRLN *(SI)*](../../example_images/output/example_shell_40px/SI_drln_example_shell_40px_4x.webp) | ![DRLN-BAM *(SI)*](../../example_images/output/example_shell_40px/SI_drln_bam_example_shell_40px_4x.webp) |

|                                            EDSR *(SI)*                                            |                                              EDSR-base *(SI)*                                               |                                           HAN *(SI)*                                            |                                            MDSR *(SI)*                                            |
|:-------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------:|
| ![EDSR *(SI)*](../../example_images/output/example_shell_40px/SI_edsr_example_shell_40px_4x.webp) | ![EDSR-base *(SI)*](../../example_images/output/example_shell_40px/SI_edsr_base_example_shell_40px_4x.webp) | ![HAN *(SI)*](../../example_images/output/example_shell_40px/SI_han_example_shell_40px_4x.webp) | ![MDSR *(SI)*](../../example_images/output/example_shell_40px/SI_mdsr_example_shell_40px_4x.webp) |

|                                              MDSR-BAM *(SI)*                                              |                                            MSRN *(SI)*                                            |                                              MSRN-BAM *(SI)*                                              |                                           PAN *(SI)*                                            |
|:---------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| ![MDSR-BAM *(SI)*](../../example_images/output/example_shell_40px/SI_mdsr_bam_example_shell_40px_4x.webp) | ![MSRN *(SI)*](../../example_images/output/example_shell_40px/SI_msrn_example_shell_40px_4x.webp) | ![MSRN-BAM *(SI)*](../../example_images/output/example_shell_40px/SI_msrn_bam_example_shell_40px_4x.webp) | ![PAN *(SI)*](../../example_images/output/example_shell_40px/SI_pan_example_shell_40px_4x.webp) |

|                                             PAN-BAM *(SI)*                                              |                                              RCAN-BAM *(SI)*                                              |                                             RealESRGAN                                              |                                            Anime4K                                            |
|:-------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------:|
| ![PAN-BAM *(SI)*](../../example_images/output/example_shell_40px/SI_pan_bam_example_shell_40px_4x.webp) | ![RCAN-BAM *(SI)*](../../example_images/output/example_shell_40px/SI_rcan_bam_example_shell_40px_4x.webp) | ![RealESRGAN](../../example_images/output/example_shell_40px/RealESRGAN_example_shell_40px_4x.webp) | ![Anime4K](../../example_images/output/example_shell_40px/Anime4K_example_shell_40px_4x.webp) |

|                                            HSDBTRE                                            |                                          hqx                                          |                                          NEDI <sup>*(m = 4)*</sup>                                           |                                             Super xBR                                             |
|:---------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------:|
| ![HSDBTRE](../../example_images/output/example_shell_40px/HSDBTRE_example_shell_40px_4x.webp) | ![hqx](../../example_images/output/example_shell_40px/hqx_example_shell_40px_4x.webp) | ![NEDI <sup>*(m = 4)*</sup>](../../example_images/output/example_shell_40px/NEDI_example_shell_40px_4x.webp) | ![Super xBR](../../example_images/output/example_shell_40px/Super_xBR_example_shell_40px_4x.webp) |

|                                          xBRZ                                           |                                          FSR                                          |                                          CAS <sup>*(sharpness = 0.5)*</sup>                                          |                                             Repetition                                              |
|:---------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------:|
| ![xBRZ](../../example_images/output/example_shell_40px/xBRZ_example_shell_40px_4x.webp) | ![FSR](../../example_images/output/example_shell_40px/FSR_example_shell_40px_4x.webp) | ![CAS <sup>*(sharpness = 0.5)*</sup>](../../example_images/output/example_shell_40px/CAS_example_shell_40px_4x.webp) | ![Repetition](../../example_images/output/example_shell_40px/Repetition_example_shell_40px_4x.webp) |

<br>

### Wiki example text *(40x109 -> 160x436)*:

images coming soon

<br> <br>

## Recommendations:

### Downscaling:

Downscaling is simpler that's why it's first  
In theory the best algorithm to use, supported by this APP is **PIL**'s implementation of `Lanchos` algorithm  
Second best in theory is **PIL**'s implementation of `Bicubic` algorithm  
In practice though the differance is that **Lanchos** will have **sharper** and **contrastier** look, 
  but sometimes it looks like it has some over-sharping artifacts  
If you are looking for even softer look try `Area Average` implementation by **CV2**
The rest of algorithms might be used as an artistic choice, sometimes with cool and interesting results

### Upscaling:

If you want to quickly scale some images with anything at 
  least a bit better than a default bilinear scaler present in most application, 
  chose either: `Bicubic` or `Lanchos`  
Though **Lanchos** is in theory better it sometimes looks over sharpened and over contrasted 
  in comparison to **Bicubic**  
`FSR` can also result in better image as it better preserves the overall shape of the object in the image, 
  but it will add some noise and grain to the output

If you wish to get the best possible results from the upscaling you can choose 1 of 2 path:

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

<br>

### [Back to main README](../../../../README.md)

<br>

*<sup>
This file contains shell images that are derived from works licensed under Creative Commons Attribution-ShareAlike 2.5 and 2.0.<br>
These images, including any modifications, are licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
</sup>*