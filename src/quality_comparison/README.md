# Quality comparisons between algorithms

[Back to main README](../../README.md)

## Summary:

*(best algorithms from every category plus unique ones)*

### Wiki Example Shell *(40px -> 160px)*:

Scaled down image *(40px)*: <br>
![Wiki Example Shell - Small](../example_images/input/example_shell_40px.png)

|                                                      Original                                                      |                                               Nearest Neighbour                                                |                                              Bicubic                                               |                                                Lanczos                                                |
|:------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------:|
| ![Original](https://upload.wikimedia.org/wikipedia/commons/a/a6/160_by_160_thumbnail_of_%27Green_Sea_Shell%27.png) | ![Nearest Neighbour](../example_images/output/example_shell_40px/CV2_INTER_NEAREST_example_shell_40px_4x.webp) | ![Bicubic](../example_images/output/example_shell_40px/CV2_INTER_CUBIC_example_shell_40px_4x.webp) | ![Lanczos](../example_images/output/example_shell_40px/CV2_INTER_LANCZOS4_example_shell_40px_4x.webp) |


|                          DRLN<sup>*(-BAM if <4x)*</sup> *(SI)*                          |                                            RealESRGAN                                            |                                          Anime4K                                           |                                          HSDBTRE                                           |
|:---------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------:|
| ![DRLN](../example_images/output/example_shell_40px/SI_drln_example_shell_40px_4x.webp) | ![RealESRGAN](../example_images/output/example_shell_40px/RealESRGAN_example_shell_40px_4x.webp) | ![Anime4K](../example_images/output/example_shell_40px/Anime4K_example_shell_40px_4x.webp) | ![HSDBTRE](../example_images/output/example_shell_40px/HSDBTRE_example_shell_40px_4x.webp) |


|                              NEDI <sup>*(m = 4)*</sup>                               |                                           Super xBR                                            |                                         xBRZ                                         |                                    FSR *1.1*                                    |
|:------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------:|
| ![NEDI](../example_images/output/example_shell_40px/NEDI_example_shell_40px_4x.webp) | ![Super xBR](../example_images/output/example_shell_40px/Super_xBR_example_shell_40px_4x.webp) | ![xBRZ](../example_images/output/example_shell_40px/xBRZ_example_shell_40px_4x.webp) | ![FSR](../example_images/output/example_shell_40px/example_shell_40px_FSR.webp) |

### Wiki Example Shell *(160px -> 640px)*:

|                                          Nearest Neighbour                                           |                                         Bicubic                                          |
|:----------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------:|
| ![Nearest Neighbour](../example_images/output/160_Sea_Shell/CV2_INTER_NEAREST_160_Sea_Shell_4x.webp) | ![Bicubic](../example_images/output/160_Sea_Shell/CV2_INTER_CUBIC_160_Sea_Shell_4x.webp) |

|                     DRLN<sup>*(-BAM if <4x)*</sup> *(SI)*                     |                                       RealESRGAN                                       |
|:-----------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------:|
| ![DRLN](../example_images/output/160_Sea_Shell/SI_drln_160_Sea_Shell_4x.webp) | ![RealESRGAN](../example_images/output/160_Sea_Shell/RealESRGAN_160_Sea_Shell_4x.webp) |

|                                     Anime4K                                      |                                     HSDBTRE                                      |
|:--------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------:|
| ![Anime4K](../example_images/output/160_Sea_Shell/Anime4K_160_Sea_Shell_4x.webp) | ![HSDBTRE](../example_images/output/160_Sea_Shell/HSDBTRE_160_Sea_Shell_4x.webp) |

|                         NEDI <sup>*(m = 4)*</sup>                          |                                      Super xBR                                       |
|:--------------------------------------------------------------------------:|:------------------------------------------------------------------------------------:|
| ![NEDI](../example_images/output/160_Sea_Shell/NEDI_160_Sea_Shell_4x.webp) | ![Super xBR](../example_images/output/160_Sea_Shell/Super_xBR_160_Sea_Shell_4x.webp) |

|                                    xBRZ                                    |                               FSR *1.1*                               |
|:--------------------------------------------------------------------------:|:---------------------------------------------------------------------:|
| ![xBRZ](../example_images/output/160_Sea_Shell/xBRZ_160_Sea_Shell_4x.webp) | ![FSR](../example_images/output/160_Sea_Shell/160_Sea_Shell_FSR.webp) |

### Wiki example text *(40x109 -> 160x436)*:

images coming soon

## All:
*(All algorithms one after another)*

### Wiki Exaple Shell (40 -> 160):

|                                                      Original                                                      |                                               Nearest Neighbour                                                |                                               Bilinear                                               |                                              Bicubic                                               |
|:------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------:|
| ![Original](https://upload.wikimedia.org/wikipedia/commons/a/a6/160_by_160_thumbnail_of_%27Green_Sea_Shell%27.png) | ![Nearest Neighbour](../example_images/output/example_shell_40px/CV2_INTER_NEAREST_example_shell_40px_4x.webp) | ![Bilinear](../example_images/output/example_shell_40px/CV2_INTER_LINEAR_example_shell_40px_4x.webp) | ![Bicubic](../example_images/output/example_shell_40px/CV2_INTER_CUBIC_example_shell_40px_4x.webp) |

|                                                Lanczos                                                |                                           EDSR *(CV2)*                                           |                                           ESPCN *(CV2)*                                            |                                            FSRCNN *(CV2)*                                            |
|:-----------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------:|
| ![Lanczos](../example_images/output/example_shell_40px/CV2_INTER_LANCZOS4_example_shell_40px_4x.webp) | ![EDSR *(CV2)*](../example_images/output/example_shell_40px/CV2_EDSR_example_shell_40px_4x.webp) | ![ESPCN *(CV2)*](../example_images/output/example_shell_40px/CV2_ESPCN_example_shell_40px_4x.webp) | ![FSRCNN *(CV2)*](../example_images/output/example_shell_40px/CV2_FSRCNN_example_shell_40px_4x.webp) |

|                                               FSRCNN-small *(CV2)*                                               |                                            LapSRN *(CV2)*                                            |                                          A2N *(SI)*                                          |                                             AWSRN-BAM *(SI)*                                             |
|:----------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------:|
| ![FSRCNN-small *(CV2)*](../example_images/output/example_shell_40px/CV2_FSRCNN_small_example_shell_40px_4x.webp) | ![LapSRN *(CV2)*](../example_images/output/example_shell_40px/CV2_LapSRN_example_shell_40px_4x.webp) | ![A2N *(SI)*](../example_images/output/example_shell_40px/SI_a2n_example_shell_40px_4x.webp) | ![AWSRN-BAM *(SI)*](../example_images/output/example_shell_40px/SI_awsrn_bam_example_shell_40px_4x.webp) |

|                                          CARN *(SI)*                                           |                                            CARN-BAM *(SI)*                                             |                                          DRLN *(SI)*                                           |                                            DRLN-BAM *(SI)*                                             |
|:----------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------:|
| ![CARN *(SI)*](../example_images/output/example_shell_40px/SI_carn_example_shell_40px_4x.webp) | ![CARN-BAM *(SI)*](../example_images/output/example_shell_40px/SI_carn_bam_example_shell_40px_4x.webp) | ![DRLN *(SI)*](../example_images/output/example_shell_40px/SI_drln_example_shell_40px_4x.webp) | ![DRLN-BAM *(SI)*](../example_images/output/example_shell_40px/SI_drln_bam_example_shell_40px_4x.webp) |

|                                          EDSR *(SI)*                                           |                                             EDSR-base *(SI)*                                             |                                          HAN *(SI)*                                          |                                          MDSR *(SI)*                                           |
|:----------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:|
| ![EDSR *(SI)*](../example_images/output/example_shell_40px/SI_edsr_example_shell_40px_4x.webp) | ![EDSR-base *(SI)*](../example_images/output/example_shell_40px/SI_edsr_base_example_shell_40px_4x.webp) | ![HAN *(SI)*](../example_images/output/example_shell_40px/SI_han_example_shell_40px_4x.webp) | ![MDSR *(SI)*](../example_images/output/example_shell_40px/SI_mdsr_example_shell_40px_4x.webp) |

|                                            MDSR-BAM *(SI)*                                             |                                          MSRN *(SI)*                                           |                                            MSRN-BAM *(SI)*                                             |                                          PAN *(SI)*                                          |
|:------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|
| ![MDSR-BAM *(SI)*](../example_images/output/example_shell_40px/SI_mdsr_bam_example_shell_40px_4x.webp) | ![MSRN *(SI)*](../example_images/output/example_shell_40px/SI_msrn_example_shell_40px_4x.webp) | ![MSRN-BAM *(SI)*](../example_images/output/example_shell_40px/SI_msrn_bam_example_shell_40px_4x.webp) | ![PAN *(SI)*](../example_images/output/example_shell_40px/SI_pan_example_shell_40px_4x.webp) |

|                                            PAN-BAM *(SI)*                                            |                                            RCAN-BAM *(SI)*                                             |                                            RealESRGAN                                            |                                          Anime4K                                           |
|:----------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------:|
| ![PAN-BAM *(SI)*](../example_images/output/example_shell_40px/SI_pan_bam_example_shell_40px_4x.webp) | ![RCAN-BAM *(SI)*](../example_images/output/example_shell_40px/SI_rcan_bam_example_shell_40px_4x.webp) | ![RealESRGAN](../example_images/output/example_shell_40px/RealESRGAN_example_shell_40px_4x.webp) | ![Anime4K](../example_images/output/example_shell_40px/Anime4K_example_shell_40px_4x.webp) |

|                                          HSDBTRE                                           |                                        hqx                                         |                                         NEDI <sup>*(m = 4)*</sup>                                         |                                           Super xBR                                            |
|:------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:|
| ![HSDBTRE](../example_images/output/example_shell_40px/HSDBTRE_example_shell_40px_4x.webp) | ![hqx](../example_images/output/example_shell_40px/hqx_example_shell_40px_4x.webp) | ![NEDI <sup>*(m = 4)*</sup>](../example_images/output/example_shell_40px/NEDI_example_shell_40px_4x.webp) | ![Super xBR](../example_images/output/example_shell_40px/Super_xBR_example_shell_40px_4x.webp) |

|                                         xBRZ                                         |                                       FSR                                       |                                       CAS <sup>*(sharpness = 0.5)*</sup>                                       |
|:------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|
| ![xBRZ](../example_images/output/example_shell_40px/xBRZ_example_shell_40px_4x.webp) | ![FSR](../example_images/output/example_shell_40px/example_shell_40px_FSR.webp) | ![CAS <sup>*(sharpness = 0.5)*</sup>](../example_images/output/example_shell_40px/example_shell_40px_CAS.webp) |

### Wiki example text:

images coming soon

<br>

## Recommendations:

coming soon