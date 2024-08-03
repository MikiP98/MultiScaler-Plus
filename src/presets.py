from abc import ABC
from utils import Algorithms


class Preset(ABC):
    config = {
        'clear_output_directory': True,

        'add_algorithm_name_to_output_files_names': True,
        'add_factor_to_output_files_names': True,

        'sort_by_algorithm': False,
        'sort_by_scale': False,
        'sort_by_image': False,
        'sort_by_file_extension': -1,  # -1 - auto, 0 - no, 1 - yes

        'file_formats': {"WEBP"},
        'lossless_compression': True,
        'additional_lossless_compression': True,
        'quality': 95,

        'multiprocessing_levels': {},
        'max_processes': (2, 2, 2),
        'override_processes_count': False,
        # If True, max_processes will set the Exact number of processes, instead of the Maximum number of them

        'copy_mcmeta': True,
        'texture_outbound_protection': False,
        # prevents multi-face (in 1 image) textures to expand over current textures border
        'texture_inbound_protection': False,
        # TODO: Implement this, prevents multi-face (in 1 image) textures to not fully cover current textures border
        'texture_mask_mode': ('alpha', 'black'),
        # What should be used to make the mask, 1st is when alpha is present, 2nd when it is not  TODO: add more options
        'disallow_partial_transparency': False,
        'try_to_fix_texture_tiling': False,
        'tiling_fix_quality': 1.0,

        'sharpness': 0.5,
        'NEDI_m': 4,
        'offset_x': 0.5,
        'offset_y': 0.5
    }
    algorithms = [Algorithms.CV2_INTER_LINEAR]
    scales = [2]


class FullAlgorithmsTest(Preset):
    config = Preset.config
    algorithms = [
        Algorithms.Anime4K,
        Algorithms.CAS,  # contrast adaptive sharpening
        Algorithms.CV2_INTER_AREA,  # resampling using pixel area relation
        Algorithms.CV2_INTER_CUBIC,  # bicubic interpolation over 4x4 pixel neighborhood
        Algorithms.CV2_INTER_LANCZOS4,  # Lanczos interpolation over 8x8 pixel neighborhood
        Algorithms.CV2_INTER_LINEAR,  # bilinear interpolation
        Algorithms.CV2_INTER_NEAREST,  # nearest-neighbor interpolation
        Algorithms.CV2_EDSR,  # Enhanced Deep Super-Resolution
        Algorithms.CV2_ESPCN,  # Efficient Sub-Pixel Convolutional Neural Network
        Algorithms.CV2_FSRCNN,  # Fast Super-Resolution Convolutional Neural Network
        Algorithms.CV2_FSRCNN_small,  # Fast Super-Resolution Convolutional Neural Network - Small
        Algorithms.CV2_LapSRN,  # Laplacian Super-Resolution Network
        Algorithms.FSR,  # FidelityFX Super Resolution
        Algorithms.hqx,  # high quality scale

        Algorithms.HSDBTRE,

        Algorithms.NEDI,  # New Edge-Directed Interpolation
        Algorithms.PIL_BICUBIC,  # less blur and artifacts than bilinear, but slower
        Algorithms.PIL_BILINEAR,
        Algorithms.PIL_LANCZOS,  # less blur than bicubic, but artifacts may appear
        Algorithms.PIL_NEAREST_NEIGHBOR,
        Algorithms.RealESRGAN,
        Algorithms.Repetition,

        Algorithms.SI_drln_bam,
        Algorithms.SI_edsr,
        Algorithms.SI_msrn,
        Algorithms.SI_mdsr,
        Algorithms.SI_msrn_bam,
        Algorithms.SI_edsr_base,
        Algorithms.SI_mdsr_bam,
        Algorithms.SI_awsrn_bam,
        Algorithms.SI_a2n,
        Algorithms.SI_carn,
        Algorithms.SI_carn_bam,
        Algorithms.SI_pan,
        Algorithms.SI_pan_bam,

        Algorithms.SI_drln,
        Algorithms.SI_han,
        Algorithms.SI_rcan_bam,

        Algorithms.Super_xBR,
        Algorithms.xBRZ,

        # Docker start
        Algorithms.SUPIR,
        Algorithms.Waifu2x,
    ]
    scales = [4, 0.25]


class FullUpscaleTest(Preset):
    config = Preset.config
    algorithms = [
        Algorithms.Anime4K,
        Algorithms.CAS,  # contrast adaptive sharpening
        Algorithms.CV2_INTER_CUBIC,  # bicubic interpolation over 4x4 pixel neighborhood
        Algorithms.CV2_INTER_LANCZOS4,  # Lanczos interpolation over 8x8 pixel neighborhood
        Algorithms.CV2_INTER_LINEAR,  # bilinear interpolation
        Algorithms.CV2_INTER_NEAREST,  # nearest-neighbor interpolation
        Algorithms.CV2_EDSR,  # Enhanced Deep Super-Resolution
        Algorithms.CV2_ESPCN,  # Efficient Sub-Pixel Convolutional Neural Network
        Algorithms.CV2_FSRCNN,  # Fast Super-Resolution Convolutional Neural Network
        Algorithms.CV2_FSRCNN_small,  # Fast Super-Resolution Convolutional Neural Network - Small
        Algorithms.CV2_LapSRN,  # Laplacian Super-Resolution Network
        Algorithms.FSR,  # FidelityFX Super Resolution
        Algorithms.hqx,  # high quality scale

        Algorithms.HSDBTRE,

        Algorithms.NEDI,  # New Edge-Directed Interpolation
        Algorithms.PIL_BICUBIC,  # less blur and artifacts than bilinear, but slower
        Algorithms.PIL_BILINEAR,
        Algorithms.PIL_LANCZOS,  # less blur than bicubic, but artifacts may appear
        Algorithms.PIL_NEAREST_NEIGHBOR,
        Algorithms.RealESRGAN,
        Algorithms.Repetition,

        Algorithms.SI_drln_bam,
        Algorithms.SI_edsr,
        Algorithms.SI_msrn,
        Algorithms.SI_mdsr,
        Algorithms.SI_msrn_bam,
        Algorithms.SI_edsr_base,
        Algorithms.SI_mdsr_bam,
        Algorithms.SI_awsrn_bam,
        Algorithms.SI_a2n,
        Algorithms.SI_carn,
        Algorithms.SI_carn_bam,
        Algorithms.SI_pan,
        Algorithms.SI_pan_bam,

        Algorithms.SI_drln,
        Algorithms.SI_han,
        Algorithms.SI_rcan_bam,

        Algorithms.Super_xBR,
        Algorithms.xBRZ
    ]
    scales = [4]


class SmartUpscaleTest(Preset):
    config = Preset.config
    algorithms = [
        Algorithms.Anime4K,
        Algorithms.CV2_INTER_LINEAR,  # bilinear interpolation
        Algorithms.CV2_FSRCNN_small,  # Fast Super-Resolution Convolutional Neural Network - Small
        Algorithms.FSR,  # FidelityFX Super Resolution
        Algorithms.hqx,  # high quality scale

        Algorithms.HSDBTRE,

        Algorithms.NEDI,  # New Edge-Directed Interpolation
        Algorithms.PIL_BILINEAR,
        Algorithms.RealESRGAN,
        Algorithms.Repetition,

        Algorithms.SI_drln_bam,

        Algorithms.Super_xBR,
        Algorithms.xBRZ
    ]
    scales = [4]


class FullDownScalingTest(Preset):
    config = Preset.config
    algorithms = [
        Algorithms.CV2_INTER_AREA,  # resampling using pixel area relation
        Algorithms.CV2_INTER_CUBIC,  # bicubic interpolation over 4x4 pixel neighborhood
        Algorithms.CV2_INTER_LANCZOS4,  # Lanczos interpolation over 8x8 pixel neighborhood
        Algorithms.CV2_INTER_LINEAR,  # bilinear interpolation
        Algorithms.CV2_INTER_NEAREST,

        Algorithms.PIL_BICUBIC,  # less blur and artifacts than bilinear, but slower
        Algorithms.PIL_BILINEAR,
        Algorithms.PIL_LANCZOS,  # less blur than bicubic, but artifacts may appear
        Algorithms.PIL_NEAREST_NEIGHBOR,

        Algorithms.Repetition
    ]
    scales = [0.25]


class SmartDownScalingTest(Preset):
    config = Preset.config
    algorithms = [
        Algorithms.CV2_INTER_LINEAR,
        Algorithms.PIL_BILINEAR,

        Algorithms.Repetition
    ]
    scales = [0.25]


class TextureProtectionTest(Preset):
    config = Preset.config.copy()
    config['texture_outbound_protection'] = True
    config['texture_inbound_protection'] = True
    algorithms = [
        Algorithms.CV2_INTER_LINEAR
    ]
    scales = [4]


class TilingsFixTest(Preset):
    config = Preset.config.copy()
    config['try_to_fix_texture_tiling'] = True
    algorithms = [
        Algorithms.xBRZ
    ]
    scales = [4]
