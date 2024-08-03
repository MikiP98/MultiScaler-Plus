# coding=utf-8


default_configs = {
    "loader": {
        'clear_output_dir': True,
        'copy_mcmeta': True
    },
    "saver": {
        "simple_config": {
            "formats": ["WEBP"],
            "compressions": [
                {
                    "additional_lossless": True,
                    "lossless": True,
                    "quality": 95
                }
            ],
            "add_compression_to_name": False
        },

        "add_factor_to_name": True,
        "sort_by_factor": False,

        # A.K.A. algorithm or filter
        "add_processing_method_to_name": True,
        "sort_by_processing_method": False,

        "sort_by_image": False,
        "sort_by_file_extension": -1,  # -1 - auto, 0 - no, 1 - yes
        # TODO: Add more auto options

        "factors": None,
        "processing_methods": None
    },
    "scaler": {
        # prevents multi-face (in 1 image) textures to expand over current textures border
        'texture_outbound_protection': False,
        # prevents multi-face (in 1 image) textures to not fully cover current textures border
        'texture_inbound_protection': False,
        # What should be used to make the mask, 1st is when alpha is present, 2nd when it is not  TODO: add more options
        'texture_mask_mode': ('alpha', 'black'),
        # if true, the alpha channel will be equal to 255 or alpha will be deleted
        'disallow_partial_transparency': False,

        'try_to_fix_texture_tiling': False,
        'tiling_fix_quality': 1.0,

        'sharpness': 0.5,
        'NEDI_m': 4,
        'offset_x': 0.5,
        'offset_y': 0.5
    }
}

# 'multiprocessing_levels': {},
# 'max_processes': (2, 2, 2),
# 'override_processes_count': False,
# # If True, max_processes will set the Exact number of processes, instead of the Maximum number of them


def get_loader_config() -> tuple[dict, bool]:
    return default_configs["loader"], True


def get_saver_config() -> tuple[dict, bool]:
    return default_configs["saver"], True


def get_scaler_config() -> tuple[dict, bool]:
    return default_configs["scaler"], True
