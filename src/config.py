# coding=utf-8


default_configs = {
    "loader": {
        'clear_output_dir': True,
        'copy_mcmeta': True
    },
    "filter": {
        "simple_config": {
            "formats": ["WEBP"],
            "compressions": [
                {
                    "additional_lossless": True,
                    "lossless": True
                }
            ],
            "add_compression_to_name": False
        },

        "add_factor_to_name": True,
        "sort_by_factor": False
    },
}


def get_loader_config() -> tuple[dict, bool]:
    return default_configs["loader"], True
