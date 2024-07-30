# coding=utf-8


default_configs = {
    "loader": {
        'clear_output_dir': True,
        'copy_mcmeta': True
    },
    "filter": {
        "simple_config": {
            "formats": ["PNG"],
            "compressions": [
                {
                    "additional_lossless": True,
                    "lossless": True
                }
            ],
            "add_compression_to_name": False
        },

        "add_factor_to_name": False,
        "sort_by_factor": True
    },
}


def get_loader_config() -> dict:
    return default_configs["loader"]
