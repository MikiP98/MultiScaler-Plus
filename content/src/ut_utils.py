# coding=utf-8
from utils import pil_fully_supported_formats, pil_read_only_formats


def generate_markdown_for_supported_read_formats():
    read_supported_formats: dict[str, list[str]] = pil_fully_supported_formats | pil_read_only_formats
    read_supported_formats = dict(sorted(read_supported_formats.items()))
    print("Supported read formats:")
    print("```")
    half = round(len(read_supported_formats) / 2)
    for format, extensions in read_supported_formats.items():
        print(f"  - **{format}** *(.{', .'.join(extensions)})*")
        if format == list(read_supported_formats.keys())[half - 1]:
            print("---")


def generate_markdown_for_example_images(split=True):
    algorithms = [
        "Original",

        "Nearest Neighbour",
        "Bilinear *(PIL)*", "Bicubic *(PIL)*", "Lanczos *(PIL)*", "Hamming *(PIL)*",
        "Bilinear *(CV2)*", "Bicubic *(CV2)*", "Lanczos *(CV2)*",

        "EDSR *(CV2)*", "ESPCN *(CV2)*", "FSRCNN *(CV2)*", "FSRCNN-small *(CV2)*", "LapSRN *(CV2)*",

        "A2N *(SI)*", "AWSRN-BAM *(SI)*", "CARN *(SI)*", "CARN-BAM *(SI)*", "DRLN *(SI)*", "DRLN-BAM *(SI)*",
        "EDSR *(SI)*", "EDSR-base *(SI)*", "HAN *(SI)*", "MDSR *(SI)*", "MDSR-BAM *(SI)*", "MSRN *(SI)*",
        "MSRN-BAM *(SI)*", "PAN *(SI)*", "PAN-BAM *(SI)*", "RCAN-BAM *(SI)*",

        "RealESRGAN", "Anime4K", "HSDBTRE",

        "hqx", "NEDI <sup>*(m = 4)*</sup>", "Super xBR", "xBRZ",

        "FSR", "CAS <sup>*(sharpness = 0.5)*</sup>",

        "Repetition"
    ]
    links = [
        "https://upload.wikimedia.org/wikipedia/commons/a/a6/160_by_160_thumbnail_of_%27Green_Sea_Shell%27.png",

        "../../example_images/output/example_shell_40px/CV2_INTER_NEAREST_example_shell_40px_4x.webp",

        "../../example_images/output/example_shell_40px/PIL_BILINEAR_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/PIL_BICUBIC_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/PIL_LANCZOS_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/PIL_HAMMING_example_shell_40px_4x.webp",

        "../../example_images/output/example_shell_40px/CV2_INTER_LINEAR_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/CV2_INTER_CUBIC_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/CV2_INTER_LANCZOS4_example_shell_40px_4x.webp",

        "../../example_images/output/example_shell_40px/CV2_EDSR_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/CV2_ESPCN_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/CV2_FSRCNN_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/CV2_FSRCNN_small_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/CV2_LapSRN_example_shell_40px_4x.webp",

        "../../example_images/output/example_shell_40px/SI_a2n_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/SI_awsrn_bam_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/SI_carn_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/SI_carn_bam_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/SI_drln_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/SI_drln_bam_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/SI_edsr_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/SI_edsr_base_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/SI_han_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/SI_mdsr_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/SI_mdsr_bam_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/SI_msrn_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/SI_msrn_bam_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/SI_pan_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/SI_pan_bam_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/SI_rcan_bam_example_shell_40px_4x.webp",

        "../../example_images/output/example_shell_40px/RealESRGAN_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/Anime4K_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/HSDBTRE_example_shell_40px_4x.webp",

        "../../example_images/output/example_shell_40px/hqx_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/NEDI_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/Super_xBR_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/xBRZ_example_shell_40px_4x.webp",

        "../../example_images/output/example_shell_40px/FSR_example_shell_40px_4x.webp",
        "../../example_images/output/example_shell_40px/CAS_example_shell_40px_4x.webp",

        "../../example_images/output/example_shell_40px/Repetition_example_shell_40px_4x.webp"
    ]

    columns = 4
    print("Example images:")
    print("```")
    print(f"| {' | '.join(algorithms[:columns])} |")
    print(f"| {' | '.join([':---:'] * columns)} |")
    print(
        f"| {' | '.join(
            [f'![{algorithm}]({link})' for algorithm, link in zip(algorithms[:columns], links[:columns])]
        )} |"
    )
    for i in range(columns, len(algorithms), columns):
        if not split:
            print(f"| {' | '.join(algorithms[i:i + columns])} |")
            print(
                f"| {' | '.join(
                    [
                        f'![{algorithm}]({link})' for algorithm, link in zip(
                            algorithms[i:i + columns], links[i:i + columns]
                        )
                    ]
                )} |"
            )
        else:
            if columns > len(algorithms) - i:
                columns = len(algorithms) - i
            print()
            print(f"| {' | '.join(algorithms[i:i + columns])} |")
            print(f"| {' | '.join([':---:'] * columns)} |")
            print(
                f"| {' | '.join(
                    [
                        f'![{algorithm}]({link})' for algorithm, link in zip(
                            algorithms[i:i + columns], links[i:i + columns]
                        )
                    ]
                )} |"
            )


if __name__ == "__main__":
    # generate_markdown_for_supported_read_formats()
    generate_markdown_for_example_images()
    ...
