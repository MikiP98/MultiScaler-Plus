# coding=utf-8
import os.path

from aenum import auto, IntEnum, unique
from typing import Iterable
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

        "Nearest Neighbour *(CV2)*",
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


@unique
class POSITIONS(IntEnum):
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()


def generate_markdown_table(table: Iterable[Iterable[str]], position: POSITIONS = POSITIONS.CENTER) -> str:
    result_builder = ['|']
    headers = [column[0] for column in table]
    result_builder.append('|'.join(headers))
    result_builder.append('|')

    if position == POSITIONS.LEFT:
        positional_string = ":--"
    elif position == POSITIONS.CENTER:
        positional_string = ":-:"
    else:
        positional_string = "--:"

    result_builder.append("\n|")
    for _ in headers:
        result_builder.append(f"{positional_string}|")

    i = 1
    while i < len(table[0]):
        row = [column[i] for column in table]
        result_builder.append("\n|")
        result_builder.append('|'.join(row))
        result_builder.append("|")

        i += 1

    return ''.join(result_builder)


def generate_big_shell_summary(p: str) -> None:
    u = f"{p}/output/160_Sea_Shell"
    big_shell_summary_data: list[tuple[str, str]] = [
        (
            "Original",
            f"{p}/other/Green_sea_shell_original_crop_640.webp"
        ),
        (
            "Nearest Neighbour *(CV2)*",
            f"{u}/CV2_INTER_NEAREST_160_Sea_Shell_4x.webp"
        ),
        (
            "Bicubic *(PIL)*",
            f"{u}/PIL_BICUBIC_160_Sea_Shell_4x.webp"
        ),
        (
            "Lanchos *(PIL)*",
            f"{u}/PIL_LANCZOS_160_Sea_Shell_4x.webp"
        ),
        (
            "DRLN<sup>*(-BAM if <4x)*</sup> *(SI)*",
            f"{u}/SI_drln_160_Sea_Shell_4x.webp"
        ),
        (
            "RealESRGAN",
            f"{u}/RealESRGAN_160_Sea_Shell_4x.webp"
        ),
        (
            "Anime4K",
            f"{u}/Anime4K_160_Sea_Shell_4x.webp"
        ),
        (
            "HSDBTRE",
            f"{u}/HSDBTRE_160_Sea_Shell_4x.webp"
        ),
        (
            "NEDI <sup>*(m = 4)*</sup>",
            f"{u}/NEDI_160_Sea_Shell_4x.webp"
        ),
        (
            "Super xBR",
            f"{u}/Super_xBR_160_Sea_Shell_4x.webp"
        ),
        (
            "xBRZ",
            f"{u}/xBRZ_160_Sea_Shell_4x.webp"
        ),
        (
            "FSR *1.1*",
            f"{u}/FSR_160_Sea_Shell_4x.webp"
        )
    ]

    for i, entry in enumerate(big_shell_summary_data):
        big_shell_summary_data[i] = (entry[0], f"![{entry[0]}]({entry[1]})")

    batches: list[tuple[tuple[str, str], tuple[str, str]]] = []

    for i in range(0, len(big_shell_summary_data), 2):
        batches.append((big_shell_summary_data[i], big_shell_summary_data[i+1]))

    # 100 x `-`
    print("\n----------------------------------------------------------------------------------------------------\n")
    for batch in batches:
        print(generate_markdown_table(batch))
        print()
    print("----------------------------------------------------------------------------------------------------\n")


def generate_small_shell_summary(p: str):
    u = f"{p}/output/example_shell_40px"
    big_shell_summary_data: list[tuple[str, str]] = [
        (
            "Original",
            "https://upload.wikimedia.org/wikipedia/commons/a/a6/160_by_160_thumbnail_of_%27Green_Sea_Shell%27.png"
        ),
        (
            "Nearest Neighbour <sup>*(CV2)*</sup>",
            f"{u}/CV2_INTER_NEAREST_example_shell_40px_4x.webp"
        ),
        (
            "Hamming",
            f"{u}/PIL_HAMMING_example_shell_40px_4x.webp"
        ),
        (
            "Bicubic *(PIL)*",
            f"{u}/PIL_BICUBIC_example_shell_40px_4x.webp"
        ),
        (
            "Lanczos *(PIL)*",
            f"{u}/PIL_LANCZOS_example_shell_40px_4x.webp"
        ),
        (
            "EDSR *(CV2)*",
            f"{u}/CV2_EDSR_example_shell_40px_4x.webp"
        ),
        (
            "DRLN<sup>*(-BAM if <4x)*</sup> *(SI)*",
            f"{u}/SI_drln_example_shell_40px_4x.webp"
        ),
        (
            "RealESRGAN",
            f"{u}/RealESRGAN_example_shell_40px_4x.webp"
        ),
        (
            "Anime4K",
            f"{u}/Anime4K_example_shell_40px_4x.webp"
        ),
        (
            "HSDBTRE",
            f"{u}/HSDBTRE_example_shell_40px_4x.webp"
        ),
        (
            "NEDI <sup>*(m = 4)*</sup>",
            f"{u}/NEDI_example_shell_40px_4x.webp"
        ),
        (
            "Super xBR",
            f"{u}/Super_xBR_example_shell_40px_4x.webp"
        ),
        (
            "xBRZ",
            f"{u}/xBRZ_example_shell_40px_4x.webp"
        ),
        (
            "FSR 1.1",
            f"{u}/FSR_example_shell_40px_4x.webp"
        ),
        (
            "Repetition",
            f"{u}/Repetition_example_shell_40px_4x.webp"
        )
    ]

    for i, entry in enumerate(big_shell_summary_data):
        big_shell_summary_data[i] = (entry[0], f"![{entry[0]}]({entry[1]})")

    batches: list[tuple[tuple[str, str], tuple[str, str], tuple[str, str], tuple[str, str]] | list[tuple[str, str]]] = []

    rest = False
    end = None
    for i in range(0, len(big_shell_summary_data), 4):
        if len(big_shell_summary_data[i:]) < 4:
            rest = True
            end = i
            break
        batches.append((
            big_shell_summary_data[i],
            big_shell_summary_data[i + 1],
            big_shell_summary_data[i + 2],
            big_shell_summary_data[i + 3]
        ))

    if rest:
        rest = big_shell_summary_data[end:]
        batches.append(rest)

    # 100 x `-`
    print("\n----------------------------------------------------------------------------------------------------\n")
    for batch in batches:
        print(generate_markdown_table(batch))
        print()
    print("----------------------------------------------------------------------------------------------------\n")


if __name__ == "__main__":
    # generate_markdown_for_supported_read_formats()
    # generate_markdown_for_example_images()

    p = "../../example_images"

    generate_small_shell_summary(p)
    # generate_big_shell_summary(p)
    ...
