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
        "Nearest Neighbour", "Bilinear", "Bicubic", "Lanczos",
        "EDSR", "ESPCN", "FSRCNN", "FSRCNN-small", "LapSRN", "RealESRGAN",
        "hqx", "NEDI <sup>*(m = 4)*</sup>", "Super xBR", "xBRZ",
        "FSR", "CAS <sup>*(sharpness = 0.5)*</sup>"
    ]
    links = [
        "https://upload.wikimedia.org/wikipedia/commons/a/a6/160_by_160_thumbnail_of_%27Green_Sea_Shell%27.png",

        "./src/example_images/output/CV2_INTER_NEAREST_example_shell_40px_4x.png",
        "./src/example_images/output/CV2_INTER_LINEAR_example_shell_40px_4x.png",
        "./src/example_images/output/CV2_INTER_CUBIC_example_shell_40px_4x.png",
        "./src/example_images/output/CV2_INTER_LANCZOS4_example_shell_40px_4x.png",

        "./src/example_images/output/CV2_EDSR_example_shell_40px_4x.png",
        "./src/example_images/output/CV2_ESPCN_example_shell_40px_4x.png",
        "./src/example_images/output/CV2_FSRCNN_example_shell_40px_4x.png",
        "./src/example_images/output/CV2_FSRCNN_small_example_shell_40px_4x.png",
        "./src/example_images/output/CV2_LapSRN_example_shell_40px_4x.png",
        "./src/example_images/output/RealESRGAN_example_shell_40px_4x.png",

        "./src/example_images/output/hqx_example_shell_40px_4x.png",
        "./src/example_images/output/NEDI_example_shell_40px_4x.png",
        "./src/example_images/output/Super_xBR_example_shell_40px_4x.png",
        "./src/example_images/output/xBRZ_example_shell_40px_4x.png",

        "./src/example_images/output/example_shell_40px_FSR.png",
        "./src/example_images/output/example_shell_40px_CAS.png",
    ]

    columns = 4
    print("Example images:")
    print("```")
    print(f"| {' | '.join(algorithms[:columns])} |")
    print(f"| {' | '.join([':---:'] * columns)} |")
    print(f"| {' | '.join([f'![{algorithm}]({link})' for algorithm, link in zip(algorithms[:columns], links[:columns])])} |")
    for i in range(columns, len(algorithms), columns):
        if not split:
            print(f"| {' | '.join(algorithms[i:i + columns])} |")
            print(f"| {' | '.join([f'![{algorithm}]({link})' for algorithm, link in zip(algorithms[i:i + columns], links[i:i + columns])])} |")
        else:
            if columns > len(algorithms) - i:
                columns = len(algorithms) - i
            print()
            print(f"| {' | '.join(algorithms[i:i + columns])} |")
            print(f"| {' | '.join([':---:'] * columns)} |")
            print(f"| {' | '.join([f'![{algorithm}]({link})' for algorithm, link in zip(algorithms[i:i + columns], links[i:i + columns])])} |")


if __name__ == "__main__":
    # generate_markdown_for_supported_read_formats()
    generate_markdown_for_example_images()
    ...
