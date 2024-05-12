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


if __name__ == "__main__":
    generate_markdown_for_supported_read_formats()