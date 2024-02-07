import os
import scaler

from PIL import Image

if __name__ == '__main__':
    # path = "input/blast_furnace_front.png"
    # image = scaler.scale_image(scaler.Algorithms.xBRZ, Image.open(path), 4)

    # Create input and output directory if they don't exist
    if not os.path.exists("input"):
        os.makedirs("input")
    if not os.path.exists("output"):
        os.makedirs("output")

    add_algorithm_name_to_output_files_names = True
    add_factor_to_output_files_names = True
    sort_by_algorithm = False

    algorithms = {scaler.Algorithms.xBRZ, scaler.Algorithms.RealESRGAN, scaler.Algorithms.NEAREST_NEIGHBOR, scaler.Algorithms.BILINEAR, scaler.Algorithms.BICUBIC, scaler.Algorithms.LANCZOS}
    # Go through all files in input directory, scale them and save them in output directory
    # if in input folder there are some directories all path will be saved in output directory
    for root, dirs, files in os.walk("input"):
        for file in files:
            path = os.path.join(root, file)
            for algorithm in algorithms:
                image = scaler.scale_image(algorithm, Image.open(path), 4)

                new_file_name = file
                if add_algorithm_name_to_output_files_names:
                    new_file_name = f"{algorithm.name}_{new_file_name}"
                if add_factor_to_output_files_names:
                    new_file_name = f"{new_file_name[:-4]}_4x{new_file_name[-4:]}"
                print(new_file_name)

                output_dir = "output"
                if sort_by_algorithm:
                    output_dir += f"/{algorithm.name}"

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                output_path = output_dir + '/' + new_file_name
                print(output_path)
                image.save(output_path)
