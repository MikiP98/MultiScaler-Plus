# coding=utf-8
import PIL.Image


def scale(frames: list[PIL.Image], factor: float, algorithm: str) -> list[PIL.Image]:
    # import docker
    # client = docker.from_env()
    # if algorithm == Algorithms.Waifu2x:
    #     # Define the image name
    #     image_name = 'waifu2x-python:3.11'
    #
    #     # Check if the image exists
    #     try:
    #         image = client.images.get(image_name)
    #         print("Image exists")
    #     except docker.errors.ImageNotFound:
    #         tar_file_path = 'docker/images/waifu2x.tar'
    #         try:
    #             with open(tar_file_path, 'rb') as file:
    #                 client.images.load(file.read())
    #         except FileNotFoundError:
    #             print("Image does not exist. Building it...")
    #             # Build the image
    #             client.images.build(path='docker/files/waifu2x', tag=image_name)
    #
    #         image = client.images.get(image_name)
    #         print("Image exists")
    #     except docker.errors.APIError as e:
    #         print(f"An error occurred: {e}")

    # image_name = "your-image-name"
    # dockerfile_location = "./docker/files"
    #
    # command = f"docker build -t {image_name} {dockerfile_location}"
    # process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    # output, error = process.communicate()
    #
    # if error:
    #     print(f"An error occurred: {error}")
    # else:
    #     print(f"Output: {output.decode('utf-8')}")
    #
    # container_name = "your-container-name"
    #
    # command = f"docker create --name {container_name} {image_name}"
    # process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    # output, error = process.communicate()
    #
    # if error:
    #     print(f"An error occurred: {error}")
    # else:
    #     print(f"Output: {output.decode('utf-8')}")
    #
    # command = f"docker start {container_name}"
    # process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    # output, error = process.communicate()
    #
    # if error:
    #     print(f"An error occurred: {error}")
    # else:
    #     print(f"Output: {output.decode('utf-8')}")

    raise NotImplementedError("Waifu2x and SUPIR are not implemented yet!")

