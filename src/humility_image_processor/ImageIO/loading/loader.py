# coding=utf-8
import os
import shutil


class Loader:
    # This is not the right place for this method, but it is the most convinient place for it
    # Clear should run before any saving and it would be best if it run before the start of any processing,
    # so around the time of the first image load
    @staticmethod
    def clear_output_directory(clear_output_dir: bool):
        if clear_output_dir:
            print(f"{b}Clearing the output directory{nr}")
            shutil.rmtree("../../output", ignore_errors=True)
            os.makedirs("../../output", exist_ok=True)  # exist_ok=True is here for multithreading safety


class ImageLoader(Loader):
    def __init__(self, input_paths: list[str], dropin_paths: list[str], config: LoaderConfig):
        self.clear_output_directory(config.clear_output_dir)

        working_dir = os.getcwd()

        relative_paths = []
        for path in input_paths + dropin_paths:
            relative_paths.append(os.path.relpath(path, working_dir))

        self.paths = iter(self.relative_paths)


    def get_next_image(self):
        return load_image(self.paths.next())


    def load_images(self):
        images = []
        for path in self.paths:
            images.append(load_image(path))
        return images


    @staticmethod
    def load_image(path):
        image = None
        return image


class TextureLoader(Loader):
    def __init__(self, input_paths, dropin_paths):
        pass


    def get_next_texture_set(self):
        return load_image(self.paths.next())


    def load_texture_sets(self):
        images = []
        return images


    @staticmethod
    def load_texture_set():
        image = None
        return image


class LoaderConfig(TypedDict):
    clear_output_dir: bool = True

    put_drag_and_drop_in_output: bool = False
    put_input_folder_in_output: bool = True
    put_other_inputs_in_output: bool = True

    copy_over: list[str] = ["mcmeta"]  # e.g. `.mcmeta` to not break processed MC textures

    prefix_filter: Optional[str]  # endswith
    suffix_filter: Optional[str]  # startswith
    name_part_filter: Optional[str]  # in
    name_filter: Optional[str]  # exact match
    extension_filter: Optional[str]  # exact match

    prefix_blacklist: Optional[str]  # not endswith
    suffix_blacklist: Optional[str]  # not startswith
    name_part_blacklist: Optional[str]  # not in
    name_blacklist: Optional[str]  # not exact match
    extension_blacklist: Optional[str]  # not exact match