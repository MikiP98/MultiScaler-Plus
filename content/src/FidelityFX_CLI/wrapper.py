# coding=utf-8
import os
import psutil

import math
import string
import subprocess


def calculate_ram_disk_size(
        max_img_size: int,
        save_margin: float = 0.1,
        offset: int | float = 0.1,
        min_offset: int = 2*1024**3,
        max_offset: int = 8*1024**3
) -> int | None:
    """
    Calculate the size of the RAM disk needed and check if there is enough available memory.

    Parameters:
    total_img_size (int): Total size of images in bytes.
    save_margin (float, optional): Margin to account for potential increase in file size (default is 0.1, or 10%).
    offset (int or float, optional): Margin to leave free in system memory,
        in bytes if int, or percentage if float (default is 10%).
    min_offset (int, optional): Minimum offset to leave free in system memory in bytes (default is 2 GB).
    max_offset (int, optional): Maximum offset to leave free in system memory in bytes (default is 8 GB).

    Returns:
    int: Size of the RAM disk needed in bytes if enough memory is available.
    None: If not enough memory is available.
    """
    # Calculate the size needed for the RAM disk
    # NTFS minimal volume size is 10 MB, it has around 12.5% metadata overhead and on smaller volumes it's >1 MB
    required_size = max(int(max_img_size * (1 + save_margin + 0.125) + 1024 ** 2), 10 * 1024 ** 2)

    # Get the total system memory and the currently available memory
    total_memory = psutil.virtual_memory().total
    available_memory = psutil.virtual_memory().available

    # Calculate the memory to be left free (offset) if offset is a float
    if isinstance(offset, float) and 0 < offset < 1024:  # Assume that offset of 102400+% is a mistake
        offset = total_memory * offset

    offset = min(max(min_offset, offset), max_offset)

    # Check if there is enough available memory
    if available_memory > required_size + offset:
        return required_size
    else:
        print("Not enough memory available for the RAM disk.")
        print(f"Required size: {required_size} bytes")
        print(f"Required offset: {offset} bytes")
        print(f"Available memory: {available_memory} bytes")
        return None


# WINDOWS FUNCTIONS --- --- --- --- --- --- --- --- ---

def is_drive_letter_available(drive_letter):
    # Check if the drive letter is available
    return not os.path.exists(f"{drive_letter}:\\")


def find_available_drive_letter(start_letter='R'):
    # Generate a list of drive letters from the given start letter
    drive_letters = string.ascii_uppercase[string.ascii_uppercase.index(start_letter):]  # TODO: Add characters before R

    for letter in drive_letters:
        if is_drive_letter_available(letter):
            return letter
    raise RuntimeError("No available drive letters found... HOW?!")


def create_ram_disk_windows(size_mb, drive_letter):
    try:
        # Create the RAM disk using ImDisk
        command = f'imdisk -a -s {size_mb}M -m {drive_letter}: -p "/fs:NTFS /q /y"'
        # subprocess.run(command, shell=True, check=True)

        elevator_command = f'launcher.exe %ComSpec% /c "{command}"'
        subprocess.run(elevator_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to create RAM disk: {e}")
        raise e


def remove_ram_disk_windows(drive_letter):
    # Remove the RAM disk using ImDisk
    command = f'imdisk -D -m {drive_letter}:'
    subprocess.run(command, shell=True, check=True)


is_virtual_drive_on: bool = False
virtual_drive_letter: str | None = None


def get_virtual_drive_letter():
    return virtual_drive_letter


def ignite_the_drive(max_pixel_count: int, max_factor: float) -> None:
    global is_virtual_drive_on, virtual_drive_letter

    if not is_virtual_drive_on:
        # Define the RAM disk size
        max_channels = 4
        bit_depth = 12
        max_needed_space = max_pixel_count * max_channels * bit_depth / 8
        needed_size = math.ceil(max_needed_space * (max_factor ** 2 * 2 + 1))

        ram_disk_size_bytes = calculate_ram_disk_size(needed_size)
        if ram_disk_size_bytes is None:
            return
        ram_disk_size_mb = math.ceil(ram_disk_size_bytes / 1024**2)
        print(f"RAM disk size: {ram_disk_size_mb} MB")

        # Find an available drive letter starting from 'R'
        try:
            ram_disk_drive_letter = find_available_drive_letter('R')
        except RuntimeError as e:
            print(e)
            return

        print(f"Using drive letter: {ram_disk_drive_letter}")

        # Create the RAM disk
        virtual_drive_letter = ram_disk_drive_letter
        create_ram_disk_windows(ram_disk_size_mb, ram_disk_drive_letter)
        is_virtual_drive_on = True
        print(f"Created RAM disk at `{ram_disk_drive_letter}:`")


def extinguish_the_drive():
    global is_virtual_drive_on, virtual_drive_letter

    if is_virtual_drive_on:
        # Remove the RAM disk
        print(f"Removing RAM disk at {virtual_drive_letter}:")
        is_virtual_drive_on = False
        remove_ram_disk_windows(virtual_drive_letter)
        virtual_drive_letter = None


# TODO: Implement Linux functions
# LINUX --- --- --- --- --- --- --- --- --

# def is_mount_point_available(mount_point):
#     # Check if the mount point already exists
#     return not os.path.ismount(mount_point)


# def create_ram_disk_linux(size_mb, mount_point):
#     if not is_mount_point_available(mount_point):
#         raise RuntimeError(f"Mount point {mount_point} is already in use.")
#
#     os.makedirs(mount_point, exist_ok=True)
#     os.system(f"sudo mount -t tmpfs -o size={size_mb}m tmpfs {mount_point}")


# def remove_ram_disk_linux(mount_point):
#     os.system(f"sudo umount {mount_point}")
#     os.rmdir(mount_point)


# def test_linux():
#     ram_disk_path = "/mnt/ramdisk"
#     ram_disk_size_mb = 100
#
#     # Create the RAM disk
#     create_ram_disk_linux(ram_disk_size_mb, ram_disk_path)
#
#     # Keep the drive for 24s
#     print("Keeping the drive for 24s")
#     time.sleep(24)
#
#     # Remove the RAM disk
#     remove_ram_disk_linux(ram_disk_path)
#     print(f"Removed RAM disk at {ram_disk_path}")
