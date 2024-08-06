# coding=utf-8

# WINDOWS --- --- --- --- --- --- --- --- ---
import math
import os
import PIL.Image
import psutil
import string
import shutil
import subprocess

# from scaling.utils import ConfigPlus


# TODO: If the images can't fit in the RAM disk, we should split them into smaller chunks and process them one by one.
def calculate_ram_disk_size(total_img_size, save_margin=0.1, offset=0.1, min_offset=2*1024**3, max_offset=8*1024**3) -> int | None:
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
    # NTFS minimal volume size is 10 MB, it has around 12.5% metadata overhead
    required_size = max(int(total_img_size * (1 + save_margin + 0.125)), 10 * 1024 ** 2)

    # Get the total system memory and the currently available memory
    total_memory = psutil.virtual_memory().total
    available_memory = psutil.virtual_memory().available

    # Calculate the memory to be left free (offset) if offset is a float
    if isinstance(offset, float) and 0 < offset < 1:
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


def is_drive_letter_available(drive_letter):
    # Check if the drive letter is available
    return not os.path.exists(f"{drive_letter}:\\")


def find_available_drive_letter(start_letter='R'):
    # Generate a list of drive letters from the given start letter
    drive_letters = string.ascii_uppercase[string.ascii_uppercase.index(start_letter):]

    for letter in drive_letters:
        if is_drive_letter_available(letter):
            return letter
    raise RuntimeError("No available drive letters found.")


def create_ram_disk_windows(size_mb, drive_letter):
    try:
        # Create the RAM disk using ImDisk
        command = f'imdisk -a -s {size_mb}M -m {drive_letter}: -p "/fs:NTFS /q /y"'
        # subprocess.run(command, shell=True, check=True)
        # elevated_command = f'powershell -Command "Start-Process -Verb runAs cmd -ArgumentList \'/c {command}\'"'
        elevator_command = f'launcher.exe %ComSpec% /c "{command}"'
        subprocess.run(elevator_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to create RAM disk: {e}")
        raise e

# def create_ram_disk_windows(size_mb, drive_letter):
#     try:
#         # PowerShell script to create a RAM disk
#         ps_script = f"""
#         $size = {size_mb}MB
#         $letter = '{drive_letter}'
#         $ramdisk = New-VirtualDisk -Size $size -FriendlyName "RAMDisk" -MediaType "SSD"
#         Initialize-Disk -Number $ramdisk.Number -PartitionStyle MBR
#         New-Partition -DiskNumber $ramdisk.Number -DriveLetter $letter -UseMaximumSize | Format-Volume -FileSystem NTFS -NewFileSystemLabel "RAMDisk" -Force
#         """
#         command = ["powershell", "-Command", ps_script]
#         subprocess.run(command, check=True, shell=True)
#     except subprocess.CalledProcessError as e:
#         print(f"Failed to create RAM disk: {e}")
#     except FileNotFoundError:
#         print("PowerShell not found. Ensure you have PowerShell installed and accessible.")


def remove_ram_disk_windows(drive_letter):
    # Remove the RAM disk using ImDisk
    command = f'imdisk -D -m {drive_letter}:'
    subprocess.run(command, shell=True, check=True)


# def remove_ram_disk_windows(drive_letter):
#     try:
#         # PowerShell script to remove the RAM disk
#         ps_script = f"""
#         $letter = '{drive_letter}:'
#         $partition = Get-Partition -DriveLetter $letter
#         Remove-Partition -PartitionNumber $partition.PartitionNumber -DiskNumber $partition.DiskNumber -Confirm:$false
#         Remove-VirtualDisk -FriendlyName "RAMDisk" -Confirm:$false
#         """
#         command = ["powershell", "-Command", ps_script]
#         subprocess.run(command, check=True, shell=True)
#     except subprocess.CalledProcessError as e:
#         print(f"Failed to remove RAM disk: {e}")


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
        max_needed_space = math.ceil(max_pixel_count * max_channels * bit_depth / 8)
        needed_size = max_needed_space * (max_factor ** 2 * 2 + 1)

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

        print(f"Created RAM disk at {ram_disk_drive_letter}:")


def extinguish_the_drive():
    global is_virtual_drive_on, virtual_drive_letter

    if is_virtual_drive_on:
        print(f"Removing RAM disk at {virtual_drive_letter}:")
        is_virtual_drive_on = False

        # Remove the RAM disk
        remove_ram_disk_windows(virtual_drive_letter)

        virtual_drive_letter = None


def main():
    # Define the RAM disk size
    ram_disk_size_mb = 100

    # Find an available drive letter starting from 'R'
    try:
        ram_disk_drive_letter = find_available_drive_letter('R')
    except RuntimeError as e:
        print(e)
        return

    print(f"Using drive letter: {ram_disk_drive_letter}")

    # Create the RAM disk
    create_ram_disk_windows(ram_disk_size_mb, ram_disk_drive_letter)

    # Paths for source and destination files
    source_file = "C:\\path\\to\\source\\image.jpg"
    destination_file = f"{ram_disk_drive_letter}:\\image.jpg"

    # Move the file to the RAM disk
    shutil.copy(source_file, destination_file)
    print(f"Copied {source_file} to {destination_file}")

    # Verify the file is in the RAM disk
    if os.path.exists(destination_file):
        print(f"File exists in RAM disk: {destination_file}")

    # Define the new destination path to move back the file
    new_destination = "C:\\path\\to\\destination\\image.jpg"
    shutil.copy(destination_file, new_destination)
    print(f"Copied {destination_file} to {new_destination}")

    # Remove the RAM disk
    remove_ram_disk_windows(ram_disk_drive_letter)
    print(f"Removed RAM disk at {ram_disk_drive_letter}:")


if __name__ == "__main__":
    main()


# LINUX --- --- --- --- --- --- --- --- ---

def is_mount_point_available(mount_point):
    # Check if the mount point already exists
    return not os.path.ismount(mount_point)


def create_ram_disk_linux(size_mb, mount_point):
    if not is_mount_point_available(mount_point):
        raise RuntimeError(f"Mount point {mount_point} is already in use.")

    os.makedirs(mount_point, exist_ok=True)
    os.system(f"sudo mount -t tmpfs -o size={size_mb}m tmpfs {mount_point}")


def remove_ram_disk_linux(mount_point):
    os.system(f"sudo umount {mount_point}")
    os.rmdir(mount_point)


# def main():
#     ram_disk_path = "/mnt/ramdisk"
#     ram_disk_size_mb = 100
#
#     # Create the RAM disk
#     create_ram_disk_linux(ram_disk_size_mb, ram_disk_path)
#
#     # Paths for source and destination files
#     source_file = "/path/to/source/image.jpg"
#     destination_file = os.path.join(ram_disk_path, "image.jpg")
#
#     # Move the file to the RAM disk
#     shutil.copy(source_file, destination_file)
#     print(f"Copied {source_file} to {destination_file}")
#
#     # Verify the file is in the RAM disk
#     if os.path.exists(destination_file):
#         print(f"File exists in RAM disk: {destination_file}")
#
#     # Define the new destination path to move back the file
#     new_destination = "/path/to/destination/image.jpg"
#     shutil.copy(destination_file, new_destination)
#     print(f"Copied {destination_file} to {new_destination}")
#
#     # Remove the RAM disk
#     remove_ram_disk_linux(ram_disk_path)
#     print(f"Removed RAM disk at {ram_disk_path}")
