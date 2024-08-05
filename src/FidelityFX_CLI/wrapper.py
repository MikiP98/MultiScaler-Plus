# coding=utf-8

# WINDOWS --- --- --- --- --- --- --- --- ---
import os
import string
import shutil
import subprocess


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
    # Create the RAM disk using ImDisk
    command = f'imdisk -a -s {size_mb}M -m {drive_letter}: -p "/fs:NTFS /q /y"'
    subprocess.run(command, shell=True)


def remove_ram_disk_windows(drive_letter):
    # Remove the RAM disk using ImDisk
    command = f'imdisk -D -m {drive_letter}:'
    subprocess.run(command, shell=True)


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

