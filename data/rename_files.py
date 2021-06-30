import argparse
import os
import subprocess


def rename_files(path:str, domain:str):
    """
    Rename the downloaded files prepending the last two subdirectories to form a unique identifier
    """
    if domain == "ssd":
        name_dir_num = 3
    elif domain == "td":
        name_dir_num = 2
    for subdir, dirs, files in os.walk(path):
        for file in files:
            filename = os.path.join(subdir, file)
            new_filename = "-".join(filename.split("/")[-name_dir_num:])
            process = subprocess.run(['cp', filename, os.path.join(path, new_filename)], stdout=subprocess.PIPE, universal_newlines=True)
            os.remove(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("domain")
    args = parser.parse_args()
    rename_files(args.path, args.domain)