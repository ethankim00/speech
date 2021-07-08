import os


def get_wav_files(path: str):
    """
    get list of all wav files in diretory

    Args:
        path (str): Path to directory

    Returns:
        (List[str]): List of wav filenames
    """
    filenames = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("wav"):
                filenames.append(os.path.join(subdir, file))
    return filenames
