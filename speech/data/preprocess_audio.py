import argparse
import os

import numpy as np
import pathlib

import librosa
import soundfile as sf


from typing import List
from speech.utils.common import get_wav_files


def get_domain(filepath: str):
    """Get domain from filepath (ssd or td)

    Args:
        filepath (str): Path to wav file

    Returns:
        (str): Domain
    """
    return filepath.split("/")[-3]


def get_filename(filepath: str):
    """Get base filename from filepath

    Args:
        filepath (str): full filepath

    Returns:
        (str): Base filename
    """
    path = pathlib.Path(filepath)
    return path.stem


def get_speaker_labels(domain: str, filename: str):
    """Get speaker timing labels
    Looks up speaker labels matching fileame from speaker label directory

    Args:
        domain (str): ssd or td
        filename (str): base filename

    Returns:
        (List[List[str]]): List of speaker labels
    """
    labels = []
    speaker_path = os.path.join(
        "./speech/data/data/", domain, "speaker", filename + ".lab"
    )  # TODO set correct path
    print(speaker_path)
    # read .lab file
    with open(speaker_path, "r") as infile:
        for line in infile:
            info = line.split(" ")
            info[0] = int(info[0]) / 10 ** 7  # convert to seconds
            info[1] = int(info[1]) / 10 ** 7
            labels.append((info))
    return labels


def remove_instructor_audio(audio: np.array, speaker_labels: List):
    """Remove instructor audio from wav files

    Args:
        audio (np.array): Raw audio signal
        speaker_labels (List): Speaker timing labels

    Returns:
        (np.array): Subsetted audio signal
    """
    new_audio = []
    for section in speaker_labels:
        if "CHILD" in section[2]:
            child_audio = list(audio[int(section[0] * 16000) : int(section[1] * 16000)])
            new_audio += child_audio
    return np.array(new_audio)


def preprocess_audio(filepath: str) -> np.array:
    """Remove instructor audio based on a filepath

    Args:
        filepath (str): Path to file

    Returns:
        np.array: Subsetted audio signal
    """
    domain = get_domain(filepath)
    filename = get_filename(filepath)
    labels = get_speaker_labels(domain, filename)
    audio, _ = librosa.load(filepath, sr=16000)
    audio = remove_instructor_audio(audio, labels)
    return audio




def main(path: str):
    """Convert all wav files in directory.
    Loads wav file, cuts out instructor audio and then overwrites wav file

    Args:
        path (str): path to directory
    """
    print("Removing Instructor Audio")
    min_duration = 1.0
    max_duration = 20.0
    sample_rate = 16000


    wav_files = get_wav_files(path)
    for wav_file in wav_files:
        print("Removing Instructor audio:", wav_file)
        try:
            audio = preprocess_audio(wav_file)
            os.remove(wav_file)
            if len(audio) > max_duration * sample_rate or len(audio) < min_duration * sample_rate:
                print("Duration out of range")
                continue
            sf.write(wav_file, audio, sample_rate)
        except:
            print("File Not Found")
        #pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    main(args.path)
