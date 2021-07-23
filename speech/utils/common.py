import os
from typing import List

import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import torchaudio


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


def build_dataframe_for_classification(td_path: str, ssd_path: str):
    """
    Build DataFrame for Speaker Classification Problem
    """
    ssd_filenames = get_wav_files(ssd_path)
    td_filenames = get_wav_files(td_path)
    df = pd.DataFrame()
    df["file_path"] = ssd_filenames + td_filenames
    df["labels"] = ["disordered" for _ in range(len(ssd_filenames))] + [
        "typical" for _ in range(len(td_filenames))
    ]
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out broken file links from dataframe

    Args:
        df (pd.DataFrame): Input dataframe
    """
    df["status"] = df["file_path"].apply(
        lambda path: True if os.path.exists(path) else None
    )
    df = df.dropna(subset=["file_path"])
    df = df.drop("status", 1)
    df = df.sample(frac=1)
    df = df.reset_index(drop=True)
    return df


def speech_file_to_array(path: str, target_sampling_rate: int = 16000) -> np.array:
    """
    Convert audio file to numpy array representation

    Args:
        path (str): Path to file
        target_sampling_rate (int, optional): [description]. Defaults to 16000.

    Returns:
        np.array: Speech vector
    """
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech


def label_to_id(label, label_list):

    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label
