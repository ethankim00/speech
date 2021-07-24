import os
from typing import List

import librosa
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


def clean_dataframe(df: pd.DataFrame, min_duration: float = 0.5) -> pd.DataFrame:
    """
    Perfrom preprocessing steps on audio dataframe
    1. Remove broken file paths
    2. Shuffle data
    3.

    Args:
        df (pd.DataFrame): Input dataframe
        min_duration (float): Minimum duration of audio to keep in seconds. Defaults to 0.5.

    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    df = filter_broken_links(df)
    df = filter_duration(df, min_duration=min_duration)
    df = shuffle_df(df)
    return df


def shuffle_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shuffle a dataframe

    Args:
        df (pd.DataFrame): Input Dataframe

    Returns:
        pd.DataFrame: Output Dataframe
    """
    df = df.sample(frac=1)
    df = df.reset_index(drop=True)
    return df


def filter_broken_links(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove broken file paths

    Args:
        df (pd.DataFrame): Input df

    Returns:
        pd.DataFrame: Output df
    """
    df["status"] = df["file_path"].apply(
        lambda path: True if os.path.exists(path) else None
    )
    df = df.dropna(subset=["file_path"])
    df = df.drop("status", 1)
    # df = df.reset_index(drop=True)
    return df


def filter_duration(
    df: pd.DataFrame, min_duration: float = 0.5, sample_rate: int = 16000
) -> pd.DataFrame:
    """
    Remove audio samples under a certain duration

    Args:
        df (pd.DataFrame): Input dataframe
        min_duration (float): Minimum duration of audio to keep in seconds. Defaults to 0.5.

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    df["duration"] = df["file_path"].apply(
        lambda path: len(librosa.load(path, sr=sample_rate)[0]) / sample_rate
    )
    print(df["duration"])
    df = df.loc[df["duration"] > min_duration]
    print(len(df))
    # df = df.reset_index(drop=True)
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
    return speech.astype(float)


def label_to_id(label, label_list):

    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label
