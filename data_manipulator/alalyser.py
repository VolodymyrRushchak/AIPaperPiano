import os
from typing import Tuple

import numpy as np


def pressed_not_ratio(data_path) -> float:
    files_names = os.listdir(data_path)
    num_pressed_keys = list(map(lambda name: len(name.split('#')[0].split('_')), files_names))
    return (len(num_pressed_keys) * 51 - sum(num_pressed_keys)) / sum(num_pressed_keys)


def get_key_array(data_path) -> np.ndarray:
    files_names = np.array(os.listdir(data_path))
    key_pressed = np.array(list(map(lambda name: list(map(int, name.split('#')[0].split('_'))), files_names)), dtype=object)
    return np.concatenate(key_pressed)


def get_rare_key_relative_frequency(frequencies: np.ndarray) -> Tuple[Tuple[int], np.ndarray]:
    max_fr = np.max(frequencies)
    rare_keys = np.array([[key, fr] for key, fr in enumerate(frequencies) if fr != max_fr])
    keys, freqs = list(zip(*rare_keys))
    relative_freqs = np.array(freqs) / sum(freqs)
    return keys, relative_freqs


def get_align_dist(frequencies: np.ndarray, threshold: int) -> np.ndarray:
    dif = threshold - frequencies
    dif[dif < 0] = 0
    return dif / np.sum(dif)
