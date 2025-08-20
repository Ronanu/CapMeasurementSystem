
from __future__ import annotations
from typing import Dict, Any
from os.path import dirname, basename, join, exists
from os import makedirs

from signal_transformations import SignalDataLoader, SignalDataSaver
from log import logger


def load_signal(file_path: str, sampling_interval: float = 0.01):
    """Lädt CSV als SignalData (Projektklasse)."""
    data_loader = SignalDataLoader(file_path=file_path, name='Original_Signal', sampling_interval=sampling_interval)
    return data_loader.signal_data


def ensure_save_dir(base_dir: str, subdir: str = "cut_data") -> str:
    save_dir = join(base_dir, subdir)
    if not exists(save_dir):
        makedirs(save_dir)
    return save_dir


def make_save_name(file_name: str) -> str:
    """Konsistente Umbenennung wie im bisherigen Skript."""
    name_parts = file_name.split('_')[:-1]
    name_parts = [n for n in name_parts if n != 'Testaufbau']
    name_parts.append('cut')
    return '_'.join(name_parts)


def save_results(after_peak_signal, results: Dict[str, Any], save_dir: str, base_name: str) -> str:
    save_path = join(save_dir, base_name + ".csv")

    header = {
        'holding_voltage': results.get('holding_voltage'),
        'unloading_parameter': results.get('post_peak_unloading_fit'),
        'peak_time': results.get('peak_time'),
        'peak_value': results.get('peak_value'),
        'peak_mean': results.get('peak_mean'),
        'plus_minus_toleranz': results.get('threshold'),
        'U3': results.get('U3'),
    }

    saver = SignalDataSaver(
        signal_data=after_peak_signal,
        filename=save_path,
        header_info=header
    )
    saver.save_to_csv()
    logger.info(f"Gespeichert: {save_path}")
    return save_path
