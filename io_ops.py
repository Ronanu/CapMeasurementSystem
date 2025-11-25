from __future__ import annotations
from typing import Dict, Any
from os.path import join, exists
from os import makedirs

from signal_transformations import SignalDataLoader, SignalDataSaver
from log import logger


def load_signal(file_path: str, sampling_interval: float = 0.01):
    """Lädt CSV als SignalData (Projektklasse)."""
    data_loader = SignalDataLoader(file_path=file_path, name='Original_Signal', sampling_interval=sampling_interval)
    return data_loader.signal_data, data_loader.state_signal_data


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


def _extract_header_from_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Baut den CSV-Header aus dem results-Dict zusammen. Erwartet, dass UI die
    Dateiname-Infos (manufacturer, capacitance, typ, methode, klass, dut, version)
    und die berechneten Kenngrößen (U_R, ESR, I_c, I_dc) bereits in results ergänzt hat.
    """
    header = {
        # Analyse-Ergebnisse (bestehend)
        'holding_voltage': results.get('holding_voltage'),
        'unloading_parameter': results.get('post_peak_unloading_fit'),
        'peak_time': results.get('peak_time'),
        'peak_value': results.get('peak_value'),
        'peak_mean': results.get('peak_mean'),
        'plus_minus_toleranz': results.get('threshold'),
        'U3': results.get('U3'),
        'U3_mean': results.get('U3_mean'),

        # Dateiname-Infos (NEU)
        'manufacturer': results.get('manufacturer'),
        'capacitance': results.get('capacitance'),
        'typ': results.get('typ'),
        'methode': results.get('methode'),
        'klass': results.get('klass'),
        'dut': results.get('dut'),
        'version': results.get('version'),

        # Berechnete Kenngrößen (NEU)
        'U_R': results.get('U_R'),
        'ESR': results.get('ESR'),
        'I_c': results.get('I_c'),
        'I_dc': results.get('I_dc'),
    }
    return header


def save_results(after_peak_signal, results: Dict[str, Any], save_dir: str, base_name: str) -> str:
    save_path = join(save_dir, base_name + ".csv")
    header = _extract_header_from_results(results)

    saver = SignalDataSaver(
        signal_data=after_peak_signal,
        filename=save_path,
        header_info=header
    )
    saver.save_to_csv()
    logger.info(f"Gespeichert: {save_path}")
    return save_path
