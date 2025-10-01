import os
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm

from scipy import optimize
from scipy import linalg
from scipy import interpolate
from scipy import signal

sys.path.append("/home/lorenzocapelli/repos/wombats")

from wombats.anomalies.increasing import *
from wombats.anomalies.invariant import *
from wombats.anomalies.decreasing import *

from pathlib import Path as Path
from matplotlib import pyplot as plt

def load_and_process_sentinel_data(data_dir, freq='4s', window_size=16, num_channels=16, rw_group='RW1'):
    """
    Loads and processes the Sentinel dataset by segmenting the data into windows,
    optionally selecting a subset of channels, and removing any windows containing NaN values.

    Parameters:
    ----------
    data_dir : str
        Base directory of the Sentinel dataset.
    freq : str, optional
        Temporal frequency of the dataset (default is '4s').
    window_size : int, optional
        Size of the window to use when reshaping the data (default is 16).
    num_channels : int, optional
        Number of channels to extract (must be 16 or 4). Default is 16.
    rw_group : str, optional
        If num_channels is 4, choose between 'RW1', 'RW2', 'RW3', 'RW4'. Default is 'RW1'.

    Returns:
    -------
    np.ndarray
        A numpy array with shape (num_valid_windows, window_size, num_selected_channels), 
        containing only the windows with no NaN values.
    """

    assert num_channels in [4, 16], "num_channels must be 4 or 16"
    if num_channels == 4:
        rw_map = {'RW1': (0, 4), 'RW2': (4, 8), 'RW3': (8, 12), 'RW4': (12, 16)}
        assert rw_group in rw_map, f"Invalid rw_group '{rw_group}'. Must be one of {list(rw_map.keys())}"
        start_ch, end_ch = rw_map[rw_group]

    # Construct the dataset name and path to the pickle file
    dataset_name = f'sentinel_{freq}_clean_std'
    file_path = os.path.join(data_dir, dataset_name, 'train_data.pkl')

    # Load the standardized training DataFrame
    data_train_std = pd.read_pickle(file_path)

    # Select desired channels if needed
    if num_channels == 4:
        data_train_std = data_train_std.iloc[:, start_ch:end_ch]

    # Get the original shape
    rows = data_train_std.shape[0]

    # Trim the number of rows to be an exact multiple of the window size
    trimmed_rows = (rows // window_size) * window_size
    df_trimmed = data_train_std.iloc[:trimmed_rows]

    # Convert to numpy array and reshape into windows
    print(df_trimmed.to_numpy().shape)
    data_grouped = df_trimmed.to_numpy().reshape(-1, window_size, num_channels)

    # Remove windows that contain any NaN values
    mask = ~np.isnan(data_grouped).any(axis=(1, 2))
    data_cleaned = data_grouped[mask]

    # Debug prints
    print(f"Selected channels: {num_channels} ({'all' if num_channels == 16 else rw_group})")
    print(f"Shape after reshaping: {data_grouped.shape}")
    print(f"Shape after removing NaNs: {data_cleaned.shape}")

    return data_cleaned

def apply_anomalies_to_channel(df_cleaned, feature_name, anomaly_name, delta=0.8, return_all=False, column_names=None):
    """
    Applica un'anomalia (o tutte) a un singolo canale (feature) e restituisce un nuovo array con il canale corrotto.
    
    Parameters:
    - df_cleaned (numpy array): il dataset originale, senza NaN. Shape attesa: (samples, time_steps, features)
    - feature_name (str): nome della feature da corrompere (es. 'RW1_motcurr')
    - anomaly_name (str): nome dell'anomalia da applicare (es. 'Constant')
    - delta (float): intensità dell'anomalia
    - return_all (bool): se True, restituisce anche un DataFrame con **tutte** le anomalie applicate
    
    Returns:
    - new_array (numpy array): array con il canale corrotto
    - Xko_df (DataFrame | None): DataFrame multi-indice con tutte le anomalie applicate (solo se return_all=True)
    """
    if column_names is None:
        # Definisci i nomi delle colonne se non sono stati passati
        # Assicurati che questi nomi corrispondano alle colonne del tuo dataset
        # Se il dataset ha colonne diverse, aggiorna questa lista di conseguenza
        column_names = ['RW1_motcurr', 'RW1_therm', 'RW1_cmd_volt', 'RW1_speed', 'RW2_motcurr',
                'RW2_therm', 'RW2_cmd_volt', 'RW2_speed', 'RW3_motcurr', 'RW3_therm',
                'RW3_cmd_volt', 'RW3_speed', 'RW4_motcurr', 'RW4_therm', 'RW4_cmd_volt',
                'RW4_speed']
    if feature_name not in column_names:
        raise ValueError(f"Feature '{feature_name}' non valida. Disponibili: {column_names}")
    # Mappa il nome della feature al suo indice
    feature_idx = column_names.index(feature_name)  # Otteniamo l'indice usando il nome
    
    # Selezioniamo il canale da corrompere
    df_try = df_cleaned[:, :, feature_idx]
    
    # Definiamo tutte le anomalie
    anomalies_dict = {
        'GWN': GWN(delta),
        'Constant': Constant(delta),
        #'Step': Step(delta),
        'saturation'
        'Impulse': Impulse(delta),
        'PrincipalSubspaceAlteration': PrincipalSubspaceAlteration(delta)
    }

    if anomaly_name not in anomalies_dict:
        raise ValueError(f"Anomalia '{anomaly_name}' non valida. Disponibili: {list(anomalies_dict.keys())}")
    
    if return_all:
        Xko_df = pd.DataFrame(
            index=np.arange(df_try.shape[0]),
            columns=pd.MultiIndex.from_product([list(anomalies_dict.keys()), np.arange(df_try.shape[1])])
        )

        for name, anomaly in tqdm(anomalies_dict.items()):
            anomaly.fit(df_try)
            distorted_data = anomaly.distort(df_try)
            Xko_df[name] = distorted_data
        
        selected_anomaly_data = Xko_df[anomaly_name].values
    else:
        anomaly = anomalies_dict[anomaly_name]
        anomaly.fit(df_try)
        selected_anomaly_data = anomaly.distort(df_try)
        Xko_df = None

    # Crea una copia di data_cleaned e sostituisci solo il canale corrotto
    new_array = df_cleaned.copy()
    
    # Sostituiamo il canale selezionato con i dati distorti dell'anomalia selezionata
    new_array[:, :, feature_idx] = selected_anomaly_data

    return new_array, Xko_df

def plot_anomalies(df_try, Xko_df, anomalies_dict, anomaly_name):
    """
    Visualizza i grafici comparativi per ogni tipo di anomalia.

    Parameters:
    - df_try (numpy array): il canale originale
    - Xko_df (DataFrame): dati con anomalie applicate
    - anomalies_dict (dict): dizionario delle anomalie
    """
    
    # Crea i sottoplot
    fig, ax = plt.subplots(len(anomalies_dict), 1, figsize=(8, 10), sharex=True, sharey=False)
    ax = ax.flatten()
    
    # Crea i grafici per ogni anomalia
    for i, (name, ax_) in enumerate(zip(anomalies_dict.keys(), ax)):
        ax_.plot(df_try[0], label='Original data', color='gray', linewidth=1.5, marker='o', markersize=4)
        ax_.plot(Xko_df.loc[0, name], label=f'Anomalous data', color='crimson', linewidth=1.2, linestyle='--', marker='x', markersize=5)
        ax_.set_ylabel("Value", fontsize=9)
        ax_.set_title(f"Anomaly: {name}", fontsize=10, fontweight='bold', pad=6)
        ax_.legend(loc='upper right', fontsize=8)
        ax_.grid(True, linestyle='--', alpha=0.4)
    
    # Etichetta asse X solo sull'ultimo subplot
    ax[-1].set_xlabel("Temporal Index", fontsize=10)
    fig.tight_layout(h_pad=1.5)
    plt.savefig(f"anomalies_comparison_{anomaly_name}.png", dpi=300)
    plt.show()

if __name__ == "__main__":

    data_dir = '/srv/newpenny/dataset/TASI/sentinel'
    data_cleaned = load_and_process_sentinel_data(data_dir=data_dir)     

    deltas = [0.05, 0.1, 0.5]
    idx =0
    delta = deltas[idx]
    anomalies_dict = {
        'GWN': GWN(delta),
        'Constant': Constant(delta),
        'Step': Step(delta),
        'Impulse': Impulse(delta),
        'PrincipalSubspaceAlteration': PrincipalSubspaceAlteration(delta)
    }

    column_names = ['RW1_motcurr', 'RW1_therm', 'RW1_cmd_volt', 'RW1_speed', 'RW2_motcurr',
                'RW2_therm', 'RW2_cmd_volt', 'RW2_speed', 'RW3_motcurr', 'RW3_therm',
                'RW3_cmd_volt', 'RW3_speed', 'RW4_motcurr', 'RW4_therm', 'RW4_cmd_volt',
                'RW4_speed']
    feature_name = 'RW1_motcurr'  # Canale da corrompere
    anomaly_name = 'Constant'  # Anomalia da applicare

    # Applica tutte le anomalie al canale selezionato
    new_data, _ = apply_anomalies_to_channel(data_cleaned, feature_name, anomaly_name)
    

