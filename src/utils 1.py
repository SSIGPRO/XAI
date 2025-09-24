
import pickle
import numpy as np
import pandas as pd
import os
import sys
import tqdm
from scipy import optimize
from scipy import linalg
from scipy import interpolate
from scipy import signal
from _base import *
from decreasing import *
from increasing import *
from invariant import *
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import math
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import json

data_dir = '/srv/newpenny/dataset/TASI/sentinel'
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
    data_grouped = df_trimmed.to_numpy().reshape(-1, window_size, num_channels)

    # Remove windows that contain any NaN values
    mask = ~np.isnan(data_grouped).any(axis=(1, 2))
    data_cleaned = data_grouped[mask]

    # Debug prints
    print(f"Selected channels: {num_channels} ({'all' if num_channels == 16 else rw_group})")
    print(f"Shape after reshaping: {data_grouped.shape}")
    print(f"Shape after removing NaNs: {data_cleaned.shape}")

    return data_cleaned

# Funzione per applicare anomalie e creare un array con il canale corrotto
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
        'Step': Step(delta),
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

# Funzione per visualizzare i grafici delle anomalie
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


# # Esempio di utilizzo
# feature_name = 'RW3_motcurr'  # Nome della feature da corrompere
# anomaly_name = 'Impulse'     # Nome dell'anomalia da applicare

# # Applica tutte le anomalie al canale selezionato
# new_data, Xko_df = apply_anomalies_to_channel(data_cleaned, feature_name, anomaly_name)

# # Visualizza il grafico delle anomalie
# plot_anomalies(data_cleaned[:, :, column_names.index(feature_name)], Xko_df, anomalies_dict, anomaly_name)

# # Restituisce l'array con il canale corrotto e Xko_df
# print("Nuovo array con anomalie applicate:", new_data.shape)

def plot_all_channels_comparison(original_data, corrupted_data, feature_names, corrupted_feature, anomaly_name, window_idx=0):
    """
    Confronta visivamente tutti i canali (feature) per una finestra specifica,
    mostrando la differenza tra i dati originali e quelli corrotti.

    Parameters:
    - original_data: array 3D originale (N_finestra, seq_len, features)
    - corrupted_data: array 3D con un canale corrotto
    - feature_names: lista dei nomi delle feature (len = numero canali)
    - corrupted_feature: nome del canale corrotto
    - anomaly_name: nome dell'anomalia applicata
    - window_idx: indice della finestra da visualizzare
    """
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 8})
    
    num_features = len(feature_names)
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 10), sharex=True)
    axs = axs.flatten()

    corrupted_idx = feature_names.index(corrupted_feature)

    for i, ax in enumerate(axs):
        ax.plot(original_data[window_idx, :, i], label="Original", color='gray', linewidth=1.5)
        ax.plot(corrupted_data[window_idx, :, i], label="Corrupted", color='crimson', linestyle='--', linewidth=1.2)

        ax.set_title(f"Channel: {i}", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)

        if i == corrupted_idx:
            ax.set_title(f" channel:{i} (Corrupted - {anomaly_name})", color='crimson', fontweight='bold', fontsize=9)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', ncol=2, fontsize=9)
    plt.suptitle(f"Original vs Corrupted signals \n (Anomaly: {anomaly_name}, Channel: {0})", fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# ESEMPIO DI UTILIZZO
# plot_all_channels_comparison(
#     original_data=data_cleaned,
#     corrupted_data=new_data,
#     feature_names=column_names,
#     corrupted_feature=feature_name,
#     anomaly_name=anomaly_name
# )


def evaluate_losses(pred_dict, data_dict):
    losses = {}
    for feature in pred_dict.keys():
        pred = torch.tensor(pred_dict[feature])
        real = torch.tensor(data_dict[feature])
        loss = F.mse_loss(pred, real)
        losses[feature] = loss.item()
    return losses


class ModelConfig:
    def __init__(self):
        self.architecture = "conv_ae1D"  # Tipo di modello
        self.kernel_size = 5  # Dimensione kernel convoluzione
        self.filter_num = 42  # Numero di filtri nelle convoluzioni
        self.stride = 2  # Passo della convoluzione
        self.pool = 0  # Fattore di pooling
        self.latent_dim = 100  # Dimensione dello spazio latente
        self.lay3 = True  # Numero di layer
        self.activation = nn.ELU(alpha=1)  # Funzione di attivazione
        self.bn = True  # Batch Normalization
        #self.increasing = 0  # Flag per determinare la crescita dei filtri
        #self.flattened = 0  # Indica se il modello è appiattito
        self.dilation = 1  # Dilation rate per convoluzioni
        self.padding = 4
        self.in_channel = 16
        

class DatasetConfig:
    def __init__(self):
        self.n_features = 16  # Numero di feature nel dataset
        self.sequence_length = 16  # Lunghezza della sequenza temporale (seq_in_length)
        self.batch_size = 500  # Dimensione del batch
        self.columns = ['RW1_motcurr', 'RW1_therm', 'RW1_cmd_volt', 'RW1_speed', 'RW2_motcurr',
       'RW2_therm', 'RW2_cmd_volt', 'RW2_speed', 'RW3_motcurr', 'RW3_therm',
       'RW3_cmd_volt', 'RW3_speed', 'RW4_motcurr', 'RW4_therm', 'RW4_cmd_volt',
       'RW4_speed']  # Lista delle colonne usate
        self.columns_subset = 16  # Numero di colonne da usare
        self.dataset_subset = None  # Numero di righe da usare
        self.shuffle = True  # Indica se mescolare i dati
        self.train_val_split = 0.8  # Percentuale di dati per il training

class OptimizerConfig:
    def __init__(self):
        self.lr = 0.003  # Learning rate
        self.lr_patience = 5  # Pazienza del learning rate scheduler
        self.epochs = 200  # Numero di epoche di training
        self.es_patience = 10  # Early stopping patience

class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.dataset = DatasetConfig()
        self.opt = OptimizerConfig()

# Creazione dell'istanza di configurazione
cfg = Config()

compression_map = {
        512 : 'large',
        256 : 'medium',
        128 : 'small',
        64 : 'xsmall',
        32 : 'xxsmall',
    }

size_map = {
    'large' : 512,
    'medium' : 256,
    'small' : 128,
    'xsmall' : 64,
    'xxsmall' : 32,
}


def select_case(model_name,model_info,device):
    """
    Seleziona il caso del modello da caricare.
    """
    compression_map = {
        1024: 'xlarge',
        512 : 'large',
        256 : 'medium',
        128 : 'small',
        64 : 'xsmall',
        32 : 'xxsmall',
    }
    
    reg = False
    if model_name.startswith('rwn'):

        import my_model
        # Istanza del modello my_model

        CHECKPOINT_PATH = f'/home/francescoaldrigo/SPACE/francescoaldrigo/model1/{model_name}.pth'
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        param = checkpoint['param_conf']
        emb_size = compression_map[model_info[model_name]['Latent Dim']]
        
        model = my_model.CONV_AE1D(
            in_channel=param.model.in_channel,
            length= model_info[model_name]['seql'],
            kernel_size=model_info[model_name]['kernel_size'],
            embedding_size=emb_size,
            lay3=model_info[model_name]['lay3'],
        ).to(device).float()
        

    elif model_name.startswith('ae1Dreg'):

        import my_model_reg
        # Istanza del modello my_model_reg

        CHECKPOINT_PATH = f'/home/francescoaldrigo/SPACE/francescoaldrigo/train_out_reg/models/{model_name}.pth' 
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        param = checkpoint['param_conf']
        emb_size = compression_map[model_info[model_name]['Latent Dim']]

        model = my_model_reg.CONV_AE1D(
            in_channel=param.model.in_channel,
            length= model_info[model_name]['seql'],
            kernel_size=model_info[model_name]['kernel_size'],
            embedding_size=emb_size,
            lay3=model_info[model_name]['lay3'],
        ).to(device).float()
        reg = True
    elif model_name.startswith('miao'):

        from model_generator import CONV_AE1D
        # Istanza del modello Leonardo

        CHECKPOINT_PATH = f'/home/francescoaldrigo/SPACE/francescoaldrigo/train_out/{model_name}.pth'
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        param = checkpoint['param_conf']

        model = CONV_AE1D(
            in_channel=param.model.in_channel,
            length=param.dataset.sequence_length,
            kernel_size=param.model.kernel_size, 
            filter_num=param.model.filter_num,
            latent_dim = param.model.latent_dim, 
            activation=param.model.activation, 
            stride=param.model.stride, 
            pool=param.model.pool,
        ).to(device).float()

      
    else:
        raise ValueError(f"Modello non supportato: {model_name}")
    # Carica lo stato del modello
    model.load_state_dict(checkpoint['model_state_dict'])
    return model,reg

def evaluate_model(model, device, dataset_clean, dataset_corrupted, loss_fn,num_samples = 1600, reg = False):
    """
    Valuta il modello su un batch di finestre randomiche.

    Parameters:
    - model: modello PyTorch
    - device: dispositivo su cui eseguire il modello (es. 'cuda' o 'cpu')
    - dataset_clean: Dataset senza anomalie (ConvDataset)
    - dataset_corrupted: Dataset con anomalie
    - loss_fn: funzione di loss (es. MSELoss)
    - num_samples: numero di finestre da campionare
    - reg: se True, calcola anche la regolarizzazione L2 sui vettori latenti

    Returns:
    - loss_clean: lista dei loss su finestre pulite
    - loss_corr: lista dei loss su finestre corrotte
    - l2_clean: lista della regolarizzazione L2 su finestre pulite (o None)
    - l2_corr: lista della regolarizzazione L2 su finestre corrotte (o None)
    """

    loss_clean = []
    loss_corr = []
    indices = np.random.choice(len(dataset_clean), size=num_samples, replace=False)
    clean_batch = torch.stack([
        torch.from_numpy(dataset_clean[i][0]) for i in indices
    ]).permute(0, 2, 1).to(device).float()

    corr_batch = torch.stack([
        torch.from_numpy(dataset_corrupted[i][0]) for i in indices
    ]).permute(0, 2, 1).to(device).float()
    
    with torch.no_grad():
        if reg:
            # Per il modello regularized
            out_clean, lat_v_clean = model(clean_batch)
            out_corr, lat_v_corr= model(corr_batch)
            loss_clean = loss_fn(out_clean, clean_batch).mean(dim=[1, 2]).cpu().numpy()
            loss_corr = loss_fn(out_corr, corr_batch).mean(dim=[1, 2]).cpu().numpy()
            l2_clean = torch.linalg.vector_norm(lat_v_clean, dim=1).pow(2).cpu().numpy()
            l2_corr = torch.linalg.vector_norm(lat_v_corr, dim=1).pow(2).cpu().numpy()
            return loss_clean.tolist(), loss_corr.tolist(), l2_clean.tolist(), l2_corr.tolist()

        else:
            # Per il modello classico
            out_clean = model(clean_batch)
            out_corr = model(corr_batch)
        

        # Loss per finestra, non aggregato
        loss_clean = loss_fn(out_clean, clean_batch).mean(dim=[1, 2]).cpu().numpy()
        loss_corr = loss_fn(out_corr, corr_batch).mean(dim=[1, 2]).cpu().numpy()
    return loss_clean.tolist(), loss_corr.tolist(), None , None