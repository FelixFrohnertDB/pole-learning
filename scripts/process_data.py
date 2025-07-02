import numpy as np 
import pandas as pd 
import dask.dataframe as dd
from tsfresh import (
    extract_features,  
    select_features
)
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction.settings import from_columns

file_prefix = "dataExt/P"
file_suffix = "_energy.pkl"
num_files = 35

all_x = np.concatenate(
    [np.load(f"{file_prefix}{i:02d}{file_suffix}", allow_pickle=True) for i in range(num_files)],
    axis=0
)

# Load and concatenate all features
file_prefix = "dataExt/P"
file_suffix = "_intensity.pkl"
num_files = 35

all_y = np.concatenate(
    [np.load(f"{file_prefix}{i:02d}{file_suffix}", allow_pickle=True) for i in range(num_files)],
    axis=0
)

label_arr = np.array([np.tile(i,np.load(f"{file_prefix}{0:02d}{file_suffix}", allow_pickle=True).shape[0]) for i in np.arange(num_files)]).flatten()

n_samples = all_y.shape[0]

def extract_feature_list_from_subset(features_intensity, features_energies):

    data_df = pd.DataFrame()
    labels_df = pd.Series()

    subsample_inx = 50
    subset_intensities = features_intensity[::subsample_inx]
    subset_energies = features_energies[::subsample_inx]

    data_df['id'] = np.repeat(np.arange(subset_intensities.shape[0]),subset_intensities.shape[1])
    data_df['time'] = np.tile(np.arange(subset_intensities.shape[1]), subset_intensities.shape[0])
    data_df['intensity'] = subset_intensities.flatten()
    data_df['energy'] = subset_energies.flatten()
    data_df.to_parquet('dataExt/subsample_data_df.parquet')

    labels_df['labels'] = label_arr[::subsample_inx]

    # Load data once
    dask_df = dd.read_parquet('dataExt/subsample_data_df.parquet', npartitions=10)

    # Extract features
    extracted_features = extract_features(dask_df, column_id="id", column_sort="time",n_jobs=-1, disable_progressbar=False).compute()

    # Handle any NaNs
    impute(extracted_features)

    # Convert labels to a Series (ensure it’s 1D if needed)
    labels_series =  pd.Series(np.array(labels_df.to_numpy()[0]))

    # Select relevant features
    features_filtered = select_features(extracted_features, labels_series, multiclass=True, fdr_level=0.05)

    kind_to_fc_parameters = from_columns(features_filtered)

    
    return kind_to_fc_parameters, features_filtered.shape[1], features_filtered

kind_to_fc_parameters, n_features, features_filtered = extract_feature_list_from_subset(all_y, all_x)

def extract_features_from_full(features_intensity, features_energies, kind_to_fc_parameters, n_samples, n_features):
    
        
    data_df = pd.DataFrame()
    labels_df = pd.Series()

    subset_intensities = features_intensity
    subset_energies = features_energies

    data_df['id'] = np.repeat(np.arange(subset_intensities.shape[0]),subset_intensities.shape[1])
    data_df['time'] = np.tile(np.arange(subset_intensities.shape[1]), subset_intensities.shape[0])
    data_df['intensity'] = subset_intensities.flatten()
    data_df['energy'] = subset_energies.flatten()
    data_df.to_parquet('dataExt/data_df.parquet')

    labels_df['labels'] = label_arr


    # Load data once
    dask_df = dd.read_parquet('dataExt/data_df.parquet', npartitions=15)

    X = extract_features(dask_df,
                        column_id="id",
                        column_sort="time",
                        pivot=False,
                        kind_to_fc_parameters=kind_to_fc_parameters,
                        disable_progressbar=False
                        )

    result = X.compute()

    np.save("dataExt/features_filtered.npy", np.reshape(result["value"].to_numpy(), (n_samples, n_features)))
    np.save("dataExt/feature_names.npy", features_filtered.columns.to_numpy())

extract_features_from_full(all_y, all_x, kind_to_fc_parameters, n_samples, n_features)