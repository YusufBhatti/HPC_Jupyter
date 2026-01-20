import os
import numpy as np
from scipy.stats import stats
os.putenv("HDF5_DISABLE_VERSION_CHECK", '1')

os.chdir('/home/ybhatti2/HPC_Jupyter/Python_Scripts/')
from utils import get_bc_ppe_data, normalize
import psutil
from esem import gp_model
from esem.utils import get_random_params
import pandas as pd
import xarray as xr
from my_functions import *
os.chdir('/home/ybhatti2/HPC_Jupyter/Python_Scripts/Analysis/')

import gc
from typing import Optional

print('Part 4')
print('This Part will Emulate the regional means.')


lats=xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/lats.nc').lat
lons=xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/lons.nc').lon

var_name = os.getenv('VARIABLE_NAME')
n_samples = int(os.getenv('NUMBER_OF_SAMPLES'))
n_samples_new = int(os.getenv('NUMBER_OF_SAMPLES_Newly_Generated'))
base_dir = os.getenv('BASE_DIR')
base_dir_regional = base_dir + '/Regional'
os.makedirs(base_dir_regional, exist_ok=True)
os.makedirs(base_dir_regional+'/'+var_name, exist_ok=True)

models = xr.open_dataset(f"/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PD/{var_name}_PPE.nc")
varname = list(models.data_vars.keys())[0]
model = models[varname].mean('month')
model = model.transpose("ensemble", "lat", "lon")
ppe_var = copy.deepcopy(model)

print(f"Part 4 - Emulate the regional means for {var_name}")

if var_name == "ERF":
    kernal=['Matern52'] # ANG , SSA, AI, M2

elif var_name == "ERFaci":
    kernal=['Linear','Matern52'] # for  ERF, AOD_M_1, AOD_M_3
    kernal=['Matern52'] # ANG , SSA, AI, M2

elif var_name == "ERFari":
    kernal=['Linear','Matern52'] # for  ERF, AOD_M_1, AOD_M_3
    kernal=['Matern52'] # ANG , SSA, AI, M2
else:
    raise ValueError("Unknown variable name.")


print(f'loaded variables {var_name}')

ppe_normalized = pd.read_csv('/home/ybhatti2/prjs1474/Datasets/Normalized_PPE_Parameters.csv')
ppe_param = pd.read_csv('/home/ybhatti2/prjs1474/Datasets/PPE_Parameters.csv')
ppe_normalized.set_index(ppe_normalized.columns[0], inplace=True)
ppe_param.set_index(ppe_param.columns[0], inplace=True)

import copy

n_total = len(ppe_param)
n_test = 70  # Number of test samples

# Generate randomized indices for splitting the data
random_indices = np.random.permutation(n_total)
random_indices = np.concatenate((random_indices, [0]))

# Split the randomized indices into test and train sets
test_indices = random_indices[:n_test]
train_indices = random_indices[n_test:]

X_test, X_train = ppe_normalized.iloc[test_indices], ppe_normalized.iloc[train_indices]
Y_test, Y_train = ppe_var.isel(sample=test_indices), ppe_var.isel(sample=train_indices)

print('split train / test')

###################################
### EMULATOR ###
###################################

gp_model = gp_model(X_train, Y_train, kernel=kernal)
gp_model.train()

gp_predictions, _ = gp_model.predict(X_test.values)
y_true = Y_test.data.flatten()
y_pred = gp_predictions.data.flatten()

print('Emulator Training complete')
print(f'Distribution of Actual vs. Predicted Values {kernal} = {np.corrcoef(y_true, y_pred)[1][0]**2:.4f}')


ppe_normalized_new_samples = pd.read_csv(f'{base_dir}/ppe_params_{n_samples_new}.csv')
ppe_normalized_new_samples.set_index(ppe_normalized_new_samples.columns[0], inplace=True)
print('Read new samples ')



emulated, var = gp_model.predict(ppe_normalized_new_samples.values)


print('Saving emulated Variables')
emulated.to_netcdf(f'{base_dir_regional}/{var_name}/emulated_Full_Ensemble_{var_name}_{n_samples_new}.nc')

print('Completed Part 4 ')
