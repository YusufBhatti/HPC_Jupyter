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

print('imported complete for Part 1')


lats=xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/lats.nc').lat
lons=xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/lons.nc').lon

var_name = os.getenv('VARIABLE_NAME')
n_samples = int(os.getenv('NUMBER_OF_SAMPLES'))
base_dir = os.getenv('BASE_DIR')

if var_name == "AOD":
    model = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_POLDER_Interpolated_MODEL.nc').TAU_2D_550nm
    kernal=['Matern52']  # AOD

elif var_name == "AI":
    model = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AI_POLDER_Interpolated_MODEL.nc').__xarray_dataarray_variable__
    kernal=['Bias','Matern52','Polynomial'] # ANG , SSA, AI, M2

elif var_name == "ANG":
    model = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/ANG_POLDER_Interpolated_MODEL.nc').ANG_440nm_670nm
    kernal=['Bias','Matern52','Polynomial'] # ANG , SSA, AI, M2

elif var_name == "SSA":
    AOD = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_POLDER_Interpolated_MODEL.nc').TAU_2D_550nm
    model = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/SSA_POLDER_Interpolated_MODEL.nc').__xarray_dataarray_variable__
    #SSA = SSA.where(AOD[:,-1] > 0.2)
    model = model.where(AOD[:,-1] > 0.1)
    kernal=['Bias','Matern52','Polynomial'] # ANG , SSA, AI, M2

elif var_name == "AAOD":
    model = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AAOD_POLDER_Interpolated_MODEL.nc').AAOD
    kernal=['Matern52']  # AAOD

elif var_name == "AOD_Mode_1":
    model= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_Mode_1_POLDER_Interpolated_MODEL.nc').__xarray_dataarray_variable__
    kernal=['Linear','Matern52'] # for  ERF, AOD_M_1, AOD_M_3

elif var_name == "AOD_Mode_2":
    model= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_Mode_2_POLDER_Interpolated_MODEL.nc').TAU_2D_MODE_CI_550nm
    # model2= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_Mode_3_POLDER_Interpolated_MODEL.nc').TAU_2D_MODE_CS_550nm
    # model = model1 + model2
    kernal=['Bias','Matern52','Polynomial'] # ANG , SSA, AI, M2
elif var_name == "AOD_Mode_3":
    model= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_Mode_3_POLDER_Interpolated_MODEL.nc').TAU_2D_MODE_CS_550nm
    kernal=['Linear','Matern52'] # for  ERF, AOD_M_1, AOD_M_3
elif var_name == "AOD_Mode_Coarse":
    model= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_Mode_Coarse_POLDER_Interpolated_MODEL.nc').__xarray_dataarray_variable__
    kernal=['Bias','Matern52','Polynomial'] # ANG , SSA, AI, M2
elif var_name == "ERF":
    model= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PD/ERF_PPE.nc').__xarray_dataarray_variable__
    kernal=['Linear','Matern52'] # for  ERF, AOD_M_1, AOD_M_3
elif var_name == "ERFaci":
    model= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PD/ERFaci_PPE.nc').__xarray_dataarray_variable__
    kernal=['Linear','Matern52'] # for  ERF, AOD_M_1, AOD_M_3
elif var_name == "ERFari":
    model= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PD/ERFari_PPE.nc').__xarray_dataarray_variable__
    kernal=['Linear','Matern52'] # for  ERF, AOD_M_1, AOD_M_3
elif var_name == "CDNC":
    model= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/CDNC_OCI_Interpolated_MODEL.nc').CDNC_INCL_CT.transpose('time', 'ensemble', 'lat', 'lon')
    kernal=['Matern52']  #  CDNC

else:
    raise ValueError("Unknown variable name.")


print(f'loaded variables {var_name}')


# AAOD = (1 - SSA) * AOD

ppe_normalized = pd.read_csv('/home/ybhatti2/prjs1474/Datasets/Normalized_PPE_Parameters.csv')
ppe_param = pd.read_csv('/home/ybhatti2/prjs1474/Datasets/PPE_Parameters.csv')
ppe_normalized.set_index(ppe_normalized.columns[0], inplace=True)
ppe_param.set_index(ppe_param.columns[0], inplace=True)

import copy


if var_name == 'ERF' or var_name == 'ERFaci' or var_name == 'ERFari':
    ppe_var = model.transpose("ensemble", "month", "lat", "lon")
else:
    ppe_var = copy.deepcopy(model[:,:-1]).groupby('time.month').mean().fillna(0)
    ppe_var = ppe_var.transpose("ensemble", "month", "lat", "lon")

# try:
#     ppe_var = copy.deepcopy(model[:,:-1]).groupby('time.month').mean().fillna(0)
#     ppe_var = ppe_var.transpose("ensemble", "month", "lat", "lon")
# except:
#   #  ppe_var = copy.deepcopy(model)#.fillna(0)
#     ppe_var = model.transpose("ensemble", "month", "lat", "lon")

n_total = len(ppe_param)
n_test = 70  # Number of test samples

# Generate randomized indices for splitting the data
random_indices = np.random.permutation(n_total)
random_indices = np.concatenate((random_indices, [0]))

# Split the randomized indices into test and train sets
test_indices = random_indices[:n_test]
train_indices = random_indices[n_test:]

X_test, X_train = ppe_normalized.iloc[test_indices], ppe_normalized.iloc[train_indices]
Y_test, Y_train = ppe_var.isel(ensemble=test_indices), ppe_var.isel(ensemble=train_indices)

print('split train / test')

###################################
### EMULATOR ###
###################################


gp_model = gp_model(X_train, Y_train, kernel=kernal)
gp_model.train()

print('Emulator Training complete')


#n_samples = int(os.getenv('NUMBER_OF_SAMPLES'))

print(f"number of samples = {n_samples}")


#else:
ppe_normalized_new_samples = pd.read_csv(f'{base_dir}/ppe_params_{n_samples}.csv')
ppe_normalized_new_samples.set_index(ppe_normalized_new_samples.columns[0], inplace=True)
print('Read new samples ')





emulated, var = gp_model.predict(ppe_normalized_new_samples.values)
print('Saving emulated Variables')
emulated.to_netcdf(f'{base_dir}/{var_name}/emulated_{var_name.lower()}_{n_samples}.nc')
var.to_netcdf(f'{base_dir}/{var_name}/emulated_var_{var_name.lower()}_{n_samples}.nc')

if var_name == 'ERF' or var_name == 'ERFaci' or var_name == 'ERFari':
    print(f'areaweighting {var_name}')
    area_emulated = areaweight(emulated,lats).mean('month')
    area_emulated.to_netcdf(f'{base_dir}/{var_name}/emulated_{var_name.lower()}_{n_samples}_areaweighted.nc')
