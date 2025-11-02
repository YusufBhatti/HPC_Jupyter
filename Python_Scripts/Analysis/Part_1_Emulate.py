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
print('imported complete')


lats=xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/lats.nc').lat
lons=xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/lons.nc').lon
lats_so = lats.sel(lat=slice(-60, -40))


AOD = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_POLDER_Interpolated_MODEL.nc').TAU_2D_550nm
AOD_SO = AOD.sel(lat=slice(-60, -30))

SSA = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/SSA_POLDER_Interpolated_MODEL.nc').__xarray_dataarray_variable__
SSA_SO = SSA.sel(lat=slice(-60, -30))
SSA = SSA.where(AOD[:,-1] > 0.2)

ANG = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/ANG_POLDER_Interpolated_MODEL.nc').ANG_440nm_670nm
ANG_SO = ANG

CN_Burden = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/CN_BURDEN_POLDER_Interpolated_MODEL.nc').CN_BURDEN

AAOD = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AAOD_POLDER_Interpolated_MODEL.nc').AAOD

AOD_m_1= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_Mode_1_POLDER_Interpolated_MODEL.nc').__xarray_dataarray_variable__

AOD_m_2= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_Mode_2_POLDER_Interpolated_MODEL.nc').TAU_2D_MODE_CI_550nm

AOD_m_3= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_Mode_3_POLDER_Interpolated_MODEL.nc').TAU_2D_MODE_CS_550nm

# Read in table of parameters and their perturbations
#AOD_ppe = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PD/TAU_2D_550nm_hifreq_PPE.nc').TAU_2D_550nm.groupby('time.month').mean()
#ANG_ppe = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PD/ANG_440nm_670nm_hifreq_PPE.nc').ANG_440nm_670nm.groupby('time.month').mean()
#SSA_ppe = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PD/ANG_440nm_670nm_hifreq_PPE.nc').__xarray_dataarray_variable__.groupby('time.month').mean()

land_mask = xr.open_dataset('/home/ybhatti2/prjs1474/Pace_PPE_Output/PPE_Experiments/PPE_ENS_1/PPE_ENS_1_202408.01_echam.nc').slm[0]

print('loaded variables')


# AAOD = (1 - SSA) * AOD

ppe_normalized = pd.read_csv('/home/ybhatti2/prjs1474/Datasets/Normalized_PPE_Parameters.csv')
ppe_param = pd.read_csv('/home/ybhatti2/prjs1474/Datasets/PPE_Parameters.csv')
ppe_normalized.set_index(ppe_normalized.columns[0], inplace=True)
ppe_param.set_index(ppe_param.columns[0], inplace=True)

import copy

#aod_monthly=AOD.groupby('time.month').mean()[:,:-1]
# aod_monthly=AOD_SO.mean('time')[:-1]
# ppe_var = copy.deepcopy(aod_monthly).fillna(0)
ppe_var_aaod = copy.deepcopy(AAOD[:,:-1]).groupby('time.month').mean().fillna(0)
ppe_var_aaod = ppe_var_aaod.transpose("ensemble", "month", "lat", "lon")
# ppe_var_aaod = xr.DataArray(
#     ppe_aaod.data,  # Use .data to extract the values
#     dims=["ensemble","month", "lat", "lon"],  # Set the dimension names to match RF_PI
#     coords={"ensemble": ppe_aaod.coords["ensemble"],
#             "month": ppe_aaod.coords["month"],
#             "lat": ppe_aaod.coords["lat"],
#             "lon": ppe_aaod.coords["lon"]},  # Set coordinates from RF_PI
#     name="AAOD"  # Optionally, assign a name
# )

ppe_var_aod_m_1 = copy.deepcopy(AOD_m_1[:,:-1]).groupby('time.month').mean().fillna(0)
ppe_var_aod_m_1 = ppe_var_aod_m_1.transpose("ensemble", "month", "lat", "lon")

ppe_var_aod_m_2 = copy.deepcopy(AOD_m_2[:,:-1]).groupby('time.month').mean().fillna(0)
ppe_var_aod_m_2 = ppe_var_aod_m_2.transpose("ensemble", "month", "lat", "lon")

ppe_var_aod_m_3 = copy.deepcopy(AOD_m_3[:,:-1]).groupby('time.month').mean().fillna(0)
ppe_var_aod_m_3 = ppe_var_aod_m_3.transpose("ensemble", "month", "lat", "lon")

#ppe_var_aod_ppe = copy.deepcopy(AOD_ppe[:]).fillna(0)
#ppe_var_ang_ppe = copy.deepcopy(ANG_ppe[:]).fillna(0)
ppe_var_cn = copy.deepcopy(CN_Burden[:,:-1]).groupby('time.month').mean().fillna(0)
ppe_var_cn = ppe_var_cn.transpose("ensemble", "month", "lat", "lon")

ppe_var_ssa = copy.deepcopy(SSA[:,:-1]).groupby('time.month').mean().fillna(0)
ppe_var_ssa = ppe_var_ssa.transpose("ensemble", "month", "lat", "lon")

ppe_var = copy.deepcopy(AOD[:,:-1]).groupby('time.month').mean().fillna(0)
ppe_var = ppe_var.transpose("ensemble", "month", "lat", "lon")

ppe_var_ang = copy.deepcopy(ANG[:,:-1]).groupby('time.month').mean().fillna(0)
ppe_var_ang = ppe_var_ang.transpose("ensemble", "month", "lat", "lon")

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

Y_test_ang, Y_train_ang = ppe_var_ang.isel(ensemble=test_indices), ppe_var_ang.isel(ensemble=train_indices)
Y_test_ssa, Y_train_ssa = ppe_var_ssa.isel(ensemble=test_indices), ppe_var_ssa.isel(ensemble=train_indices)
Y_test_aaod, Y_train_aaod = ppe_var_aaod.isel(ensemble=test_indices), ppe_var_aaod.isel(ensemble=train_indices)
Y_test_cn, Y_train_cn = ppe_var_cn.isel(ensemble=test_indices), ppe_var_cn.isel(ensemble=train_indices)
Y_test_aod_m_1, Y_train_aod_m_1 = ppe_var_aod_m_1.isel(ensemble=test_indices), ppe_var_aod_m_1.isel(ensemble=train_indices)
Y_test_aod_m_2, Y_train_aod_m_2 = ppe_var_aod_m_2.isel(ensemble=test_indices), ppe_var_aod_m_2.isel(ensemble=train_indices)
Y_test_aod_m_3, Y_train_aod_m_3 = ppe_var_aod_m_3.isel(ensemble=test_indices), ppe_var_aod_m_3.isel(ensemble=train_indices)

print('split train / test')

###################################
### EMULATOR ###
###################################

kernal=['Linear','Matern52'] # for  ERF, AOD_M_1, AOD_M_3
gp_model_aod_m1 = gp_model(X_train, Y_train_aod_m_1, kernel=kernal)
gp_model_aod_m1.train()

gp_model_aod_m3 = gp_model(X_train, Y_train_aod_m_3, kernel=kernal)
gp_model_aod_m3.train()

kernal=['Matern52']  # AOD
gp_model_ = gp_model(X_train, Y_train, kernel=kernal)
gp_model_.train()

kernal=['Bias','Matern52','Polynomial'] # ANG , SSA
gp_model_ang = gp_model(X_train, Y_train_ang, kernel=kernal)
gp_model_ang.train()

gp_model_ssa = gp_model(X_train, Y_train_ssa, kernel=kernal)
gp_model_ssa.train()

# gp_model_cn = gp_model(X_train, Y_train_cn, kernel=kernal)
# gp_model_cn.train()

gp_model_aod_m2 = gp_model(X_train, Y_train_aod_m_2, kernel=kernal)
gp_model_aod_m2.train()

kernal=['Matern52']  # AAOD
gp_model_aaod = gp_model(X_train, Y_train_aaod, kernel=kernal)
gp_model_aaod.train()
print('Emulator Training complete')

ppe_dist = copy.deepcopy(ppe_param)
# extract control row
control_row = ppe_dist.loc[["PPE_Control"]]
# number of new samples
# n_samples = os.getenv('NUMBER_OF_SAMPLES')
n_samples = int(os.getenv('NUMBER_OF_SAMPLES'))

print(f"number of samples = {n_samples}")
#n_samples = 20000
#n_samples = 8000
# sample each column with replacement from its empirical distribution
sampled_cols = {
    col: np.random.choice(ppe_dist[col].values, size=n_samples, replace=True)
    for col in ppe_dist.columns
}

new_samples_empirical = pd.DataFrame(sampled_cols)

# concat control row on top
ppe_non_normalized_new_samples = pd.concat([control_row, new_samples_empirical], ignore_index=True)


ppe_normalized_new_samples = (ppe_non_normalized_new_samples - ppe_non_normalized_new_samples.min()) / (ppe_non_normalized_new_samples.max() - ppe_non_normalized_new_samples.min())
# ppe_non_normalized_new_samples.OC_RAD_NI=ppe_non_normalized_new_samples.OC_RAD_NI.iloc[0]
# ppe_normalized_new_samples.OC_RAD_NI=ppe_normalized_new_samples.OC_RAD_NI.iloc[0]

ppe_normalized_new_samples.to_csv(f'/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PD/Emulated_data/Implausibility/Var/ppe_params_{n_samples}.csv')
ppe_non_normalized_new_samples.to_csv(f'/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PD/Emulated_data/Implausibility/Var/ppe_params_non_norm_{n_samples}.csv')


emulated_ang, var_ang = gp_model_ang.predict(ppe_normalized_new_samples.values)
# emulated_ang = emulated_ang.where(~land_mask_interp.isnull())

#emulated_ang.to_netcdf('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PD/Emulated_data/for_aerocom/Emulated_ANG_Southern_Ocean_DMS_SSA.nc')

emulated_aod, var_aod = gp_model_.predict(ppe_normalized_new_samples.values)
# emulated_aod = emulated_aod.where(~land_mask_interp.isnull())

emulated_ssa, var_ssa = gp_model_ssa.predict(ppe_normalized_new_samples.values)
# emulated_aod = emulated_aod.where(~land_mask_interp.isnull())

emulated_aaod, var_aaod = gp_model_aaod.predict(ppe_normalized_new_samples.values)
# emulated_aod = emulated_aod.where(~land_mask_interp.isnull())

emulated_aod_m_1, var_aod_m_1 = gp_model_aod_m1.predict(ppe_normalized_new_samples.values)
# emulated_aod = emulated_aod.where(~land_mask_interp.isnull())

emulated_aod_m_2, var_aod_m_2 = gp_model_aod_m2.predict(ppe_normalized_new_samples.values)
# emulated_aod = emulated_aod.where(~land_mask_interp.isnull())

emulated_aod_m_3, var_aod_m_3 = gp_model_aod_m3.predict(ppe_normalized_new_samples.values)
# emulated_aod = emulated_aod.where(~land_mask_interp.isnull())

# emulated_cn, var_cn = gp_model_cn.predict(ppe_normalized_new_samples.values)
# # emulated_aod = emulated_aod.where(~land_mask_interp.isnull())
# obs_vec = CN_Burden[:,-1].groupby('time.month').mean()
# valid_mask = ~np.isnan(obs_vec)
# emulated_cn_masked = emulated_cn.where(valid_mask)
# var_cn_masked = var_cn.where(valid_mask)
print('Saving emulated Variables')

base_dir = "/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PD/Emulated_data/Implausibility/Var"

emulated_aod.to_netcdf(f'{base_dir}/AOD/emulated_aod_{n_samples}.nc')
emulated_ang.to_netcdf(f'{base_dir}/ANG/emulated_ang_{n_samples}.nc')
emulated_ssa.to_netcdf(f'{base_dir}/SSA/emulated_ssa_{n_samples}.nc')
emulated_aaod.to_netcdf(f'{base_dir}/AAOD/emulated_aaod_{n_samples}.nc')
emulated_aod_m_1.to_netcdf(f'{base_dir}/AOD_Mode_1/emulated_aod_mode_1_{n_samples}.nc')
emulated_aod_m_2.to_netcdf(f'{base_dir}/AOD_Mode_2/emulated_aod_mode_2_{n_samples}.nc')
emulated_aod_m_3.to_netcdf(f'{base_dir}/AOD_Mode_3/emulated_aod_mode_3_{n_samples}.nc')

var_aod.to_netcdf(f'{base_dir}/AOD/emulated_var_aod_{n_samples}.nc')
var_ang.to_netcdf(f'{base_dir}/ANG/emulated_var_ang_{n_samples}.nc')
var_ssa.to_netcdf(f'{base_dir}/SSA/emulated_var_ssa_{n_samples}.nc')
var_aaod.to_netcdf(f'{base_dir}/AAOD/emulated_var_aaod_{n_samples}.nc')
var_aod_m_1.to_netcdf(f'{base_dir}/AOD_Mode_1/emulated_var_aod_mode_1_{n_samples}.nc')
var_aod_m_2.to_netcdf(f'{base_dir}/AOD_Mode_2/emulated_var_aod_mode_2_{n_samples}.nc')
var_aod_m_3.to_netcdf(f'{base_dir}/AOD_Mode_3/emulated_var_aod_mode_3_{n_samples}.nc')

