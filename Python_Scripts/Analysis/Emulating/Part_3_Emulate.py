import numpy as np
import xarray as xr
import gc
import os
from my_functions import *

print('Part 3')
print('This Part will Calculate Regional averages from the simulations using the rejection map from Part 2 to filter out rejected cells.')


# ===============================
# CONFIGURATION
# ===============================
#var_name = "AOD"  # <<< CHANGE THIS (AOD, ANG, SSA, or AAOD)
var_name = os.getenv('VARIABLE_NAME')

#base_dir = "/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PD/Emulated_data/Implausibility/Var"
n_samples = int(os.getenv('NUMBER_OF_SAMPLES'))
#base_dir = "/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PD/Emulated_data/Implausibility/Var"
base_dir = os.getenv('BASE_DIR')

base_dir_regional = base_dir + '/Regional'
os.makedirs(base_dir_regional, exist_ok=True)
os.makedirs(base_dir_regional+'/'+var_name, exist_ok=True)


# ===============================
# LOAD DATA
# ===============================
try:
    reject_map = xr.open_dataset(f"{base_dir}/{var_name}/reject_map_{var_name.lower()}_{n_samples}.nc").reject_mask
    print(f"reject_map and implaus_mask for {var_name}")
except:
    print(f" NO reject_map and implaus_mask for {var_name}")
    pass
# --- Uncertainty settings ---
if var_name == "AOD":
    obs_data = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_POLDER_Interpolated_MODEL.nc').TAU_2D_550nm.load()
elif var_name == "AI":
    obs_data = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AI_POLDER_Interpolated_MODEL.nc').__xarray_dataarray_variable__.load()

elif var_name == "ANG":
    obs_data = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/ANG_POLDER_Interpolated_MODEL.nc').ANG_440nm_670nm.load()
elif var_name == "SSA":
    AOD = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_POLDER_Interpolated_MODEL.nc').TAU_2D_550nm.load()
    obs_data = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/SSA_POLDER_Interpolated_MODEL.nc').__xarray_dataarray_variable__.load()
    #obs_data = obs_data.where(AOD[:,-1] > 0.2)
    obs_data = obs_data.where(AOD[:,-1] > 0.1)

elif var_name == "AAOD":
    obs_data = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AAOD_POLDER_Interpolated_MODEL.nc').AAOD.load()

elif var_name == "AOD_Mode_1":
    obs_data= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_Mode_1_POLDER_Interpolated_MODEL.nc').__xarray_dataarray_variable__.load()

elif var_name == "AOD_Mode_2":
    obs_data= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_Mode_2_POLDER_Interpolated_MODEL.nc').TAU_2D_MODE_CI_550nm.load()
elif var_name == "AOD_Mode_3":
    obs_data= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_Mode_3_POLDER_Interpolated_MODEL.nc').TAU_2D_MODE_CS_550nm.load()
elif var_name == "AOD_Mode_Coarse":
    obs_data= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_Mode_Coarse_POLDER_Interpolated_MODEL.nc').__xarray_dataarray_variable__.load()
elif var_name == "CDNC":
    obs_data= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/CDNC_OCI_Interpolated_MODEL.nc').CDNC_INCL_CT.load()
elif var_name == "ERF":
    obs_data= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PD/ERF_PPE.nc').__xarray_dataarray_variable__.load()
elif var_name == "ERFaci":
    obs_data= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PD/ERFaci_PPE.nc').__xarray_dataarray_variable__.load()
elif var_name == "ERFari":
    obs_data= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PD/ERFari_PPE.nc').__xarray_dataarray_variable__.load()
else:
    raise ValueError("Unknown variable name.")

try:
    obs_data = obs_data.transpose('time', 'ensemble', 'lat', 'lon')
    ppe_vec = obs_data[:, :-1].groupby('time.month').mean()
    obs_vec = obs_data[:, -1].groupby('time.month').mean()
    print("Obtained obs and ppe data")
except:
    obs_data = obs_data.transpose('month', 'ensemble', 'lat', 'lon')
    ppe_vec = obs_data[:, :]
    obs_vec = obs_data[:, 0]

    print("Obtained ppe data")

print(f"Processing variable: {var_name}")


# --- Regional aggregation ---
def Interpolate_Regional_uncertainty(Predicted, region_name, ds_mask):
    """
    Interpolates Predicted data onto the grid of the selected region
    and applies the regional mask.
    """
    # Extract the region-specific mask
    try:
        region_mask = ds_mask.sel(
            region=ds_mask.where(ds_mask.region_name == region_name, drop=True).region[0]
        )
    except:
        region_mask = ds_mask.sel(
            region=ds_mask.where(ds_mask.region_name == region_name, drop=True).region
    )

    masked_array = Predicted.where(region_mask)
    return masked_array

regional_mask = xr.open_dataset('/gpfs/home3/ybhatti2/HPC_Jupyter/Python_Scripts/Analysis/Regional_Mask_Ships.nc').regional_mask.load()

region_names = [
    'Australia', 'Europe', 'South-East Asia', 'Siberia','Savannah',
    'North America', 'Boreal America',
    'Amazon', 'North Atlantic', 'South Atlantic', 'North Pacific',
    'South Pacific', 'Tropic Atlantic' , 'Tropic Pacific', 'Tropic Indian', 'South Indian',
    'Dust belt','Shipping'
]

Region_data = []
Region_OBS = []
boxplot_ppe_data = []
boxplot_ppe_data_OBS=[]

# try:
#     emulated_masked_rejected = ppe_vec.where(reject_map)
#     obs_vec_rejected = obs_vec.where(reject_map)
# except:
if var_name == 'ERF' or var_name == 'ERFaci' or var_name == 'ERFari':
    emulated_masked_rejected = ppe_vec
    obs_vec_rejected = obs_vec
    print(f"{var_name} so NOT needed to mask rejection cells")
else:
    emulated_masked_rejected = ppe_vec.where(reject_map)
    obs_vec_rejected = obs_vec.where(reject_map)
    print(f"{var_name} so needed to mask rejection cells")

    
for region_name in region_names:
    print(f"{region_name} for {var_name}")
    region_gp = Interpolate_Regional_uncertainty(emulated_masked_rejected, region_name, regional_mask)
    gpmeans = areaweight(region_gp, region_gp.lat)
    Region_data.append(gpmeans.values)
    boxplot_ppe_data.append(gpmeans.mean('month'))
    
    region_obs = Interpolate_Regional_uncertainty(obs_vec_rejected, region_name, regional_mask)
    obsmeans = areaweight(region_obs, region_obs.lat)
    Region_OBS.append(obsmeans.values)
    boxplot_ppe_data_OBS.append(obsmeans.mean('month'))
    #end
boxplot_ppe_data.append(areaweight(emulated_masked_rejected, emulated_masked_rejected.lat).mean('month'))
boxplot_ppe_data_OBS.append(areaweight(obs_vec_rejected, obs_vec_rejected.lat).mean('month'))

Region_data.append(areaweight(emulated_masked_rejected, emulated_masked_rejected.lat))
Region_OBS.append(areaweight(obs_vec_rejected, obs_vec_rejected.lat))
print(f'Global for {var_name}')
Region_data = np.array(Region_data)
Region_OBS = np.array(Region_OBS)
boxplot_ppe_data = np.array(boxplot_ppe_data)
boxplot_ppe_data_OBS = np.array(boxplot_ppe_data_OBS)

Region_data = np.concatenate([Region_data, Region_OBS[:, :, np.newaxis]], axis=2)
#Region_data_box = np.concatenate([boxplot_ppe_data, boxplot_ppe_data_OBS[:, np.newaxis, :]], axis=1)
Region_data_box = np.concatenate([boxplot_ppe_data, boxplot_ppe_data_OBS[:, np.newaxis]], axis=1)

sample_coords = np.append(emulated_masked_rejected.ensemble.values, -1)
# sample_coords_box = emulated_masked_rejected.sample.values

region_names = [
    'Australia', 'Europe', 'South-East Asia', 'Siberia','Savannah',
    'North America', 'Boreal America',
    'Amazon', 'North Atlantic', 'South Atlantic', 'North Pacific',
    'South Pacific', 'Tropic Atlantic' , 'Tropic Pacific', 'Tropic Indian', 'South Indian',
    'Dust belt','Shipping','Global'
]

Region_data = xr.DataArray(
    Region_data,
    dims=("region", "month",  "sample"),
    coords={"region": region_names, "sample": sample_coords, "month": np.arange(1, 13)},
    name=var_name
)
Region_data.to_netcdf(f"{base_dir_regional}/{var_name}/Regional_Uncertainty_{var_name.lower()}_{n_samples}.nc")



box_data = xr.DataArray(
    Region_data_box,
    dims=("region", "sample"),
    coords={"region": region_names, "sample": sample_coords},
    name=var_name
)
box_data.to_netcdf(f"{base_dir_regional}/{var_name}/Boxplot_Unconstrained_Uncertainty_{var_name.lower()}_{n_samples}.nc")


print(f"Completed variable: {var_name} for Part 3")
gc.collect()
