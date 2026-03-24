import numpy as np
import xarray as xr
import gc
import os

# os.chdir('/home/ybhatti2/HPC_Jupyter/Python_Scripts/')


from my_functions import *

# os.chdir('/home/ybhatti2/HPC_Jupyter/Python_Scripts/Analysis/')

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
elif var_name == "CDNC_OCI_spx":
    obs_data= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/CDNC_OCI_Filtered_Interpolated_MODEL.nc').CDNC_INCL_CT.load()
    land_mask = xr.open_dataset('/home/ybhatti2/prjs1474/Pace_PPE_Output/PPE_Experiments/PPE_ENS_1/conc_aerocom_DMS_sea.nc').DMS_sea.mean('time')

elif var_name == "CDNC_Filtered_OCI":
    obs_data= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/CDNC_OCI_Interpolated_MODEL_Swath.nc').CDNC_INCL_CT.load()
    land_mask = xr.open_dataset('/home/ybhatti2/prjs1474/Pace_PPE_Output/PPE_Experiments/PPE_ENS_1/conc_aerocom_DMS_sea.nc').DMS_sea.mean('time')

elif var_name == "CDNC_Filtered_HARP":
    obs_data= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/CDNC_HARP_Interpolated_MODEL_Swath.nc').CDNC_INCL_CT.load()
    land_mask = xr.open_dataset('/home/ybhatti2/prjs1474/Pace_PPE_Output/PPE_Experiments/PPE_ENS_1/conc_aerocom_DMS_sea.nc').DMS_sea.mean('time')

elif var_name == "CDNC_Filtered_SPEXone":
    obs_data= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/CDNC_SPEXone_Filtered_Interpolated_MODEL.nc').CDNC_INCL_CT.load()
    land_mask = xr.open_dataset('/home/ybhatti2/prjs1474/Pace_PPE_Output/PPE_Experiments/PPE_ENS_1/conc_aerocom_DMS_sea.nc').DMS_sea.mean('time')

elif var_name == "REFFL_CT_OCI":
    obs_data= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/REFFL_CT_OCI_Interpolated_MODEL_Swath.nc').REFFL_CT.load()
    land_mask = xr.open_dataset('/home/ybhatti2/prjs1474/Pace_PPE_Output/PPE_Experiments/PPE_ENS_1/conc_aerocom_DMS_sea.nc').DMS_sea.mean('time')
elif var_name == "REFFL_CT_SPEXone":
    obs_data= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/REFFL_CT_SPEXone_Interpolated_MODEL.nc').REFFL_CT.load()
    land_mask = xr.open_dataset('/home/ybhatti2/prjs1474/Pace_PPE_Output/PPE_Experiments/PPE_ENS_1/conc_aerocom_DMS_sea.nc').DMS_sea.mean('time')
elif var_name == "REFFL_CT_HARP":
    obs_data= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/REFFL_CT_HARP2_Interpolated_MODEL_Swath.nc').REFFL_CT.load()
    land_mask = xr.open_dataset('/home/ybhatti2/prjs1474/Pace_PPE_Output/PPE_Experiments/PPE_ENS_1/conc_aerocom_DMS_sea.nc').DMS_sea.mean('time')

elif var_name == "TAU355":
    obs_data = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/TAU355_2km_EarthCARE_Interpolated_MODEL.nc').TAU_3D_355nm.load()
elif var_name == "TAU355_daily":
    obs_data = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/TAU355_2km_EarthCARE_Interpolated_MODEL_daily.nc').TAU_3D_355nm.load()

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
    ppe_vec = obs_data.sel(ensemble=slice(0,250)).groupby('time.month').mean()
    obs_vec = obs_data.sel(ensemble=-1).groupby('time.month').mean()
    print("Obtained obs and ppe data")
except:
    obs_data = obs_data.transpose('month', 'ensemble', 'lat', 'lon')
    ppe_vec = obs_data[:, :]
    obs_vec = obs_data.sel(ensemble=0)

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

regional_mask = xr.open_dataset('/gpfs/home3/ybhatti2/HPC_Jupyter/Python_Scripts/Analysis/Regional_Mask_Ships_African_Horn.nc').regional_mask.load()

region_names = [
    'Australia', 'Europe', 'South-East Asia', 'Siberia','Savannah',
    'North America', 'Boreal America',
    'Amazon', 'North Atlantic', 'South Atlantic', 'North Pacific',
    'South Pacific', 'Tropic Atlantic' , 'Tropic Pacific', 'Tropic Indian', 'South Indian',
    'Dust belt','Shipping','African Horn'
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
    'Dust belt','Shipping','African Horn','Global'
]

# ----------------------------------------------------
# Add Hemispheric Difference (NH–SH) as final region
# ----------------------------------------------------
if var_name.startswith(("CDNC", "REFFL_CT")):
    print(f"Hemispheric Difference for {var_name}")
    ocean_mask = land_mask.notnull()   # True over ocean, False over land
    # emulated_ocean = ppe_vec.where(ocean_mask)
    # obs_ocean = obs_vec.where(ocean_mask)
    emulated_ocean = emulated_masked_rejected.where(ocean_mask)
    obs_ocean = obs_vec_rejected.where(ocean_mask)

    obs_ocean_ens = obs_ocean.expand_dims(ensemble=[-1])
    emulated_cdnc_ocean = xr.concat([emulated_ocean,obs_ocean_ens], dim="ensemble")

    # Northern Hemisphere (0–90°N)
    NH = areaweight(
        emulated_cdnc_ocean.sel(lat=slice(60, 30)),
        emulated_cdnc_ocean.sel(lat=slice(60, 30)).lat
    )
    
    # Southern Hemisphere (0–90°S)
    SH = areaweight(
        emulated_cdnc_ocean.sel(lat=slice(-30, -60)),
        emulated_cdnc_ocean.sel(lat=slice(-30, -60)).lat
    )
    HEM_Diff = NH - SH

    # hemi_vals = HEM_Diff.values  # 
    # # Add to Region_data (as model)
    # Region_data.append(hemi_vals)

    # # Add OBS row: select obs ensemble = -1
    # hemi_obs = hemi_vals[:, -1]
    # Region_OBS.append(hemi_obs)

    HEM_DIFF = HEM_Diff.values  # convert to numpy if needed

    # Add to Region_data (month × sample)
    Region_data = np.concatenate(
        [Region_data, HEM_DIFF[np.newaxis, :, :]],
        axis=0
    )

    # Add to boxplot data (sample only)
    HEM_Diff_box = HEM_DIFF.mean(axis=0)  # monthly mean removed to match boxplot dims
    Region_data_box = np.concatenate(
        [Region_data_box, HEM_Diff_box[np.newaxis, :]],
        axis=0
    )

    region_names = [
    'Australia', 'Europe', 'South-East Asia', 'Siberia','Savannah',
    'North America', 'Boreal America',
    'Amazon', 'North Atlantic', 'South Atlantic', 'North Pacific',
    'South Pacific', 'Tropic Atlantic' , 'Tropic Pacific', 'Tropic Indian', 'South Indian',
    'Dust belt','Shipping','African Horn','Global','Hemispheric_Difference'
    ]


if var_name == 'TAU355':
    Region_data = xr.DataArray(
        Region_data,
        dims=("region", "month",  "sample"),
        coords={"region": region_names, "sample": sample_coords, "month": np.arange(1, 9)},
        name=var_name
    )
    Region_data.to_netcdf(f"{base_dir_regional}/{var_name}/Regional_Uncertainty_{var_name.lower()}_{n_samples}.nc")
else:
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
