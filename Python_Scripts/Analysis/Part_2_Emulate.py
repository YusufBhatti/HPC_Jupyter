import numpy as np
import xarray as xr
import gc
import os
from my_functions import *

print('Part 2')




# CN_Burden = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/CN_BURDEN_POLDER_Interpolated_MODEL.nc').CN_BURDEN





print('loaded variables')


# ===============================
# CONFIGURATION
# ===============================
#var_name = "AOD"  # <<< CHANGE THIS (AOD, ANG, SSA, or AAOD)
var_name = os.getenv('VARIABLE_NAME')

base_dir = "/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PD/Emulated_data/Implausibility/Var"
n_samples = int(os.getenv('NUMBER_OF_SAMPLES'))

# ===============================
# LOAD DATA
# ===============================
# --- Uncertainty settings ---
if var_name == "AOD":
    instr_frac, instr_abs, repr_frac = 0.15, 0.035, 0.10
    obs_data = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_POLDER_Interpolated_MODEL.nc').TAU_2D_550nm.load()
elif var_name == "ANG":
    instr_frac, instr_abs, repr_frac = 0, 0.25, 0.10
    obs_data = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/ANG_POLDER_Interpolated_MODEL.nc').ANG_440nm_670nm.load()

# instr_abs_ssa_filter  = 0.06   # absolute instrument uncertainty

elif var_name == "SSA":
    instr_frac, instr_abs, repr_frac = 0, 0.06, 0
    AOD = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_POLDER_Interpolated_MODEL.nc').TAU_2D_550nm.load()
    obs_data = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/SSA_POLDER_Interpolated_MODEL.nc').__xarray_dataarray_variable__.load()
    obs_data = obs_data.where(AOD[:,-1] > 0.2)

elif var_name == "AAOD":
    instr_frac, instr_abs, repr_frac = 0.10, 0.01, 0.10
    obs_data = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AAOD_POLDER_Interpolated_MODEL.nc').AAOD.load()

elif var_name == "AOD_Mode_1":
    instr_frac, instr_abs, repr_frac = 0.10, 0.03, 0.10
    obs_data= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_Mode_1_POLDER_Interpolated_MODEL.nc').__xarray_dataarray_variable__.load()

elif var_name == "AOD_Mode_2":
    instr_frac, instr_abs, repr_frac = 0.10, 0.03, 0.10
    obs_data= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_Mode_2_POLDER_Interpolated_MODEL.nc').TAU_2D_MODE_CI_550nm.load()

elif var_name == "AOD_Mode_3":
    instr_frac, instr_abs, repr_frac = 0.10, 0.03, 0.10
    obs_data= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_Mode_3_POLDER_Interpolated_MODEL.nc').TAU_2D_MODE_CS_550nm.load()

else:
    raise ValueError("Unknown variable name.")


print(f"Processing variable: {var_name}")
emulated = xr.open_dataarray(f"{base_dir}/{var_name}/emulated_{var_name.lower()}_{n_samples}.nc").load()
# obs_map = {
#     "AOD": AOD, "ANG": ANG, "SSA": SSA, "AAOD": AAOD, "AAOD": AAOD, "AOD_Mode_1": AOD_Mode_1, "AOD_Mode_2": AOD_Mode_2, "AOD_Mode_3": AOD_Mode_2,
# }
obs_vec = obs_data[:, -1].groupby('time.month').mean()
valid_mask = ~np.isnan(obs_vec)
emulated_masked = emulated.where(valid_mask)

    
# --- Compute variances ---
def compute_variances(obs, instr_frac, instr_abs, repr_frac):
    if var_name == 'AOD' or var_name == 'AOD_Mode_1' or var_name == 'AOD_Mode_2' or var_name == 'AOD_Mode_3':
        # Elementwise: choose larger of fractional (10%) or absolute (0.03)
        frac_unc = instr_frac * obs
        abs_unc = xr.full_like(obs, instr_abs)
        instr_unc = xr.where(frac_unc > abs_unc, frac_unc, abs_unc)
        instr_unc = instr_unc.where(~np.isnan(obs))
        Var_O = instr_unc**2
        Var_R = (repr_frac * obs)**2
        
    elif var_name == 'ANG':
        abs_unc = xr.full_like(obs, instr_abs)
        Var_O = (abs_unc.where(~np.isnan(obs)))**2
        Var_R = (repr_frac * obs)**2

    elif var_name == 'SSA':
        abs_unc = xr.full_like(obs, instr_abs)
        Var_O = (abs_unc.where(~np.isnan(obs)))**2
        Var_R = (repr_frac * obs)**2

    elif var_name == 'AAOD':
        frac_unc = instr_frac * obs
        abs_unc = xr.full_like(obs, instr_abs)
        instr_unc = xr.where(frac_unc > abs_unc, frac_unc, abs_unc)
        instr_unc = instr_unc.where(~np.isnan(obs))
        Var_O = instr_unc**2
        Var_R = (repr_frac * obs)**2

    elif var_name == 'CN_Burden':
        # fractional (30%) 
        frac_unc = instr_frac * obs
        instr_unc = frac_unc.where(~np.isnan(obs))
        Var_O = instr_unc**2
        Var_R = (repr_frac * obs)**2
    else:
        raise ValueError("var must be 'AOD' or 'ANG' or 'SSA' or 'CN_Burden'")

    return Var_O, Var_R
Var_O, Var_R = compute_variances(obs_vec, instr_frac, instr_abs, repr_frac)
print("Var_O, Var_R")

# --- Load emulator variance if available ---
try:
    var_emulated = xr.open_dataarray(f"{base_dir}/{var_name}/emulated_var_{var_name.lower()}_{n_samples}.nc")
    var_emulated = var_emulated.where(valid_mask)
    print("Valid variance")

except FileNotFoundError:
    print("No emulator variance file found; assuming zero variance.")
    var_emulated = xr.zeros_like(emulated_masked)

# --- Implausibility computation ---
def compute_implausibility(emulated, var_emulated, obs, Var_O, Var_R):
    diff = np.abs(emulated - obs)
    Var_total = var_emulated + Var_O + Var_R
    I_field = diff / np.sqrt(Var_total)
    I_max = I_field.max(dim=("month", "lat", "lon"))
    return I_field, I_max

I_field, I_max = compute_implausibility(emulated_masked, var_emulated, obs_vec, Var_O, Var_R)
I_field.to_netcdf(f"{base_dir}/{var_name}/I_field_{var_name.lower()}_{n_samples}.nc")

# --- Rejection map ---
def rejection_map_monthly(I_field, threshold=1):
    n_sample, n_month, n_lat, n_lon = I_field.shape
    reject_mask = np.full((n_month, n_lat, n_lon), True)
    for m in range(n_month):
        for lat_i in range(n_lat):
            for lon_i in range(n_lon):
                vals = I_field.isel(month=m, lat=lat_i, lon=lon_i).values
                vals = vals[~np.isnan(vals)]
                q2p5 = np.nanpercentile(vals, 2.5) if len(vals) > 0 else np.nan
                reject_mask[m, lat_i, lon_i] = False if np.isnan(q2p5) or q2p5 > threshold else True
    return xr.DataArray(reject_mask, coords={"month": I_field.month, "lat": I_field.lat, "lon": I_field.lon},
                        dims=("month", "lat", "lon"), name="reject_mask")

reject_map = rejection_map_monthly(I_field, threshold=1)
reject_map.to_netcdf(f"{base_dir}/{var_name}/reject_map_{var_name.lower()}_{n_samples}.nc")

implaus_mask = I_field.where(reject_map)
implaus_mask.to_netcdf(f"{base_dir}/{var_name}/implausibility_values_{var_name.lower()}_{n_samples}.nc")
print(f"reject_map and implaus_mask for {var_name}")

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

regional_mask = xr.open_dataset('/gpfs/home3/ybhatti2/HPC_Jupyter/Python_Scripts/Analysis/Regional_Mask.nc').regional_mask.load()

region_names = [
    'Australia', 'Europe', 'South-East Asia', 'Siberia','Savannah',
    'North America', 'Boreal America',
    'Amazon', 'North Atlantic', 'South Atlantic', 'North Pacific',
    'South Pacific', 'Tropic Atlantic' , 'Tropic Pacific', 'Tropic Indian', 'South Indian',
    'Dust belt'
]

Region_data = []
Region_OBS = []
boxplot_ppe_data = []

emulated_masked_rejected = emulated_masked.where(reject_map)
obs_vec_rejected = obs_vec.where(reject_map)

for region_name in region_names:
    print(f"{region_name} for {var_name}")
    region_gp = Interpolate_Regional_uncertainty(emulated_masked_rejected, region_name, regional_mask)
    gpmeans = areaweight(region_gp, region_gp.lat)
    Region_data.append(gpmeans.values)
    boxplot_ppe_data.append(gpmeans.mean('month'))
    region_obs = Interpolate_Regional_uncertainty(obs_vec_rejected, region_name, regional_mask)
    obsmeans = areaweight(region_obs, region_obs.lat)
    Region_OBS.append(obsmeans.values)

Region_data = np.array(Region_data)
Region_OBS = np.array(Region_OBS)
boxplot_ppe_data = np.array(boxplot_ppe_data)

Region_data = np.concatenate([Region_data, Region_OBS[:, np.newaxis, :]], axis=1)
sample_coords = np.append(emulated_masked_rejected.sample.values, -1)
sample_coords_box = emulated_masked_rejected.sample.values

Region_data = xr.DataArray(
    Region_data,
    dims=("region", "sample", "month"),
    coords={"region": region_names, "sample": sample_coords, "month": np.arange(1, 13)},
    name=var_name
)
Region_data.to_netcdf(f"{base_dir}/{var_name}/Regional_Uncertainty_{var_name.lower()}_{n_samples}.nc")

box_data = xr.DataArray(
    boxplot_ppe_data,
    dims=("region", "sample"),
    coords={"region": region_names, "sample": sample_coords_box},
    name=var_name
)
Region_data.to_netcdf(f"{base_dir}/{var_name}/Boxplot_Unconstrained_Uncertainty_{var_name.lower()}_{n_samples}.nc")


print(f"Completed variable: {var_name}")
gc.collect()
