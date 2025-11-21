import numpy as np
import xarray as xr
import gc
import os
from my_functions import *

print('Part 2')


# ===============================
# CONFIGURATION
# ===============================
#var_name = "AOD"  # <<< CHANGE THIS (AOD, ANG, SSA, or AAOD)
var_name = os.getenv('VARIABLE_NAME')

#base_dir = "/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PD/Emulated_data/Implausibility/Var"
n_samples = int(os.getenv('NUMBER_OF_SAMPLES'))
#base_dir = "/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PD/Emulated_data/Implausibility/Var"
base_dir = os.getenv('BASE_DIR')

# ===============================
# LOAD DATA
# ===============================
# --- Uncertainty settings ---
if var_name == "AOD":
    instr_frac, instr_abs, repr_frac = 0.15, 0.035, 0.10
    instr_frac, instr_abs, repr_frac = 0.1, 0.030, 0.10
    obs_data = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_POLDER_Interpolated_MODEL.nc').TAU_2D_550nm.load()
    
elif var_name == "AI":
    instr_frac, instr_abs, repr_frac = 0.15, 0.035, 0.10
    instr_frac, instr_abs, repr_frac = 0.15, 0.025, 0.10
    obs_data = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AI_POLDER_Interpolated_MODEL.nc').__xarray_dataarray_variable__.load()

elif var_name == "ANG":
    instr_frac, instr_abs, repr_frac = 0, 0.25, 0.10
    instr_frac, instr_abs, repr_frac = 0, 0.2, 0.10
    obs_data = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/ANG_POLDER_Interpolated_MODEL.nc').ANG_440nm_670nm.load()

# instr_abs_ssa_filter  = 0.06   # absolute instrument uncertainty

elif var_name == "SSA":
    instr_frac, instr_abs, repr_frac = 0, 0.04, 0.10
    instr_frac, instr_abs, repr_frac = 0, 0.04, 0.10
    AOD = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_POLDER_Interpolated_MODEL.nc').TAU_2D_550nm.load()
    obs_data = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/SSA_POLDER_Interpolated_MODEL.nc').__xarray_dataarray_variable__.load()
    #obs_data = obs_data.where(AOD[:,-1] > 0.2)
    obs_data = obs_data.where(AOD[:,-1] > 0.1)

elif var_name == "AAOD":
    instr_frac, instr_abs, repr_frac = 0.10, 0.03, 0.10
    instr_frac, instr_abs, repr_frac = 0.10, 0.03, 0.10
    obs_data = xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AAOD_POLDER_Interpolated_MODEL.nc').AAOD.load()

elif var_name == "AOD_Mode_1":
    instr_frac, instr_abs, repr_frac = 0.10, 0.03, 0.10
    instr_frac, instr_abs, repr_frac = 0.10, 0.035, 0.10
    obs_data= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_Mode_1_POLDER_Interpolated_MODEL.nc').__xarray_dataarray_variable__.load()

elif var_name == "AOD_Mode_2":
    instr_frac, instr_abs, repr_frac = 0.10, 0.03, 0.10
#    instr_frac, instr_abs, repr_frac = 0.06, 0.025, 0.10
    obs_data= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_Mode_2_POLDER_Interpolated_MODEL.nc').TAU_2D_MODE_CI_550nm.load()
elif var_name == "AOD_Mode_3":
    instr_frac, instr_abs, repr_frac = 0.10, 0.03, 0.10
    instr_frac, instr_abs, repr_frac = 0.05, 0.02, 0.10
    obs_data= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_Mode_3_POLDER_Interpolated_MODEL.nc').TAU_2D_MODE_CS_550nm.load()
elif var_name == "AOD_Mode_Coarse":
    instr_frac, instr_abs, repr_frac = 0.06, 0.025, 0.10
    obs_data= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/AOD_Mode_Coarse_POLDER_Interpolated_MODEL.nc').__xarray_dataarray_variable__.load()
elif var_name == "CDNC":
    instr_frac, instr_abs, repr_frac = 0.40, 0.00, 0.10
    obs_data= xr.open_dataset('/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PACE_Co_locating/Processed/CDNC_OCI_Interpolated_MODEL.nc').CDNC_INCL_CT.load()

else:
    raise ValueError("Unknown variable name.")

obs_data = obs_data.transpose('time', 'ensemble', 'lat', 'lon')

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
    if var_name == 'AOD' or var_name == 'AOD_Mode_1' or var_name == 'AOD_Mode_2' or var_name == 'AOD_Mode_3' or var_name == 'AI' or var_name == 'AOD_Mode_Coarse':
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

    elif var_name == 'CN_Burden' or var_name == 'CDNC':
        # fractional (30%) or 50% for CDNC
        print(f"Uncertainties are {instr_frac}% and {instr_abs} absolute")
        frac_unc = instr_frac * obs
        instr_unc = frac_unc.where(~np.isnan(obs))
        Var_O = instr_unc**2
        Var_R = (repr_frac * obs)**2
    else:
        raise ValueError("var must be 'AOD' or 'ANG' or 'SSA' or 'CN_Burden' or 'CDNC'")

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
#I_field.to_netcdf(f"{base_dir}/{var_name}/I_field_{var_name.lower()}_{n_samples}.nc")
print(f"saved I_field for {var_name}")

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

reject_file = f"{base_dir}/{var_name}/reject_map_{var_name.lower()}_{n_samples}.nc"
if os.path.exists(reject_file):
    os.remove(reject_file)
    print(f"Removing previous reject_map for {var_name}")


reject_map = rejection_map_monthly(I_field, threshold=1)
reject_map.to_netcdf(reject_file)

#implaus_mask = I_field.where(reject_map)
#implaus_mask.to_netcdf(f"{base_dir}/{var_name}/implausibility_values_{var_name.lower()}_{n_samples}.nc")
print(f"Saving reject_map and implaus_mask for {var_name}")

outfile = f"{base_dir}/{var_name}/emulated_{var_name.lower()}_{n_samples}.nc"
if os.path.exists(outfile):
    os.remove(outfile)
    print(f"Removing previous Emulation for {var_name}")

outfile = f"{base_dir}/{var_name}/emulated_var_{var_name.lower()}_{n_samples}.nc"
if os.path.exists(outfile):
    os.remove(outfile)
    print(f"Removing previous Emulation Var for {var_name}")

gc.collect()
print(f"Finishing {var_name} for Part 2")
