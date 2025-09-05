import xarray as xr
import numpy as np
import os
import pandas as pd
import sys

# Run this after POLDER process overpass complete. This will give you processed lat/lon readable data for each Land/Ocean for each variable.

ANG_HF_PD=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PD/Processed/ANG_550_865_Hifreq_PPE.nc').ANG_550nm_865nm#[0]
AOD_HF_PD=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PD/Processed/AOD_Hifreq_PPE.nc').TAU_2D_550nm#[0]
SSA_HF_PD=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PD/Processed/SSA_Hifreq_PPE.nc').__xarray_dataarray_variable__#[0]


lats=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PI/Processed/AOD_PPE.nc').lat
lons=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PI/Processed/AOD_PPE.nc').lon


#Variable_of_interest = input('What is your Variable of Interest? (SSA, ANG, AOD): ')
# Variable_of_interest = os.getenv('VARIABLE_OF_INTEREST')
# BIO_of_interest = os.getenv('BIO_OF_INTEREST')
# Get user inputs
Variable_of_interest = input('VARIABLE_OF_INTEREST (SSA, ANG, AOD): ')
BIO_of_interest = input('BIO_OF_INTEREST: ')

# Handle the loading of different datasets based on the variable of interest
if Variable_of_interest == 'SSA':
    # Load the SSA data
    VAR = xr.open_mfdataset(f'/home/ybhatti/prjs1076/Processed_Data/PD/Observational_Comparison/Pre_Processed/POLDER/{BIO_of_interest}/SSA/Polder_Overpass*', 
                            combine='nested', concat_dim='day').__xarray_dataarray_variable__.load()
elif Variable_of_interest == 'ANG':
    # Load the ANG data
    VAR = xr.open_mfdataset(f'/home/ybhatti/prjs1076/Processed_Data/PD/Observational_Comparison/Pre_Processed/POLDER/{BIO_of_interest}/ANG/Polder_Overpass*', 
                            combine='nested', concat_dim='day').load().ANG_550nm_865nm
elif Variable_of_interest == 'AOD':
    # Load the AOD data
    VAR = xr.open_mfdataset(f'/home/ybhatti/prjs1076/Processed_Data/PD/Observational_Comparison/Pre_Processed/POLDER/{BIO_of_interest}/AOD/Polder_Overpass*', 
                            combine='nested', concat_dim='day').load().TAU_2D_550nm
else:
    print("ERROR: Unknown variable of interest")
    sys.exit(1)  # Exit the script with an error code

#########################################################################
#### NEW ATTEMPT ###
import numpy as np
import xarray as xr

# Define empty list to store results
daily_data = []

# Loop through each day
for d in range(len(VAR)):  
    # Filter out NaN lat/lon values
    valid_mask = ~VAR[d].lon.isnull() & ~VAR[d].lat.isnull()
    print(d)
    # Assign lat/lon as new coordinates
    ssa_obs_new = VAR[d].sel(overpass=valid_mask).assign_coords(
        lat=("overpass", VAR[d].lat[valid_mask].values),
        lon=("overpass", VAR[d].lon[valid_mask].values),
        time=("overpass", VAR[d].time[valid_mask].values)
    )
    
    # Convert overpass index into (lat, lon)
    ssa_obs_new = ssa_obs_new.set_index(overpass=("lat", "lon","time"))
    
    # Group by (lat, lon) and take the mean to remove duplicates
    ssa_obs_new = ssa_obs_new.groupby("overpass").mean()
    
    # Unstack to reshape into (lat, lon, ensemble)
    ssa_obs_new = ssa_obs_new.unstack("overpass")
    mean_data = ssa_obs_new.mean(dim='time')
    mean_data = mean_data.assign_coords(time=ssa_obs_new['time'][0])

    # Append daily result to the list
    daily_data.append(mean_data)

#    end
    # if d == 10:
    #     end
# Stack into a single xarray DataArray (time, ensemble, lat, lon)
#end
#times = np.arange(len(daily_data))  # Create a new time axis
#daily_datas = [ds.assign_coords(time=t) for ds, t in zip(daily_data, times)]
#ssa_obs_full = xr.concat(daily_datas, dim="time")

ssa_obs_full = xr.concat(daily_data, dim="time").sortby('time')

# Assign time coordinates (assuming 336 consecutive days)
#ssa_obs_full = ssa_obs_full.assign_coords(time=np.arange(len(VAR)))

# Final DataArray: shape (time=336, ensemble, lat, lon)
print(ssa_obs_full)

ssa_obs_full.to_netcdf(f'/home/ybhatti/prjs1076/Processed_Data/PD/Observational_Comparison/Processed/POLDER/2010/{BIO_of_interest}/{Variable_of_interest}_POLDER_Interpolated_MODEL.nc')





# # Create new DataArray with reshaped data
# reshaped_da = xr.DataArray(
#     reshaped_data,
#     dims=('day', 'lat', 'lon', 'ensemble'),
#     coords={
#         'day': da.coords['day'],
#         'lat': (('day', 'lat'), np.array(unique_lat_per_day, dtype=object)),
#         'lon': (('day', 'lon'), np.array(unique_lon_per_day, dtype=object)),
#         'ensemble': da.coords['ensemble']
#     }
# )

# print(reshaped_da)

# #####################################################
 
# import scipy.interpolate

# # Define target grid
# lat_bins = np.linspace(VAR.lat.min(), VAR.lat.max(), num=180)  # Adjust resolution
# lon_bins = np.linspace(VAR.lon.min(), VAR.lon.max(), num=360)

# # Create a new dataset with regular lat-lon bins
# lat_grid, lon_grid = np.meshgrid(lat_bins, lon_bins)

# # Prepare an empty list to store interpolated results
# var_interp_list = []

# # Loop over each day
# for day_idx in range(VAR.sizes["day"]):
#     ens_interp_list = []  # Store interpolations for each ensemble at this day
    
#     for ens_idx in range(VAR.sizes["ensemble"]):
#         # Extract SSA, lat, and lon for the current day and ensemble
#         var_ens = VAR.isel(day=day_idx, ensemble=ens_idx).values.flatten()
#         lat_day = VAR.lat.isel(day=day_idx).values.flatten()
#         lon_day = VAR.lon.isel(day=day_idx).values.flatten()
    
#         # Remove NaNs for interpolation
#         mask = ~np.isnan(lat_day) & ~np.isnan(lon_day) & ~np.isnan(var_ens)
#         lat_day, lon_day, var_ens = lat_day[mask], lon_day[mask], var_ens[mask]
    
#         # Perform interpolation
#         var_interp = scipy.interpolate.griddata(
#             (lat_day, lon_day), var_ens, (lat_grid, lon_grid), method="linear"
#         )
        
#         ens_interp_list.append(var_interp)  # Store for this ensemble
        
#     var_interp_list.append(ens_interp_list)  # Store all ensembles for this day
#     print(day_idx)
    
# # Convert list to numpy array and reshape
# var_interp_array = np.array(var_interp_list)  # Shape: (day, ensemble, lat, lon)

# # Convert to xarray DataArray
# var_interp_xr = xr.DataArray(
#     var_interp_array.transpose(0, 1, 3, 2),  # Swap lat and lon
#     coords={"day": VAR.day, "ensemble": VAR.ensemble, "lat": lat_bins, "lon": lon_bins},
#     dims=["day", "ensemble", "lat", "lon"],  # Corrected dimension order
# )

# var_interp_xr.to_netcdf(f'/home/ybhatti/prjs1076/Processed_Data/PD/Observational_Comparison/Pre_Processed/POLDER/OCEAN/{Variable_of_interest}/POLDER_Interpolated_MODEL.nc')
