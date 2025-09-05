import xarray as xr
import numpy as np
import os
import pandas as pd
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import sys

## This is first step. Run this 6 times for the 3 different variables across 2 land/ocean

ANG_HF_PD=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PD/Processed/ANG_550_865_Hifreq_PPE.nc').ANG_550nm_865nm#[0]
AOD_HF_PD=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PD/Processed/AOD_Hifreq_PPE.nc').TAU_2D_550nm#[0]
SSA_HF_PD=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PD/Processed/SSA_Hifreq_PPE.nc').__xarray_dataarray_variable__#[0]


lats=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PI/Processed/AOD_PPE.nc').lat
lons=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PI/Processed/AOD_PPE.nc').lon


reference_time = xr.DataArray(
    pd.date_range("2010-01-01", "2010-12-31", freq="3h"),
    dims="time",
    name="time"
    )


# Function to extract date from filename
def extract_date(file_path):
    # Assuming date format is in 'YYYYMMDD' as 'SRON-POLDER-L2-YYYYMMDD-LAND'
    filename = os.path.basename(file_path)
    date_str = filename.split('-')[3]  # Extracts the 'YYYYMMDD' part
    return date_str
def convert_julday_to_hours(file):
    #date=xr.open_dataset(file).julday.load()
    # Extract the fractional part of the day
    fractional_part = file - file.astype(int)
    # Convert to hours
    hours = fractional_part * 24
    julday_with_hours = hours.assign_coords(hours=hours)

    return julday_with_hours
    
def create_datetime_xarray(months, days, hours, reference_time):
    """
    Converts month, day, and hour arrays into a datetime xarray.DataArray 
    mapped to a reference time array.
    
    Parameters:
    - months: Array of month values (1-12).
    - days: Array of day values (1-31).
    - hours: Array of hour values (0.0-24.0).
    - reference_time: Reference xarray.DataArray with datetime64[ns] values.
    
    Returns:
    - datetime_xr: xarray.DataArray of datetimes corresponding to the input.
    """
    # Generate a meshgrid of months, days, and hours for combinations
    months, days, hours = np.meshgrid(months, days, hours, indexing="ij")
    
    # Combine the components into a pandas datetime object
    datetimes = pd.to_datetime({
        "year": reference_time.dt.year.values[0],
        "month": months.flatten(),
        "day": days.flatten(),
        "hour": hours.flatten().astype(int),
        "minute": ((hours.flatten() % 1) * 60).astype(int)
    }, errors='coerce')  # coerce invalid dates into NaT

    # Reshape into the original dimensions
    datetimes = datetimes.values.reshape(months.shape)
    
    # Create an xarray.DataArray
    datetime_xr = xr.DataArray(
        datetimes,
        dims=("month", "day", "hour"),
        coords={"month": np.unique(months), "day": np.unique(days), "hour": np.unique(hours)}
    )

    return datetime_xr
def find_nearest(array, value): # GPT
    return np.abs(array - value).argmin()
# def find_nearest(array, value): ### MINE 
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return array[idx]

def POLDER_Ensemble(MODEL_Overpass,POLDER_variable_Data):
    reference_time = xr.DataArray(
        pd.date_range("2010-01-01", "2010-12-31", freq="3h"),
        dims="time",
        name="time"
    )
    
    combined = xr.concat(MODEL_Overpass, dim="overpass").sortby('ensemble')
    dataset0 = xr.DataArray(
    POLDER_variable_Data[i],    dims=["overpass"],
        coords={"overpass": np.arange(len(MODEL_Overpass))}  # Adjust the length to match "overpass"
    )
    # Create a new ensemble coordinate for dataset0
    new_ensemble_id = -1  # New ensemble identifier
    
    # Align dataset0 to the overpass dimension
    dataset0_aligned = xr.DataArray(
        dataset0.values,  # Values from dataset0
        dims=["overpass"],  # Use the existing overpass dimension
        coords={"overpass": combined["overpass"]}  # Align with the original overpass
    )
    
    # Add the ensemble dimension and coordinate to dataset0_aligned
    dataset0_expanded = dataset0_aligned.expand_dims("ensemble")
    dataset0_expanded["ensemble"] = [new_ensemble_id]  # Assign the new ensemble coordinate
    
    # Concatenate the updated dataset along the ensemble dimension
    updated_data = xr.concat([combined, dataset0_expanded], dim="ensemble")
    return updated_data


#Variable_of_interest = input('What is your Variable of Interest? (SSA, ANG, AOD): ')
Variable_of_interest = os.getenv('VARIABLE_OF_INTEREST')
BIO_of_interest = os.getenv('BIO_OF_INTEREST')


import glob
# Define the input directory containing the .nc files
input_dir = f"/home/ybhatti/prjs1076/Observational_Data/POLDER_1.0x1.0_basedon0.1_NPge2/2010/{BIO_of_interest}"
file_list = glob.glob(input_dir+'/*.nc')
sorted_files = sorted(file_list, key=extract_date)

# Define the output directory to save transformed files
output_dir = "/home/ybhatti/prjs1076/Observational_Data/POLDER_Processed"
os.makedirs(output_dir, exist_ok=True)
lats_POL=[]
lons_POL=[]
time_POL=[]
day_POL=[]
month_POL=[]
ANG_POL=[]
AOT_POL=[]
SSA_POL=[]

# Loop through all .nc files in the directory
for file in (sorted_files):
    # Load the dataset
    month_POL.append(xr.open_dataset(file).month.load().data)
    day_POL.append(xr.open_dataset(file).day.load().data)
    SSA_POL.append(xr.open_dataset(file).SSA.load()[:,7].data) # Take SSA when AOD > 0.2
    ANG_POL.append(xr.open_dataset(file).AE_490_670.load()[:].data)
    AOT_POL.append(xr.open_dataset(file).AOT550.load()[:].data)
    lats_POL.append(xr.open_dataset(file).lat.load().data)
    lons_POL.append(xr.open_dataset(file).assign_coords(lon=((xr.open_dataset(file).lon + 360) % 360)).lon.load().data)
    time_POL.append(convert_julday_to_hours(xr.open_dataset(file).julday).load().data)
    #end
        # # Apply the transformation: convert lat/lon to 2D
       # dataset_2d = dataset.set_index({'len': ['lat', 'lon']}).unstack('len')

for i in range(len(lats_POL)):
    # Apply condition for AOT_POL < 0.2
    SSA_POL[i][AOT_POL[i] < 0.2] = np.nan
    ANG_POL[i][AOT_POL[i] < 0.2] = np.nan

model_time=ANG_HF_PD.time.dt.dayofyear
# Assuming the 'times' and 'month_soap' are already initialized
for i in range(len(lats_POL)):
    times_recorded=[]
    OVERPASS_ANG=[]
    OVERPASS_SSA=[]
    OVERPASS_AOD=[]
    
    for la in range(len(lats_POL[i])):
        latt = find_nearest(lats, lats_POL[i][la])  # Find the nearest latitude
        lonss = find_nearest(lons, lons_POL[i][la])  # Find the nearest longitude
        OBS_TIME=create_datetime_xarray(month_POL[i][la],day_POL[i][la],time_POL[i][la],reference_time)
        time_index = find_nearest(ANG_HF_PD.time, OBS_TIME)  # Find the nearest longitude
            
        # Extract the model value (ANG_550nm_865nm) at the closest grid point
        ANG_model_value = ANG_HF_PD.isel(time=time_index, lat=latt, lon=lonss)#.values
        SSA_model_value = SSA_HF_PD.isel(time=time_index, lat=latt, lon=lonss)#.values
        AOD_model_value = AOD_HF_PD.isel(time=time_index, lat=latt, lon=lonss)#.values

        OVERPASS_ANG.append(ANG_model_value)  # Store the model value for this point
        OVERPASS_SSA.append(SSA_model_value)  # Store the model value for this point
        OVERPASS_AOD.append(AOD_model_value)  # Store the model value for this point

        # Record the corresponding time
        times_recorded.append(str(ANG_HF_PD.time.isel(time=time_index).values))

    updated_data_ANG = POLDER_Ensemble(OVERPASS_ANG,ANG_POL)
    updated_data_AOD = POLDER_Ensemble(OVERPASS_AOD,AOT_POL)
    updated_data_SSA = POLDER_Ensemble(OVERPASS_SSA,SSA_POL)

   # end
    updated_data_ANG.to_netcdf(f'/home/ybhatti/prjs1076/Processed_Data/PD/Observational_Comparison/Pre_Processed/POLDER/{BIO_of_interest}/ANG/Polder_Overpass_Day_{i+1}.nc')
    updated_data_AOD.to_netcdf(f'/home/ybhatti/prjs1076/Processed_Data/PD/Observational_Comparison/Pre_Processed/POLDER/{BIO_of_interest}/AOD/Polder_Overpass_Day_{i+1}.nc')
    updated_data_SSA.to_netcdf(f'/home/ybhatti/prjs1076/Processed_Data/PD/Observational_Comparison/Pre_Processed/POLDER/{BIO_of_interest}/SSA/Polder_Overpass_Day_{i+1}.nc')

    times_recorded=[]
    OVERPASS_ANG=[]
    OVERPASS_SSA=[]
    OVERPASS_AOD=[]
    # if i ==2:
    #     end
