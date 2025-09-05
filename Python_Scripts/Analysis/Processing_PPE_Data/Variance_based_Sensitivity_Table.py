import os
import numpy as np
import iris
os.chdir('/home/ybhatti/prjs1076/Emulator')
from PDFs import *
from utils import get_bc_ppe_data, normalize
import psutil
from esem import gp_model
from esem.utils import get_random_params
import seaborn
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
os.chdir('/home/ybhatti/yusufb/Scripts/')
from my_functions import *
os.chdir('/home/ybhatti/prjs1076/ESEm')
from esem import cnn_model
os.chdir('/home/ybhatti/prjs1076/Emulator/PAPER_Plotting')
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import matplotlib.colors as mcolors
import gc
from typing import Optional
from matplotlib.contour import QuadContourSet  # Importing QuadContourSet
import copy

# This script basically creates the table needed for Figure 4(?), It shows the causes of uncertainty, globally. For each variable. To run this properly, i need four variables run at different times. AOD, ERF, ERFaci, ERFari. This will create four csv files. Regional uncertainties for each specific variable.

def pdf_parameters(n,original_parameter_logged):
# Define the number of samples
#n = 30000  # Adjust as needed

    # Initialize your PDF generator
    pdf_generator = EmissionPDFs()
    
    # Define a dictionary to map PDF functions to new_samples columns
    pdf_mapping = {
        "BC_RAD_PDF": "V_SCALE_BC_RAD_NI",
        "DU_RAD_PDF": "V_SCALE_DU_RAD_NI",
        "DRYDEP_ACC_PDF": "V_SCALE_DRYDEP_ACC",
        "DRYDEP_AIT_PDF": "V_SCALE_DRYDEP_AIT",
        "DRYDEP_COA_PDF": "V_SCALE_DRYDEP_COA",
        "EMI_BB_PDF": "V_SCALE_EMI_BB",
        "EMI_BF_PDF": "V_SCALE_EMI_BF",
        "EMI_ANTH_SO2_PDF": "V_SCALE_EMI_ANTH_SO2",
        "EMI_DUST_PDF": "V_SCALE_EMI_DUST",
        "SO2_CHEM_PDF": "V_SCALE_SO2_REACTIONS",
        "EMI_SSA_PDF": "V_SCALE_EMI_SSA",
        "CMR_BB_PDF": "V_SCALE_EMI_CMR_BB",
        "CMR_BF_PDF": "V_SCALE_EMI_CMR_BF",
        "CMR_FF_PDF": "V_SCALE_EMI_CMR_FF",
        "EMI_DMS_PDF": "V_SCALE_EMI_DMS",
        "EMI_FF_PDF": "V_SCALE_EMI_FF",
        "NUC_FT_PDF": "V_SCALE_NUC_FT",
        "SO4_COATING_PDF": "V_SCALE_SO4_COATING",
        "CLOUD_PH_PDF": "V_SCALE_PH_PERT",
        "WETDEP_BC_PDF": "V_SCALE_WETDEP_BC",
        "WETDEP_IC_PDF": "V_SCALE_WETDEP_IC",
        "KAPPA_SS_PDF": "V_SCALE_KAPPA_SS",
        "KAPPA_SO4_PDF": "V_SCALE_KAPPA_SO4",
        "CDNC_MIN_PDF": "V_SCALE_CDNC_MIN",
      #  "VERTICAL_VELOCITY_PDF": "V_SCALE_VERTICAL_VELOCITY",
    }
    
    # Initialize the new_samples DataFrame with correct columns and number of rows
    new_samples = pd.DataFrame(index=range(n), columns=pdf_mapping.values())
    
    # Populate new_samples with samples from each PDF function
    for pdf_func, col_name in pdf_mapping.items():
        # Get the PDF function from the pdf_generator class
        func = getattr(pdf_generator, pdf_func, None)
        if callable(func):
            # Generate samples using the PDF function
            _, _, sam = func(n=n)
            # Assign samples to the appropriate column in new_samples
            new_samples[col_name] = sam
            
    #new_samples.set_index(new_samples.columns[0], inplace=True)
    
    # Create the new row with all columns set to 1, and update specific values
    new_row = {col: 1 for col in new_samples.columns}
    new_row.update({
        'V_SCALE_BC_RAD_NI': 0.71,
        'V_SCALE_DU_RAD_NI': 0.001,
        'V_SCALE_PH_PERT': 5.6020599913279625,
        'V_SCALE_CDNC_MIN': 40,
        'V_SCALE_KAPPA_SO4': 0.6,
        'V_SCALE_EMI_CMR_BB': 75,
        'V_SCALE_EMI_CMR_BF': 30,
        'V_SCALE_EMI_CMR_FF': 30
    })
    
    # Create a DataFrame for the new row
    new_row_df = pd.DataFrame([new_row], index=['Control'])
    
    # Concatenate the new row DataFrame with the original DataFrame
    ppe_params_test = pd.concat([new_row_df, new_samples])
    ppe_params_test['V_SCALE_EMI_CMR_BB'] = ppe_params_test['V_SCALE_EMI_CMR_BB'] * 0.013333333333333334
    ppe_params_test['V_SCALE_EMI_CMR_BF'] = ppe_params_test['V_SCALE_EMI_CMR_BF'] * 0.03333333333333333
    ppe_params_test['V_SCALE_EMI_CMR_FF'] = ppe_params_test['V_SCALE_EMI_CMR_FF'] * 0.03333333333333333
    ppe_params_test['V_SCALE_CDNC_MIN'] = 40

    ppe_paramater_test = np.log(ppe_params_test)#.iloc[1:])

    param_pert_test = (ppe_paramater_test - original_parameter_logged.min())/(original_parameter_logged.max() - original_parameter_logged.min())
    #param_pert_test = ppe_paramater_test.apply(normalize, axis=0)
    return param_pert_test,ppe_params_test

def Interpolate_Regional_uncertainty(Predicted,mask_number,Regions_nc):
    mask = Regions_nc == mask_number

    regional_array = Regions_nc.where(mask == True,drop=True)
    # Interpolate Predicted_Control to match the lat/lon grid of reg2
    Predicted_Control_interpolated = Predicted.interp(lat=regional_array.lat, lon=regional_array.lon, method="nearest")
    # Create a mask for values in reg2 equal to 5
    masks = regional_array == mask_number
    # Initialize a new array filled with NaNs, same shape as reg2
    new_array = xr.full_like(regional_array, np.nan, dtype=float)
    # Replace the '5' values in regional array with corresponding values from Predicted_Control_interpolated
    new_array = new_array.where(~masks, Predicted_Control_interpolated)
    return new_array


lats=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PI/Processed/AOD_PPE.nc').lat
lons=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PI/Processed/AOD_PPE.nc').lon


AOD_PI=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PI/Processed/AOD_PPE.nc').TAU_2D_550nm.groupby('time.month').mean()#[0]
AOD_PD=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PD/Processed/AOD_PPE.nc').TAU_2D_550nm.groupby('time.month').mean()#[0]
AOD = AOD_PD - AOD_PI
AOD_PD=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PD/Processed/AOD_PPE.nc').TAU_2D_550nm.mean('time')
top_net_SW_PI=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PI/Processed/SW_All_Sky_Flux_PPE.nc').srad0.groupby('time.month').mean()#[0]# - 0.28734403
top_net_SW_PD=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PD/Processed/SW_All_Sky_Flux_PPE.nc').srad0.groupby('time.month').mean()#[0] 
top_net_SW_Clear_Sky_PI=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PI/Processed/SW_Clear_Sky_Flux_PPE.nc').sraf0.groupby('time.month').mean()#[0]
top_net_SW_Clear_Sky_PD=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PD/Processed/SW_Clear_Sky_Flux_PPE.nc').sraf0.groupby('time.month').mean()#[0] 
top_net_LW_PI=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PI/Processed/LW_All_Sky_Flux_PPE.nc').trad0.groupby('time.month').mean()#[0]
top_net_LW_PD=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PD/Processed/LW_All_Sky_Flux_PPE.nc').trad0.groupby('time.month').mean()#[0]
top_net_LW_Clear_Sky_PI=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PI/Processed/LW_Clear_Sky_Flux_PPE.nc').traf0.groupby('time.month').mean()#[0]
top_net_LW_Clear_Sky_PD=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PD/Processed/LW_Clear_Sky_Flux_PPE.nc').traf0.groupby('time.month').mean()#[0]
CF_PI=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PI/Processed/Cloud_Cover_PPE.nc').aclcov.groupby('time.month').mean()#[0]
CF_PD=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PD/Processed/Cloud_Cover_PPE.nc').aclcov.groupby('time.month').mean()#[0]
SSA_PD=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PD/Processed/SSAlbedo_PPE.nc').__xarray_dataarray_variable__.mean('month')
ANG_PD=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/PD/Processed/ANG_550_865_PPE.nc').ANG_550nm_865nm.groupby('time.month').mean().mean('month')

#SSA_PD=SSA_PD.mean('month')
RF_CS_PD = top_net_LW_Clear_Sky_PD + top_net_SW_Clear_Sky_PD
RF_CS_PI = top_net_LW_Clear_Sky_PI + top_net_SW_Clear_Sky_PI
ERF_ARI= RF_CS_PD - RF_CS_PI
ACI_PD = top_net_SW_PD - top_net_SW_Clear_Sky_PD
ACI_PI = top_net_SW_PI - top_net_SW_Clear_Sky_PI
ERF_ACI = ACI_PD - ACI_PI
ERF_SW = top_net_SW_PD - top_net_SW_PI
ERF_LW = top_net_LW_PD - top_net_LW_PI 

ERF = (ERF_SW + ERF_LW)
ppe_aod_data = (AOD_PD - AOD_PI)  # Use .data to compute


ppe_param_extended = pd.read_csv('~/yusufb/Branches/PPE_Scripts/parameter_values_data/Extended_PPE_Values.csv')[:]
ppe_param_extended['V_SCALE_PH_PERT'].values[211:] = np.log10(ppe_param_extended['V_SCALE_PH_PERT'].values[211:]) * -1
ppe_param_extended = ppe_param_extended.drop(columns=["V_SCALE_VERTICAL_VELOCITY"])
ppe_paramater = np.log(ppe_param_extended)#.iloc[1:])
ppe_para = ppe_paramater.apply(normalize, axis=0)


ppe_rf = xr.DataArray(ERF).mean('month')  # Convert back to xarray after computation
ppe_rf = xr.DataArray(
    ppe_rf.data,  # Use .data to extract the values
    dims=["ensemble", "lat", "lon"],  # Set the dimension names to match RF_PI
    coords={"ensemble": AOD_PD.coords["ensemble"],
            "lat": AOD_PD.coords["lat"],
            "lon": AOD_PD.coords["lon"]},  # Set coordinates from RF_PI
        name="emulator"  # Optionally, assign a name
)

ppe_erf_aci = xr.DataArray(ERF_ACI).mean('month')  # Convert back to xarray after computation
ppe_erf_aci = xr.DataArray(
    ppe_erf_aci.data,  # Use .data to extract the values
    dims=["ensemble", "lat", "lon"],  # Set the dimension names to ma12 currentlytch RF_PI
    coords={"ensemble": ERF_ACI.coords["ensemble"],
            "lat": ERF_ACI.coords["lat"],
            "lon": ERF_ACI.coords["lon"]},  # Set coordinates from RF_PI
    name=""  # Optionally, assign a name
)

ppe_erf_ari = xr.DataArray(ERF_ARI).mean('month')  # Convert back to xarray after computation
ppe_erf_ari = xr.DataArray(
    ppe_erf_ari.data,  # Use .data to extract the values
    dims=["ensemble", "lat", "lon"],  # Set the dimension names to match RF_PI
    coords={"ensemble": AOD_PI.coords["ensemble"],
            "lat": AOD_PI.coords["lat"],
            "lon": AOD_PI.coords["lon"]},  # Set coordinates from RF_PI
    name=""  # Optionally, assign a name
)


regions_nn=xr.open_dataset('/home/ybhatti/prjs1076/Processed_Data/reg_18_regridded_nn.nc').reg
for i in range(1,19):
    reg1s = np.where(regions_nn==i,regions_nn,np.nan)
    reg1 = regions_nn.where(regions_nn == i, drop=True)
    laty = reg1.where(regions_nn == i, drop=True).lat



Region_Criteria = ["NA",
    "Arctic_Ocean", "Arctic_Land", "North_Pacific_Ocean", "North_America", 
    "North_Atlantic_Ocean", "Europe", "Asia", "Tropical_Pacific_Ocean", 
    "Tropical_Atlantic_Ocean", "Africa", "Tropical_Indian_Ocean", 
    "South_Pacific_Ocean", "South_America", "South_Atlantic_Ocean", 
    "South_Indian_Ocean", "Australia", "Antarctic_Ocean", "Antarctica"
]


variable = 'AOD_PD'
#ppe_var=copy.deepcopy(ANG_PD.mean('month'))
ppe_var=copy.deepcopy(AOD_PD)
#n = 10


n_total = len(ppe_para)
n_test = 66  # Number of test samples
print(variable)
random_indices = np.random.permutation(n_total)
test_indices = random_indices[:n_test]
train_indices = random_indices[n_test:]
X_test, X_train = ppe_para.iloc[test_indices], ppe_para.iloc[train_indices]
Y_test, Y_train = ppe_var.isel(ensemble=test_indices), ppe_var.isel(ensemble=train_indices)
#kernal=['Linear','Matern52'] # for SSA, ERF
kernal=['Linear'] # For AOD and ANG, CF_PD_PI, AOD_PD_PI

gp_model_ = gp_model(X_train, Y_train, kernel=kernal)
gp_model_.train()

n = 100000
norm_para,para = pdf_parameters(n,ppe_paramater)
PPE_Prediction, _ = gp_model_.predict(norm_para.values)

# Initialize an empty list to hold the results
results = []
#ERF_mean = areaweight(Predicted_Control[0],lats).mean()
for r in range(1,19): 
    print(f"{Region_Criteria[r]}")
    ERF_Uncertainty_regional = Interpolate_Regional_uncertainty(PPE_Prediction,r,regions_nn)
    ERF_Uncertainty_regional = ERF_Uncertainty_regional.transpose("sample", "lat", "lon")

    PPE_std_regional = areaweight(ERF_Uncertainty_regional,ERF_Uncertainty_regional.lat)
    #ERF_Uncertainty.interp(lat=regional_array.lat, lon=regional_array.lon, method="nearest")

    for col,i in zip(para.columns,range(0,len(para.columns))):
        #norm_para,para = pdf_parameters(n,ppe_paramater)
        norm_param = parameter_testing(norm_para,col)
        gp_prediction, _ = gp_model_.predict(norm_param.values)
        region  = Interpolate_Regional_uncertainty(gp_prediction,r,regions_nn)
        region = region.transpose("sample", "lat", "lon")

        gpmeans = areaweight(region,region.lat)
        print(col)
        gp_uncert = gpmeans.std()
        gp_mean = gpmeans.mean()
                
        # Append the results as a dictionary
        results.append({
            'Region': Region_Criteria[r],
            'Variable': col,
            f"std": gp_uncert.data,
            '5%': np.percentile(gpmeans,5),
         #   '%Variance %': ((np.percentile(gpmeans,95) - np.percentile(gpmeans,5)) / (np.percentile(PPE_std_regional,95) - np.percentile(PPE_std_regional,5)) *100),
            # '%range': (np.percentile(gpmeans,95) - np.percentile(gpmeans,5))*100,
            #'Variance %': (gp_uncert.data / PPE_std_regional.data)*100,
            '95%': np.percentile(gpmeans,95),
            f"mean": gp_mean.data,

    #        'Diff': difference
        })
    #results_df.to_csv(f"/home/ybhatti/prjs1076/Processed_Data/Analysed_Data/Raw/Table_of_Regional_Uncertainty_{Region_Criteria[r]}.csv", index=False)
    print('-------------')



# Convert the accumulated results into a DataFrame after the loop
results_df = pd.DataFrame(results)

# Calculate Variance % for each region
for r in range(1, 19):
    region_rows = results_df['Region'] == Region_Criteria[r]
    # Contributions to the regional variance (STD ^ 2)
#    co = ((results_df.loc[region_rows, 'std'])**2 / (results_df.loc[region_rows, 'std']**2).sum() * 100)
    co = (results_df.loc[region_rows, 'std'] / results_df.loc[region_rows, 'std'].sum() * 100)
    results_df.loc[region_rows, f"Uncertainty_{variable}"] = co

# Optional: Save or inspect the dataframe
#print(results_df)
results_df.to_csv(f"/home/ybhatti/prjs1076/Processed_Data/Analysed_Data/CDNC_40/Table_of_Regional_Uncertainty_{variable}_{n}.csv", index=False)




variable = 'ANG_PD'
#ppe_var=copy.deepcopy(ANG_PD.mean('month'))
ppe_var=copy.deepcopy(ANG_PD)
#n = 10


n_total = len(ppe_para)
n_test = 66  # Number of test samples
print(variable)
random_indices = np.random.permutation(n_total)
test_indices = random_indices[:n_test]
train_indices = random_indices[n_test:]
X_test, X_train = ppe_para.iloc[test_indices], ppe_para.iloc[train_indices]
Y_test, Y_train = ppe_var.isel(ensemble=test_indices), ppe_var.isel(ensemble=train_indices)
#kernal=['Linear','Matern52'] # for SSA, ERF
kernal=['Linear'] # For AOD and ANG, CF_PD_PI, AOD_PD_PI

gp_model_ = gp_model(X_train, Y_train, kernel=kernal)
gp_model_.train()

n = 100000
norm_para,para = pdf_parameters(n,ppe_paramater)
PPE_Prediction, _ = gp_model_.predict(norm_para.values)

# Initialize an empty list to hold the results
results = []
#ERF_mean = areaweight(Predicted_Control[0],lats).mean()
for r in range(1,19): 
    print(f"{Region_Criteria[r]}")
    ERF_Uncertainty_regional = Interpolate_Regional_uncertainty(PPE_Prediction,r,regions_nn)
    ERF_Uncertainty_regional = ERF_Uncertainty_regional.transpose("sample", "lat", "lon")

    PPE_std_regional = areaweight(ERF_Uncertainty_regional,ERF_Uncertainty_regional.lat)
    #ERF_Uncertainty.interp(lat=regional_array.lat, lon=regional_array.lon, method="nearest")

    for col,i in zip(para.columns,range(0,len(para.columns))):
        #norm_para,para = pdf_parameters(n,ppe_paramater)
        norm_param = parameter_testing(norm_para,col)
        gp_prediction, _ = gp_model_.predict(norm_param.values)
        region  = Interpolate_Regional_uncertainty(gp_prediction,r,regions_nn)
        region = region.transpose("sample", "lat", "lon")

        gpmeans = areaweight(region,region.lat)
        print(col)
        gp_uncert = gpmeans.std()
        gp_mean = gpmeans.mean()
                
        # Append the results as a dictionary
        results.append({
            'Region': Region_Criteria[r],
            'Variable': col,
            f"std": gp_uncert.data,
            '5%': np.percentile(gpmeans,5),
         #   '%Variance %': ((np.percentile(gpmeans,95) - np.percentile(gpmeans,5)) / (np.percentile(PPE_std_regional,95) - np.percentile(PPE_std_regional,5)) *100),
            # '%range': (np.percentile(gpmeans,95) - np.percentile(gpmeans,5))*100,
            #'Variance %': (gp_uncert.data / PPE_std_regional.data)*100,
            '95%': np.percentile(gpmeans,95),
            f"mean": gp_mean.data,

    #        'Diff': difference
        })
    #results_df.to_csv(f"/home/ybhatti/prjs1076/Processed_Data/Analysed_Data/Raw/Table_of_Regional_Uncertainty_{Region_Criteria[r]}.csv", index=False)
    print('-------------')



# Convert the accumulated results into a DataFrame after the loop
results_df = pd.DataFrame(results)

# Calculate Variance % for each region
for r in range(1, 19):
    region_rows = results_df['Region'] == Region_Criteria[r]
    # Contributions to the regional variance (STD ^ 2)
#    co = ((results_df.loc[region_rows, 'std'])**2 / (results_df.loc[region_rows, 'std']**2).sum() * 100)
    co = (results_df.loc[region_rows, 'std'] / results_df.loc[region_rows, 'std'].sum() * 100)
    results_df.loc[region_rows, f"Uncertainty_{variable}"] = co

# Optional: Save or inspect the dataframe
#print(results_df)
results_df.to_csv(f"/home/ybhatti/prjs1076/Processed_Data/Analysed_Data/CDNC_40/Table_of_Regional_Uncertainty_{variable}_{n}.csv", index=False)



variable = 'SSA_PD'
#ppe_var=copy.deepcopy(ANG_PD.mean('month'))
ppe_var=copy.deepcopy(SSA_PD)
#n = 10


n_total = len(ppe_para)
n_test = 66  # Number of test samples
print(variable)
random_indices = np.random.permutation(n_total)
test_indices = random_indices[:n_test]
train_indices = random_indices[n_test:]
X_test, X_train = ppe_para.iloc[test_indices], ppe_para.iloc[train_indices]
Y_test, Y_train = ppe_var.isel(ensemble=test_indices), ppe_var.isel(ensemble=train_indices)
kernal=['Linear','Matern52'] # for SSA, ERF
#kernal=['Linear'] # For AOD and ANG, CF_PD_PI, AOD_PD_PI

gp_model_ = gp_model(X_train, Y_train, kernel=kernal)
gp_model_.train()

n = 100000
norm_para,para = pdf_parameters(n,ppe_paramater)
PPE_Prediction, _ = gp_model_.predict(norm_para.values)

# Initialize an empty list to hold the results
results = []
#ERF_mean = areaweight(Predicted_Control[0],lats).mean()
for r in range(1,19): 
    print(f"{Region_Criteria[r]}")
    ERF_Uncertainty_regional = Interpolate_Regional_uncertainty(PPE_Prediction,r,regions_nn)
    ERF_Uncertainty_regional = ERF_Uncertainty_regional.transpose("sample", "lat", "lon")

    PPE_std_regional = areaweight(ERF_Uncertainty_regional,ERF_Uncertainty_regional.lat)
    #ERF_Uncertainty.interp(lat=regional_array.lat, lon=regional_array.lon, method="nearest")

    for col,i in zip(para.columns,range(0,len(para.columns))):
        #norm_para,para = pdf_parameters(n,ppe_paramater)
        norm_param = parameter_testing(norm_para,col)
        gp_prediction, _ = gp_model_.predict(norm_param.values)
        region  = Interpolate_Regional_uncertainty(gp_prediction,r,regions_nn)
        region = region.transpose("sample", "lat", "lon")

        gpmeans = areaweight(region,region.lat)
        print(col)
        gp_uncert = gpmeans.std()
        gp_mean = gpmeans.mean()
                
        # Append the results as a dictionary
        results.append({
            'Region': Region_Criteria[r],
            'Variable': col,
            f"std": gp_uncert.data,
            '5%': np.percentile(gpmeans,5),
         #   '%Variance %': ((np.percentile(gpmeans,95) - np.percentile(gpmeans,5)) / (np.percentile(PPE_std_regional,95) - np.percentile(PPE_std_regional,5)) *100),
            # '%range': (np.percentile(gpmeans,95) - np.percentile(gpmeans,5))*100,
            #'Variance %': (gp_uncert.data / PPE_std_regional.data)*100,
            '95%': np.percentile(gpmeans,95),
            f"mean": gp_mean.data,

    #        'Diff': difference
        })
    #results_df.to_csv(f"/home/ybhatti/prjs1076/Processed_Data/Analysed_Data/Raw/Table_of_Regional_Uncertainty_{Region_Criteria[r]}.csv", index=False)
    print('-------------')



# Convert the accumulated results into a DataFrame after the loop
results_df = pd.DataFrame(results)

# Calculate Variance % for each region
for r in range(1, 19):
    region_rows = results_df['Region'] == Region_Criteria[r]
    # Contributions to the regional variance (STD ^ 2)
#    co = ((results_df.loc[region_rows, 'std'])**2 / (results_df.loc[region_rows, 'std']**2).sum() * 100)
    co = (results_df.loc[region_rows, 'std'] / results_df.loc[region_rows, 'std'].sum() * 100)
    results_df.loc[region_rows, f"Uncertainty_{variable}"] = co

# Optional: Save or inspect the dataframe
#print(results_df)
results_df.to_csv(f"/home/ybhatti/prjs1076/Processed_Data/Analysed_Data/CDNC_40/Table_of_Regional_Uncertainty_{variable}_{n}.csv", index=False)



#### This will merge the tables and give a total regional uncertainty csv to save. It needs to be done after all ### This will merge the tables and give a total regional uncertainty csv to save.
# table_erf = pd.read_csv('/home/ybhatti/prjs1076/Processed_Data/Analysed_Data/Raw/Table_of_Regional_Uncertainty_ERF_100000.csv')
# table_aod = pd.read_csv('/home/ybhatti/prjs1076/Processed_Data/Analysed_Data/Raw/Table_of_Regional_Uncertainty_AOD_100000.csv')
# table_aci = pd.read_csv('/home/ybhatti/prjs1076/Processed_Data/Analysed_Data/Raw/Table_of_Regional_Uncertainty_ERF_ACI_100000.csv')
# table_ari = pd.read_csv('/home/ybhatti/prjs1076/Processed_Data/Analysed_Data/Raw/Table_of_Regional_Uncertainty_ERF_ARI_100000.csv')


# table = pd.merge(table_erf, table_aod, on=["Region", "Variable"], how="left", suffixes=("_erf", "_aod"))
# table = pd.merge(table, table_aci, on=["Region", "Variable"], how="left", suffixes=("", "_aci"))
# table = pd.merge(table, table_ari, on=["Region", "Variable"], how="left", suffixes=("", "_ari"))
# table["Variable"] = table["Variable"].str.replace("V_SCALE_", "", regex=False)

# table_erf = pd.read_csv('/home/ybhatti/prjs1076/Processed_Data/Analysed_Data/Raw/Table_of_Global_Uncertainty_250000_ERF.csv')
# table_aci = pd.read_csv('/home/ybhatti/prjs1076/Processed_Data/Analysed_Data/Raw/Table_of_Global_Uncertainty_200000_ERF_ACI.csv')
# table_ari = pd.read_csv('/home/ybhatti/prjs1076/Processed_Data/Analysed_Data/Raw/Table_of_Global_Uncertainty_200000_ERF_ARI.csv')
# table_aod = pd.read_csv('/home/ybhatti/prjs1076/Processed_Data/Analysed_Data/Raw/Table_of_Global_Uncertainty_200000_AOD.csv')
# merged_table = pd.merge(table_erf, table_aod, on=["Variable"], how="left", suffixes=("_erf", "_aod"))
# merged_table = pd.merge(merged_table, table_aci, on=["Variable"], how="left", suffixes=("", "_aci"))
# merged_table = pd.merge(merged_table, table_ari, on=["Variable"], how="left", suffixes=("", "_ari"))
# merged_table["Variable"] = merged_table["Variable"].str.replace("V_SCALE_", "", regex=False)
# merged_table.insert(0, 'Region', 'Global')

# table_columns = set(table.columns)
# merged_table_columns = set(merged_table.columns)

# # Convert the set of common columns to a list
# common_columns_list = list(common_columns)

# # Now use the list to filter the DataFrames
# table_common = table[common_columns_list]
# merged_table_common = merged_table[common_columns_list]


# table = pd.concat([merged_table_common, table_common], ignore_index=True)


# table.to_csv("/home/ybhatti/prjs1076/Processed_Data/Analysed_Data/Raw/Table_of_Regional_Uncertainty_Variance_Total.csv")




