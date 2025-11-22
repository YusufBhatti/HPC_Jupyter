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

models = xr.open_dataset(f"{base_dir_regional}/{var_name}/Regional_Uncertainty_{var_name.lower()}_{n_samples}.nc")
varname = list(models.data_vars.keys())[0]
model = models[varname]
model = model.transpose("sample", "month", "region")
ppe_var = copy.deepcopy(model[:-1]).fillna(0)
model_da = model[-1].expand_dims(sample=[-1])

print(f"Part 4 - Emulate the regional means for {var_name}")

if var_name == "AOD":
    kernal=['Bias','Polynomial'] # ANG , SSA, AI, M2

elif var_name == "AI":
    kernal=['Polynomial'] # ANG , SSA, AI, M2

elif var_name == "ANG":
    kernal=['Bias','Matern52'] # ANG , SSA, AI, M2

elif var_name == "SSA":
    kernal=['Bias','Matern52'] # ANG , SSA, AI, M2

elif var_name == "AAOD":
    kernal=['Polynomial']  # AAOD

elif var_name == "AOD_Mode_1":
    kernal=['Matern52','Polynomial'] # ANG , SSA, AI, M2
    
elif var_name == "AOD_Mode_2":
    # model = model1 + model2
    kernal=['Matern52','Polynomial'] # ANG , SSA, AI, M2
    
elif var_name == "AOD_Mode_3":
    kernal=['Matern52'] # for  ERF, AOD_M_1, AOD_M_3
    
elif var_name == "AOD_Mode_Coarse":
    kernal=['Polynomial'] # ANG , SSA, AI, M2
    
elif var_name == "ERF":
    kernal=['Matern52'] # ANG , SSA, AI, M2

elif var_name == "ERFaci":
    kernal=['Linear','Matern52'] # for  ERF, AOD_M_1, AOD_M_3
    kernal=['Matern52'] # ANG , SSA, AI, M2

elif var_name == "ERFari":
    kernal=['Linear','Matern52'] # for  ERF, AOD_M_1, AOD_M_3
    kernal=['Matern52'] # ANG , SSA, AI, M2
elif var_name == "CDNC" or var_name == "CDNC_Filtered" or var_name == "REFFL_CT" :
    kernal=['Bias', 'Matern52']  #  CDNC

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


ppe_normalized_new_samples = pd.read_csv(f'{base_dir}/ppe_params_{n_samples_new}_Important_Parameters.csv')
ppe_normalized_new_samples.set_index(ppe_normalized_new_samples.columns[0], inplace=True)
print('Read new samples ')


def parameter_testing(table,PARAMETER):
    para_table=copy.deepcopy(table)
    # Assuming 'Control' row is the first row (index 0)
    control_row = para_table.iloc[0]
    
    # Update all rows to match the Control row, except for Variable
    for column in table.columns:
        if column != PARAMETER:
            para_table[column] = control_row[column]
    return para_table
    norm_param = parameter_testing(norm_para,'')



emulated, var = gp_model.predict(ppe_normalized_new_samples.values)

# 
print(f'For {var_name} we will now obtain the contribution to global annual mean uncertainty')
# Initialize an empty list to hold the results
results = []
for col,i in zip(ppe_normalized_new_samples.columns,range(0,len(ppe_normalized_new_samples.columns))):
    print(col)
    norm_param = parameter_testing(ppe_normalized_new_samples,col)
    gp_prediction, _ = gp_model.predict(norm_param.values)
    gpmeans = gp_prediction.sel(region='Global').mean('month')
    gp_uncert = gpmeans.std()
    # Append the results as a dictionary
    results.append({
        'Variable': col,
        f"std": gp_uncert.data,
    })
# Convert the accumulated results into a DataFrame after the loop
results_df = pd.DataFrame(results)
results_df[f"Uncertainty_{var_name}"] = (results_df['std'] / results_df['std'].sum()) * 100
results_df.to_csv(f'{base_dir_regional}/{var_name}/Contribution_of_{var_name.lower()}_{n_samples_new}_Important_Parameters.csv')


if var_name == 'ERF' or var_name == 'ERFaci' or var_name == 'ERFari':
    pass
else:
    emulated = xr.concat([emulated, model_da], dim='sample')


print('Saving emulated Variables')
emulated.to_netcdf(f'{base_dir_regional}/{var_name}/emulated_{var_name.lower()}_{n_samples_new}_Important_Parameters.nc')

print('Completed Part 4 ')
