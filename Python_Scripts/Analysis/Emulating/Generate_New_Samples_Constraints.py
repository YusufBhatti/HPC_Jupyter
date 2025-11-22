import os
import numpy as np
from scipy.stats import stats
os.putenv("HDF5_DISABLE_VERSION_CHECK", '1')

from my_functions import *
import pandas as pd
import xarray as xr
import gc
from typing import Optional

print('imported complete')

n_samples = int(os.getenv('NUMBER_OF_SAMPLES'))
base_dir = os.getenv('BASE_DIR')


# AAOD = (1 - SSA) * AOD

ppe_normalized = pd.read_csv('/home/ybhatti2/prjs1474/Datasets/Normalized_PPE_Parameters.csv')
ppe_param = pd.read_csv('/home/ybhatti2/prjs1474/Datasets/PPE_Parameters.csv')
ppe_normalized.set_index(ppe_normalized.columns[0], inplace=True)
ppe_param.set_index(ppe_param.columns[0], inplace=True)


ppe_dist = copy.deepcopy(ppe_param)
# extract control row
control_row = ppe_dist.loc[["PPE_Control"]]
# number of new samples


print(f"number of samples = {n_samples}")

print("Generating new samples...")

# sample each column with replacement from its empirical distribution
sampled_cols = {
    col: np.random.choice(ppe_dist[col].values, size=n_samples, replace=True)
    for col in ppe_dist.columns
}

new_samples_empirical = pd.DataFrame(sampled_cols)

# concat control row on top
ppe_non_normalized_new_samples = pd.concat([control_row, new_samples_empirical], ignore_index=True)


# Check environment variable from batch.sh



ppe_normalized_new_samples = (
    (ppe_non_normalized_new_samples - ppe_non_normalized_new_samples.min()) /
    (ppe_non_normalized_new_samples.max() - ppe_non_normalized_new_samples.min())
)

uninteresting_params = ['SCALE_EMI_SHP_SO2', 'SCALE_EMI_SS_COARSE', 'DU_RAD_NI', 'SCALE_WETDEP_BC', 'SCALE_DRYDEP_ACC', 'KAPPA_OC', 'SCALE_NUC_FT']
print("Uninteresting parameters:", uninteresting_params)

# Fix constant columns
for param in uninteresting_params:
    print(f"Fixing {param} to Default")
    ppe_non_normalized_new_samples[param] = ppe_non_normalized_new_samples[param].iloc[0]
    ppe_normalized_new_samples[param] = ppe_normalized_new_samples[param].iloc[0]

# print("Fixing OC_RAD_NI to Default")

# ppe_non_normalized_new_samples["OC_RAD_NI"] = ppe_non_normalized_new_samples["OC_RAD_NI"].iloc[0]
# ppe_normalized_new_samples["OC_RAD_NI"] = ppe_normalized_new_samples["OC_RAD_NI"].iloc[0]

# print("Fixing SCALE_SEASALT_EXPO to Default")

# ppe_non_normalized_new_samples["SCALE_SEASALT_EXPO"] = ppe_non_normalized_new_samples["SCALE_SEASALT_EXPO"].iloc[0]
# ppe_normalized_new_samples["SCALE_SEASALT_EXPO"] = ppe_normalized_new_samples["SCALE_SEASALT_EXPO"].iloc[0]

# Save results
ppe_normalized_new_samples.to_csv(f"{base_dir}/ppe_params_{n_samples}_Important_Parameters.csv")
ppe_non_normalized_new_samples.to_csv(f"{base_dir}/ppe_params_non_norm_{n_samples}_Important_Parameters.csv")
print("✅ Saved new samples.")


