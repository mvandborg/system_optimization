
# %%

import numpy as np
from scipy.io import loadmat


dir = r'C:\Users\madshv\data_phd\botdr_erbium_data'
file_id_vec = ['17','17_5','18','18_5','19','19_5','20']
dat = []

for fid in file_id_vec:
    fname = dir+'\data_att'+fid+'.mat'
    dat.append(loadmat(fname))



# %%
