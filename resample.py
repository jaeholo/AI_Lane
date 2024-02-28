import os
import numpy as np
from obspy import read
from tqdm import tqdm
from scipy.signal import resample

# sac_e = "D:/Bafang/data/airportHighway/lucas0614/sac/E/230216.180000.EB000228.EHE.sac"
# sac_n = "D:/Bafang/data/airportHighway/lucas0614/sac/N/230216.180000.EB000228.EHN.sac"
sac_z = r"F:\sac4timeshift\1122\231122.000000.EB003035.HHZ.sac"
# Read the SAC file
st = read(sac_z)
# check current sampling rate
# print(st)
#
# # Resample the SAC data to a new sampling rate (e.g., 250 Hz)
new_sampling_rate = 250
st_resampled = st.resample(new_sampling_rate)
#
# # Save the resampled data as a SAC file
output_filename = r"F:\sac4timeshift\1122\231122.000000.EB003035.HHZ250.sac"
st_resampled.write(output_filename, format="SAC")