import os
import datetime
import numpy
import obspy
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt
from multiprocessing import Pool
import numpy as np


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Define the function to apply the filter
def butter_bandpass_lfilter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


sac_n = r"F:\sac4timeshift\1122\231122.000000.EB003035.HHN250.sac"
sac_z = r"F:\sac4timeshift\1122\231122.000000.EB003035.HHZ250.sac"
skipped = open("noskiptry.txt", 'w')


def sac2binary(npysubdir, savedir):
    min_freq = 1
    max_freq = 50
    st_n = obspy.read(sac_n)#.filter('bandpass', freqmin=min_freq, freqmax=max_freq, corners=1, zerophase=True)
    st_z = obspy.read(sac_z)#.filter('bandpass', freqmin=min_freq, freqmax=max_freq, corners=1, zerophase=True)
    files = sorted(os.listdir(npysubdir))
    os.makedirs(savedir, exist_ok=True)


    for f in tqdm(files):
        time_stamp = f.split("_")[0]
        # npts = 4096

        start = obspy.UTCDateTime(datetime.datetime.utcfromtimestamp((float(time_stamp) - 2048) / 1000))
        end = obspy.UTCDateTime(datetime.datetime.utcfromtimestamp((float(time_stamp) + 2100) / 1000))

        st_n_slice = st_n.slice(starttime=start, endtime=end)[0].data
        st_z_slice = st_z.slice(starttime=start, endtime=end)[0].data

        st_n_slice = np.resize(st_n_slice, 1024)
        st_z_slice = np.resize(st_z_slice, 1024)

        label = numpy.load(os.path.join(npysubdir, f)).flatten()
        savepath = os.path.join(savedir, f.split(".")[0] + ".npy")

        # if (len(st_z_slice) == 1024 and len(label) == 4096):
        if (len(st_z_slice) == 1024):
            trace_data = numpy.concatenate((st_n_slice,st_z_slice,label))
            numpy.save(savepath, trace_data)
        else:
            # print(str(savepath))
            skipped.write(str(savepath) + "\n")


def video2array_directory(labeldir, basedir):
    # npydir = os.path.join(path_dir, "35/res_bird64")
    # basedir = os.path.join(path_dir, "35/250nz64")
    for i, d in enumerate(sorted(os.listdir(npydir),  key=lambda x: int(x))):
        full_path = os.path.join(npydir, d)
        if os.path.isdir(full_path):
            savedir = os.path.join(basedir, str(i))
            sac2binary(full_path, savedir)


# def viewSac():
#     st = obspy.read(sac_z)
#     # st.plot()
#     # plt.show()
#
#     time_stamp = 1676603460000
#     ts2 = 1676603651000
#     start = obspy.UTCDateTime(datetime.datetime.utcfromtimestamp((float(time_stamp)) / 1000))
#     end = obspy.UTCDateTime(datetime.datetime.utcfromtimestamp((float(ts2)) / 1000))
#     check = st.slice(starttime=start, endtime=end)
#     check.plot()
#     plt.show()

#input is the directory with the generated labels
#ie:video2array_directory(path)
#path contains subdir bird and simple
video2array_directory("F:/35/res_bird64", "F:/35/250nz64")
# exit()

# if __name__ == '__main__':
#     paths = [
#              'D:/Bafang/data/hope/17',
#              'D:/Bafang/data/hope/18',
#              'D:/Bafang/data/hope/20',
#              # 'D:/Bafang/data/hopen/17',
#              ]
#
#     with Pool(4) as p:
#         p.map(video2array_directory, paths)
#     # video2array_directory("D:/Bafang/data/hope/17/bird", "D:/Bafang/data/hope/17/enz_bp")
#     # video2array_directory("D:/Bafang/data/hope/18/bird", "D:/Bafang/data/hope/18/enz_bp")
#     # video2array_directory("D:/Bafang/data/hope/19/bird", "D:/Bafang/data/hope/19/enz_bp")
#     # video2array_directory("D:/Bafang/data/hope/20/bird", "D:/Bafang/data/hope/20/enz_bp")
#
#     skipped.close()