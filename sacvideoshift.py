import os

os.add_dll_directory("C:\\Program Files (x86)\\VTK\\bin")  # not sure why interpreter is not finding this

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import obspy
import datetime
from PIL import Image
from scipy.signal import butter, filtfilt, lfilter


'''
Here in my task, sample rate is 250HZ, which means 250 data points per second
time 
'''


# the symmetry makes only half of the data useful
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def findOffset(audio_name, sac_name, bang, offset):
    fs = 48000
    lowfreq = 1000 * 12    # have to adjust
    highfreq = 1000 * 16
    ymax = 1000 * 1
    # audio = np.load("D:/Bafang/video/audio/17/0/1676603660000_48000_0.npy")
    # sac_file = obspy.read("D:/Bafang/video/sac/17/230217.000000.EB000228.EHZ.sac")
    audio = np.load(audio_name)   # npy file
    sac_file = obspy.read(sac_name)
    time = np.linspace(0, 4, int(fs * 4))
    stime = np.linspace(0, 4, int(1000 * 4))
    # t = 12
    t = bang

    s = t - 2  # start
    e = t + 2  # end
    # os = -9.5
    ofs = offset
    offset = 1000 * ofs   # s 2 ms
    fn = os.path.basename(audio_name)
    ts = fn.split("_")[0]
    # audio_ts = 1676603660000
    audio_ts = int(ts)
    sac_ts = audio_ts + offset + s * 1000

    raw = audio[int(s * fs) : int(e * fs)]
    audio_slice = apply_bandpass_filter(raw, lowfreq, highfreq, fs)

    # plt.ylim(-ymax,ymax)
    # plt.plot(time,audio_slice)
    plt.plot(audio_slice)
    plt.axvline(x=len(audio_slice)/2, color='r', linestyle='--')
    plt.show()

    # find peak in a small window for the bang
    audio_zoom = abs(audio_slice[int(fs * 1.7):int(fs * 2.3)])
    max_audio = np.argmax(audio_zoom)
    audio_shift = (2.3-1.7) * 48000 / 2
    audio_shift -= max_audio
    audio_shift = audio_shift / 48000

    # using the time shift, plot the audio signal
    real_t = bang - audio_shift
    s = real_t - 2
    e = real_t + 2
    raw = audio[int(s * fs) : int(e * fs)]
    audio_slice = apply_bandpass_filter(raw, lowfreq, highfreq, fs)

    # plt.ylim(-ymax, ymax)
    plt.plot(audio_slice)
    plt.axvline(x=len(audio_slice)/2, color='r', linestyle='--')
    plt.show()

    # plot original sac trace
    start = obspy.UTCDateTime(datetime.datetime.utcfromtimestamp((float(sac_ts - 1000)) / 1000))
    end = obspy.UTCDateTime(datetime.datetime.utcfromtimestamp((float(sac_ts + 4999)) / 1000))

    st_buffer = sac_file.slice(starttime=start, endtime=end)[0].data
    st = st_buffer[1000:5000]

    # plt.ylim(-0,20)
    plt.plot(stime, st)
    plt.axvline(x=2, color='r', linestyle='--')
    plt.show()

    st_window = st[1700:2300]
    max_sac = np.argmax(st_window)
    sac_shift = int(len(st_window) / 2 - max_sac)
    sac_shift_seconds = sac_shift / 1000

    # st = st_buffer[1000-sac_shift:5000-sac_shift]
    # plt.ylim(-20,20)
    # plt.plot(stime, st)
    # plt.axvline(x=2, color='r', linestyle='--')
    # plt.show()

    # get the real offset
    real_offset = offset + audio_shift * 1000 - sac_shift
    # sac_ts = audio_ts + offset + s * 1000
    sac_ts = audio_ts + real_offset + s * 1000

    # use the real offset to find plot the sac trace
    start = obspy.UTCDateTime(datetime.datetime.utcfromtimestamp((float(sac_ts)) / 1000))
    end = obspy.UTCDateTime(datetime.datetime.utcfromtimestamp((float(sac_ts + 3999)) / 1000))

    st_buffer = sac_file.slice(starttime=start, endtime=end)[0].data
    st = st_buffer

    # plt.ylim(-20,20)
    plt.plot(stime, st)
    plt.axvline(x=2, color='r', linestyle='--')
    plt.show()
    return int(real_offset)


def findFrame(video):
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        print("Error opening video file")
        exit()


    framenumber = int(112.5*30)
    cap.set(1, framenumber)

    for i in range(30):
        ret, frame = cap.read()
        cv2.imwrite(f'{framenumber+i}.jpg', frame)

    exit()

# video time + offset = phone time

# bang: time to start stomping
# offset: difference between video file name  & UTC+8
x = findOffset(r"F:\video2extract\64\1700707308000_48000_0.npy", r"F:\sac4timeshift\1123\231123.000000.EB003064.HHZ.sac",
           144.2,60)
print(x)
exit()