import obspy
import numpy as np
from pydub import AudioSegment

AudioSegment.converter = 'F:/ffmpeg/ffmpeg-6.0-full_build/bin/ffmpeg'

# extract audio from the input video, save the audio as numpy ndarray
def extract_audio(mp4file):
    # assume first data point
    audio = AudioSegment.from_file(mp4file)
    samples = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate
    channels = audio.channels
    duration = audio.duration_seconds
    return samples, sample_rate, channels, duration

d,sr,c,t=extract_audio(r"F:\video2extract\64\DJI_20231123104148_0003_D.MP4")
channel1 = d[0::2]
channel2 = d[1::2]
# "1700..._48000_0.npy"
np.save(r"F:\video2extract\64\1700707308000_48000_0.npy", channel1)
