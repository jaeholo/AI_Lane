from detectcar import processframes
import os
import numpy as np
from tqdm import tqdm
import multiprocessing
from analyzeframes import analyzenpy
from compare import comparingnpy

# exit()
if __name__ == '__main__':
    '''
    here we will use the trained model to detect the car, and compare with the truth to test the model's performance
    STATION, DATE, SAMPLERATE are decided by the task itself
    p1: the truth of car detection, which is taken as label
    p2: the detection given by our model, to compare with the truth
    modelnum: which model is going to be tested, 1st, 2nd, 3rd...
    TIMES: How many times have you tried to train the model 
    '''
    STATION = "35"
    DATE = "22"
    SAMPLERATE = "250hz"
    TIMES = "3"
    # processframes(dir="F:/35/label1", savedir=f"F:/35/preds/{SAMPLERATE}/{STATION}_{DATE}/{TIMES}/label", label=True)
    # processframes(dir="F:/35/result1", savedir=f"F:/35/preds/{SAMPLERATE}/{STATION}_{DATE}/{TIMES}/model",
    #               label=False)
    # # p1 = multiprocessing.Process(target=detectcar.processframes, args=(f"F:/35/check_label", f"F:/35/preds/{SAMPLERATE}/{STATION}_{DATE}/{TIMES}/label", True))
    # # p2 = multiprocessing.Process(target=detectcar.processframes, args=(f"F:/35/result1", f"F:/35/preds/{SAMPLERATE}/{STATION}_{DATE}/{TIMES}/model", False))
    # # p1.start()
    # # p2.start()
    # print("Detection done!")

    # labeldir = r"F:\35\preds\250hz\35_22\2\label"
    # modeldir = r"F:\35\preds\250hz\35_22\2\model"
    # analyzenpy(labeldir=labeldir, modeldir=modeldir, samplerate=SAMPLERATE, station=STATION, date=DATE, modelnum=3)
    # print("Analysis done!")

    comparingnpy(samplerate=SAMPLERATE, station=STATION, date=DATE, modelnum=TIMES)
    print("Comparing done!")
