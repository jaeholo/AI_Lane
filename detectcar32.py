import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import threading
import multiprocessing

# test the model behaviour

threshold = 2.5 * 16 / 2  # testing value
def framecarsmiddle(lanes, label=False):
    s = 5
    e = int(45/2)
    if label:
        s = int(27/2)
        e = int(32/2)
    list_of_cars = np.zeros(3)
    for i in range(3):
        for x in range(s,e,2):
            check = sum(lanes[i][x:x+6])
            if check > threshold:
                list_of_cars[i] = 1
                break
    return list_of_cars

def processframes(dir, savedir, label=False):
    npyfiles = os.listdir(dir)
    starttime = npyfiles[0].split("_")[0]

    final1 = []
    final2 = []
    final3 = []
    for i, fn in tqdm(enumerate(npyfiles)):
        frame = np.load(os.path.join(dir, fn))

        lane1 = np.sum(frame[2:10, :], axis=0)
        lane2 = np.sum(frame[12:20, :], axis=0)
        lane3 = np.sum(frame[22:30, :], axis=0)
        lanes = np.stack((lane1,lane2,lane3))

        loc = framecarsmiddle(lanes,label)
        final1.append(loc[0])
        final2.append(loc[1])
        final3.append(loc[2])

    os.makedirs(savedir, exist_ok=True)
    np.save(os.path.join(savedir,f"{starttime}_lane1.npy"),np.array(final1))
    np.save(os.path.join(savedir,f"{starttime}_lane2.npy"),np.array(final2))
    np.save(os.path.join(savedir,f"{starttime}_lane3.npy"),np.array(final3))


###
###
###
station = "34"
date = "22"
samplerate = "250hz"

#1700534766588
# 1, 2 = 6 hr
# 3 = 3 hr
# 4 = 12 hr worse
# 5 = 9 hr
# 6 = 12 hr better
if __name__ == "__main__":
    # create processes
    # args, 1 = input, 2 = output, 3 = label or not

    p1 = multiprocessing.Process(target=processframes, args=(f"F:/34/250nz32npz/check_label",f"F:/34/preds/{samplerate}/{station}_{date}/1/label", True))
    p2 = multiprocessing.Process(target=processframes, args=(f"F:/34/result",f"F:/34/preds/{samplerate}/{station}_{date}/1/model", False))

    # p3 = multiprocessing.Process(target=processframes, args=(f"C:/Users/Lucas/Desktop/data/new/combined/12",f"C:/Users/Lucas/Desktop/data/new/preds/{samplerate}/{station}_{date}/new12/label", True))
    # p4 = multiprocessing.Process(target=processframes, args=(f"C:/Users/Lucas/Desktop/data/m20/results/250nz1",f"C:/Users/Lucas/Desktop/data/m20/preds/{samplerate}/{station}_{date}/250nz1/model", False))

    # p5 = multiprocessing.Process(target=processframes, args=(f"C:/Users/Lucas/Desktop/data/new/combined/13",f"C:/Users/Lucas/Desktop/data/new/preds/{samplerate}/{station}_{date}/new13/label", True))
    # p6 = multiprocessing.Process(target=processframes, args=(f"C:/Users/Lucas/Desktop/data/m20/results/250z1",f"C:/Users/Lucas/Desktop/data/m20/preds/{samplerate}/{station}_{date}/250z1/model", False))

    # 11 12
    # start processes 1700635937466_0_0

    p1.start()
    p2.start()
    # p3.start()
    # p4.start()
    # p5.start()
    # p6.start()

    # wait until both processes finish
    p1.join()
    p2.join()
    # p3.join()
    # p4.join()
    # p5.join()
    # p6.join()

    print("Done")