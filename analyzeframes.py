import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing


# def analyzeframes(dir):
#     i = 0
#     for f in os.listdir(dir):
#         i += 1
#         p = np.load(os.path.join(dir, f))
#         c = np.count_nonzero(p[1] >= 0)
#         if c > 1:
#             print(i)
#             print(p[1])

def iteratecars(lane):
    # if car less than length we know its not a car and the gap is for the same car
    # actually wont work cuz sometimes you only catch the tail end of a car
    # actually it can work because if you catch above 5 then its a car, but if theres no following car
    # you know that that was the actual end
    # otherwise if its too short and it disappears then its the same car?

    carlist = []
    i = 0

    while i < len(lane):
        x = lane[i]
        if x == 1:
            if i+6 > len(lane):
                break
            # 5 or 6?
            iscar = np.sum(lane[i:i+6]) >= 4 #orig
            # iscar = np.sum(lane[i:i+5]) >= 3
            if iscar:
                carlist.append(i)
                potentialend = 0
                idx = 0
                count = 0
                while True:
                    idx += 1
                    if i+idx >= len(lane):
                        potentialend = idx
                        break
                    if lane[i+idx] == 0:
                        count += 1
                        if count == 1:
                            potentialend = idx
                        if count == 4:
                            if potentialend < 13: # orig
                            # if potentialend < 10:
                                continue
                            break
                        if count == 8:
                            break
                    else:
                        count = 0
                i += potentialend
                continue
        i += 1 # don't forget this!
    return carlist

def analyzenpy(labeldir, modeldir, samplerate, station, date, modelnum):
    for i, f in enumerate(os.listdir(labeldir)):
        os.makedirs(f"{samplerate}/{station}_{date}/{modelnum}", exist_ok=True)
        label1 = np.load(os.path.join(labeldir, f))
        carlist = iteratecars(label1)
        np.save(f"{samplerate}/{station}_{date}/{modelnum}/label{i + 1}.npy", carlist)
        print(f"{samplerate}/{station}_{date}/{modelnum}/lane{i}/label: " + str(len(carlist)))

        model1 = np.load(os.path.join(modeldir,f))
        carlist = iteratecars(model1)
        np.save(f"{samplerate}/{station}_{date}/{modelnum}/model{i+1}.npy", carlist)
        print(f"{samplerate}/{station}_{date}/{modelnum}/lane{i}/model: " + str(len(carlist)))



def analyzeframes(samplerate, station_date, modelnum):
    inputdir_label = f"F:/35/preds/{samplerate}/{station_date}/{modelnum}/label"
    inputdir_model = f"F:/35/preds/{samplerate}/{station_date}/{modelnum}/model"
    for i, f in enumerate(os.listdir(inputdir_label)):
        ts = f.split("_")[0]
        os.makedirs(f"{samplerate}/{station_date}/{modelnum}", exist_ok=True)
        # label results
        label1 = np.load(os.path.join(inputdir_label,f))
        carlist = iteratecars(label1)
        np.save(f"{samplerate}/{station_date}/{modelnum}/{ts}_label{i+1}.npy",carlist)
        print(f"{samplerate}/{station_date}/{modelnum}/lane1/label: " + str(len(carlist)))

        # model results
        model1 = np.load(os.path.join(inputdir_model,f))
        carlist = iteratecars(model1)
        np.save(f"{samplerate}/{station_date}/{modelnum}/{ts}_model{i+1}.npy",carlist)
        print(f"{samplerate}/{station_date}/{modelnum}/lane1/model: " + str(len(carlist)))


station = "35"
date = "22"
samplerate = "250hz"
modelnum = ""
if __name__ == "__main__":
    # create processes
    p1 = multiprocessing.Process(target=analyzeframes, args=(f"{samplerate}", f"{station}_{date}", modelnum))
    # p2 = multiprocessing.Process(target=analyzeframes, args=(f"{samplerate}", f"{station}_{date}", "8"))
    # p3 = multiprocessing.Process(target=analyzeframes, args=(f"{samplerate}", f"{station}_{date}", "9"))
    # p4 = multiprocessing.Process(target=analyzeframes, args=(f"{samplerate}", f"{station}_{date}", "10"))
    # p5 = multiprocessing.Process(target=analyzeframes, args=(f"{samplerate}", f"{station}_{date}", "13"))
    # p6 = multiprocessing.Process(target=analyzeframes, args=(f"{samplerate}", f"{station}_{date}", "14"))

    # start processes
    p1.start()
    # p2.start()
    # p3.start()
    # p4.start()
    # p5.start()
    # p6.start()

    # wait until both processes finish
    p1.join()
    # p2.join()
    # p3.join()
    # p4.join()
    # p5.join()
    # p6.join()

    print("Done")
