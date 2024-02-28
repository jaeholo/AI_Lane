import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def comparearrays1(label,model,header):
    for i in range(len(label)):
        dif = []
        for j in range(len(model)):
            dif.append(abs(model[j]-label[i]))

        if len(dif) == 0:
            return 0, 0

        minidx = np.argmin(dif)
        if dif[minidx] < 15:
            model[minidx] = 1000000
            label[i] = 1000000
    precision = np.count_nonzero(label == 1000000) / len(model)
    recall = np.count_nonzero(label == 1000000) / len(label)
    f1 = (2 * precision * recall) / (precision + recall)
    print(f'{header}: '
          f'{np.count_nonzero(label == 1000000)}, {len(model)}, {len(label)}, '
          f'{precision}, {recall}, {f1}')

    return np.count_nonzero(label == 1000000), len(model)

def comparearrays2(label,model,header):
    for i in range(len(label)):
        dif = []
        for j in range(len(model)):
            dif.append(abs(model[j]-label[i]))

        if len(dif) == 0:
            return 0, 0

        minidx = np.argmin(dif)
        if dif[minidx] < 30 * 3:
            model[minidx] = 1000000
            label[i] = 1000000

    precision = np.count_nonzero(label == 1000000) / len(model)
    recall = np.count_nonzero(label == 1000000) / len(label)
    f1 = (2 * precision * recall) / (precision + recall)
    print(f'{header}: '
          f'{np.count_nonzero(label == 1000000)}, {len(model)}, {len(label)}, '
          f'{precision}, {recall}, {f1}')

    return np.count_nonzero(label == 1000000), len(model)

def truerrors(label,model,header):
    for i in range(len(model)):
        dif = []
        for j in range(len(label)):
            dif.append(abs(label[j] - model[i]))

        minidx = np.argmin(dif)
        if dif[minidx] < 30 * 3:
            label[minidx] = 1000000
            # model[i] = 1000000

    precision = np.count_nonzero(label == 1000000) / len(model)
    recall = np.count_nonzero(label == 1000000) / len(label)
    f1 = (2 * precision * recall) / (precision + recall)
    print(f'{header}: '
          f'{np.count_nonzero(label == 1000000)}, {len(model)}, {len(label)}, '
          f'{precision}, {recall}, {f1}')

    return np.count_nonzero(label == 1000000), len(model)


def comparing(samplerate, station_date, modelnum):
    print(f"samplerate components: {samplerate}\n"
          f"station_date: {station_date}\n"
          f"modelnum: {modelnum}")
    print("lane# window: correct, predicted, label, precision, recall, f1score")
    files=os.listdir(f"{samplerate}/{station_date}/{modelnum}")
    ts = files[0].split("_")[0]
    l1 = np.load(f"{samplerate}/{station_date}/{modelnum}/{ts}_label1.npy")
    l2 = np.load(f"{samplerate}/{station_date}/{modelnum}/{ts}_label2.npy")
    l3 = np.load(f"{samplerate}/{station_date}/{modelnum}/{ts}_label3.npy")
    m1 = np.load(f"{samplerate}/{station_date}/{modelnum}/{ts}_model1.npy")
    m2 = np.load(f"{samplerate}/{station_date}/{modelnum}/{ts}_model2.npy")
    m3 = np.load(f"{samplerate}/{station_date}/{modelnum}/{ts}_model3.npy")
    l = np.concatenate((l1,l2,l3))
    m = np.concatenate((m1,m2,m3))
    c1_05, p1_05 = comparearrays1(l1,m1,"lane1 0.5s")
    c1_10, p1_10 = comparearrays2(l1,m1,"lane1 1.0s")
    c2_05, p2_05 = comparearrays1(l2,m2,"lane2 0.5s")
    c2_10, p2_10 = comparearrays2(l2,m2,"lane2 1.0s")
    c3_05, p3_05 = comparearrays1(l3,m3,"lane3 0.5s")
    c3_10, p3_10 = comparearrays2(l3,m3,"lane3 1.0s")
    ca_05, pa_05 = comparearrays1(l,m,"anylane 0.5s")
    ca_10, pa_10 = comparearrays2(l,m,"anylane 1.0s")
    c, p = truerrors(l,m,"repeat ok 1.0s: ")
    cr05 = (c1_05 + c2_05 + c3_05) / ca_05
    cr10 = (c1_10 + c2_10 + c3_10) / ca_10
    cp05 = (c1_05 + c2_05 + c3_05) / pa_05
    cp10 = (c1_10 + c2_10 + c3_10) / pa_10
    print(f"Correct Lane / Real Car - 0.5s: {cr05}")
    print(f"Correct Lane / Real Car - 0.5s: {cr05}")
    print(f"Correct Lane / Real Car - 1.0s: {cr10}")
    print(f"Correct Lane / Predicted Car - 0.5s: {cp05}")
    print(f"Correct Lane / Predicted Car - 1.0s: {cp10}")
    print("")

def list_cars(samplerate, station_date, modelnum):

    files=os.listdir(f"{samplerate}/{station_date}/{modelnum}")
    ts = files[0].split("_")[0]

    l1 = np.load(f"{samplerate}/{station_date}/{modelnum}/{ts}_model1.npy")
    l2 = np.load(f"{samplerate}/{station_date}/{modelnum}/{ts}_model2.npy")
    l3 = np.load(f"{samplerate}/{station_date}/{modelnum}/{ts}_model3.npy")
    print(len(l1))
    for x in l1:
        print(x)
    print()
    print(len(l2))
    for x in l2:
        print(x)
    print()
    print(len(l3))
    for x in l3:
        print(x)

# list_cars("run1", "123_123")
# exit(1)
station="35"
date="22"
# comparing("250hz", f"{station}_{date}", "1")
# compare("1000nz", f"{station}_{date}", "2")
# compare("1000nz", f"{station}_{date}", "3")
# compare("250nz", "228_20")
# compare("250enz", "194_17")
# compare("250nz", "194_17")

# 194 adj
# window, lane, correct, detected, label, accuracy, recall
# 0.5s 1, 55, 145, 195, 38%, 28%
# 1.0s 1, 70, 162, 195, 48%, 39%
# 0.5s 2, 66, 164, 203, 40%, 33%
# 1.0s 2, 91, 164, 203, 55%, 49%
# 0.5s 3, 47, 52, 115, 90%, 41%
# 1.0s 3, 47, 52, 115, 90%, 41%
# 0.5s nolane, 283, 361, 513, 78%, 55%
# 1.0s nolane, 302, 361, 513, 84%, 59%
# if we dont care about car number, but just timeframe
# 1.0s 302, 361, 513, 84%, 59%
# 0.5s lane placement = 59% correct
# 1.0s lane placement = 69% correct

# 194 mult
# window, lane, correct, detected, label, accuracy, recall
# 0.5s 1, 71, 160, 195, 44%, 36%
# 1.0s 1, 89, 160, 195, 56%, 46%
# 0.5s 2, 133, 299, 203, 44%, 66%
# 1.0s 2, 162, 299, 203, 54%, 80%
# 0.5s 3, 75, 101, 115, 74%, 65%
# 1.0s 3, 77, 101, 115, 76%, 67%
# 0.5s nolane, 367, 560, 513, 66%, 72%
# 1.0s nolane, 402, 560, 513, 72%, 78%
# if we dont care about car number, but just timeframe
# 1.0s 402, 560, 513, 89%, 80%
# 0.5s lane placement = 76% correct
# 1.0s lane placement = 82% correct

# 194 norm
# window, lane, correct, detected, label, accuracy, recall
# 0.5s 1, 72, 162, 195, 44%, 37%
# 1.0s 1, 90, 162, 195, 55%, 46%
# 0.5s 2, 108, 223, 203, 48%, 53%
# 1.0s 2, 131, 223, 203, 59%, 65%
# 0.5s 3, 65, 76, 115, 86%, 57%
# 1.0s 3, 67, 76, 115, 88%, 58%
# 0.5s nolane, 339, 461, 513, 74%, 66%
# 1.0s nolane, 364, 461, 513, 79%, 71%
# if we dont care about car number, but just timeframe
# 1.0s nolane, 364, 461, 513, 79%, 71%
# 0.5s lane placement = 72% correct
# 1.0s lane placement = 79% correct

# 20
# window, lane, correct, detected, label, accuracy, recall
# 0.5s 1, 111, 152, 163, 73%, 68%
# 1.0s 1, 117, 152, 163, 77%, 72%
# 0.5s 2, 128, 169, 160, 76%, 80%
# 1.0s 2, 133, 169, 160, 79%, 83%
# 0.5s 3, 71, 76, 80, 93%, 89%
# 1.0s 3, 72, 76, 80, 95%, 90%
# 0.5s nolane, 338, 397, 403, 85%, 84%
# 1.0s nolane, 342, 397, 403, 86%, 85%
# if we dont care about car number, but just timeframe
# 1.0s nolane, 342, 397, 403, 86%, 85%
# 0.5s lane placement = 92% correct
# 1.0s lane placement = 94% correct

# 17
# window, lane, correct, detected, label, accuracy, recall
# 0.5s 1, 159, 206, 229, 77%, 69%
# 1.0s 1, 166, 206, 229, 81%, 72%
# 0.5s 2, 221, 252, 255, 88%, 87%
# 1.0s 2, 224, 252, 255, 89%, 88%
# 0.5s 3, 112, 112, 124, 100%, 90%
# 1.0s 3, 112, 112, 124, 100%, 90%
# 0.5s nolane, 510, 570, 608, 89%, 86%
# 1.0s nolane, 522, 570, 608, 92%, 86%
# if we don't care about car number, but just timeframe
# 1.0s nolane, 522, 570, 608, 92%, 86%
# 0.5s lane placement = 96% correct
# 1.0s lane placement = 96% correct

# miniseed
#

def comparingnpy(samplerate, station, date, modelnum):
    print(f"samplerate components: {samplerate}\n"
          f"station_date: {station}_{date}\n"
          f"modelnum: {modelnum}")
    print("lane# window: correct, predicted, label, precision, recall, f1score")
    l1 = np.load(f"{samplerate}/{station}_{date}/{modelnum}/label1.npy")
    l2 = np.load(f"{samplerate}/{station}_{date}/{modelnum}/label2.npy")
    l3 = np.load(f"{samplerate}/{station}_{date}/{modelnum}/label3.npy")
    m1 = np.load(f"{samplerate}/{station}_{date}/{modelnum}/model1.npy")
    m2 = np.load(f"{samplerate}/{station}_{date}/{modelnum}/model2.npy")
    m3 = np.load(f"{samplerate}/{station}_{date}/{modelnum}/model3.npy")
    l = np.concatenate((l1,l2,l3))
    m = np.concatenate((m1,m2,m3))
    c1_05, p1_05 = comparearrays1(l1,m1,"lane1 0.5s")
    c1_10, p1_10 = comparearrays2(l1,m1,"lane1 1.0s")
    c2_05, p2_05 = comparearrays1(l2,m2,"lane2 0.5s")
    c2_10, p2_10 = comparearrays2(l2,m2,"lane2 1.0s")
    c3_05, p3_05 = comparearrays1(l3,m3,"lane3 0.5s")
    c3_10, p3_10 = comparearrays2(l3,m3,"lane3 1.0s")
    ca_05, pa_05 = comparearrays1(l,m,"anylane 0.5s")
    ca_10, pa_10 = comparearrays2(l,m,"anylane 1.0s")
    c, p = truerrors(l,m,"repeat ok 1.0s: ")
    cr05 = (c1_05 + c2_05 + c3_05) / ca_05
    cr10 = (c1_10 + c2_10 + c3_10) / ca_10
    cp05 = (c1_05 + c2_05 + c3_05) / pa_05
    cp10 = (c1_10 + c2_10 + c3_10) / pa_10
    print(f"Correct Lane / Real Car - 0.5s: {cr05}")
    print(f"Correct Lane / Real Car - 0.5s: {cr05}")
    print(f"Correct Lane / Real Car - 1.0s: {cr10}")
    print(f"Correct Lane / Predicted Car - 0.5s: {cp05}")
    print(f"Correct Lane / Predicted Car - 1.0s: {cp10}")
    print("")