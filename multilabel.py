import numpy as np
import torch
import os
from tqdm import tqdm


'''
label i
'''


label_dir = "F:/35/res_simple64"
count = 0

for file in os.listdir(label_dir):
    count += len(os.listdir(label_dir+"/"+file))
# print(count)

#
# for f in tqdm(files):
#     print(f)

# a = np.asarray([[1,2,3],[4,5,6],[7,8,9]])
# print("a is" + str(a))
# a_tran = a.transpose()
# print("After transposing, a looks like" + str(a_tran))
# b = a_tran[:, 1:].flatten()
# print("After flattening, a looks like" + str(b))

# label_dir = "F:/35/test"
# # data = np.load(r"F:\35\test\1700621343194_0_0.npy")
# for filename in os.listdir(label_dir):
#     if filename != "1700621343194_0_0.npy":
#         data = np.append(data, np.load(os.path.join(label_dir, filename)), axis=0)
#
# print(data.shape)
# data.reshape()


def transferlabel(labeldir):
    count = 0
    i = 0
    for file in os.listdir(label_dir):
        count += len(os.listdir(label_dir + "/" + file))

    label = np.empty((count,3,3))
    for subdir in tqdm(os.listdir(labeldir)):
        for file in os.listdir(label_dir + "/" + subdir):
            label[i] = np.load(label_dir + "/" + subdir + "/" + file).transpose()
            i += 1
    np.save("F:/35/test/label.npy", label)

transferlabel(labeldir=label_dir)



    # first_file = os.listdir(labeldir)[0]
    # data = np.load(os.path.join(labeldir, first_file)).transpose()
    # for filename in os.listdir(labeldir):
    #     if filename != first_file:
    #         current_label = np.load(os.path.join(labeldir, filename))
    #         current_label = current_label.transpose()
    #         data = np.append(data, current_label, axis=0)
    # np.save(savedir+"label3.npy", data)
    # print("Done!")
#
# transferlabel("F:/35/res_simple64/3","F:/35/test/")

# data = np.load(r"F:\35\res_simple64\3\1700623959775_1_1.npy")
# print(data)
# print(len(os.listdir("F:/35/test")))
