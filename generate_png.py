
import numpy as np
import matplotlib.pyplot as plt
import os
os.add_dll_directory("C:\\Program Files (x86)\\VTK\\bin")
import cv2
import glob

# data = np.asarray([[1,2,3],[4,5,6],[7,8,9]])
# plt.figure()
# plt.imshow(data)
# # plt.show()
# plt.savefig(r"C:\Users\hp\Desktop\bird_view\1.png")
# exit()
def generate_png(npydir, savedir):
    i = 0
    for file in os.listdir(npydir):
        f = os.path.join(npydir, file)
        data = np.load(f)
        plt.figure()
        plt.imshow(data)
        plt.savefig(savedir + f"{i}.png")
        plt.close()
        i += 1

def img2video():
    FRAMESIZE = (640,384)
    out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 24, FRAMESIZE)
    for filename in glob.glob('F:/Project/AI_Lane/runs/detect/predict/cut_video_frames/*.jpg'):
        img = cv2.imread(filename)
        out.write(img)
    out.release()

# generate_png("C:/Users/hp/Desktop/npy", "C:/Users/hp/Desktop/bird_view/")

img2video()