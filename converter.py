import torch
import os

os.add_dll_directory("C:\\Program Files (x86)\\VTK\\bin")  # not sure why interpreter is not finding this

import cv2
import glob
import numpy as np
import time
import videoutils
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ultralytics import YOLO
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool

'''
transfer the video into bird view pic
'''
class Converter:
    def __init__(self, model="yolov8x.pt", w_out=64, h_out=64, batch_size=32, time_offset=0, start_ms=0,
                 input_video_directory="", output_array_directory="", output_array_directory_simple="", output_array_subdirectory="", resolution=360,
                 lanes=(740,587,536,500), trees=(340, 1580), tilt=0.75):
        # 17th numbers [915, 760, 712, 680, 264, 1635] 0
        # 18th numbers [775, 642, 597, 568, 364, 1716] -0.6
        # 19th numbers [860, 725, 680, 650, 295, 1650] 1.5
        # 20th numbers [773, 640, 595, 567, 265, 1625] -0.2a
        # n17 numbers  [725, 592, 550, 520, 320, 1600]
        # 3 numbers [844, 730, 690, 660, 360, 1560] 0.4
        # 12 numbers [775, 664, 621, 588, 346, 1574] 0.75
        # 13 numbers [815, 674, 624, 588, 344, 1576] -0.55
        # 35 numbers[500, 536, 587, 740, 340, 1580] 0.75

        self.__model = YOLO(model)  # input model
        self.__w_out = w_out  # default width of 2d array output
        self.__h_out = h_out  # default height of 2d array output
        self.__batch_size = batch_size  # default input batch size for generating 2d arrays
        self.__time_offset = time_offset  # time offset between the video and SAC file
        self.__start_ms = start_ms  # video number being converted
        self.__input_video_directory = input_video_directory  # path to the directory of the input videos
        self.__output_array_directory = output_array_directory  # path to the directory to output the arrays
        self.__output_array_directory_simple = output_array_directory_simple  # path to the directory to output the arrays
        self.__output_array_subdirectory = output_array_subdirectory  # subdirectory to output the arrays
        self.__resolution = resolution  # default resolution for direct animation of YOLO output
        self.__lanes = lanes
        self.__trees = trees
        self.__tilt = tilt

    def set_model(self, model):
        self.__model = model

    def set_w_out(self, w_out):
        self.__w_out = w_out

    def set_h_out(self, h_out):
        self.__h_out = h_out

    def set_batch_size(self, batch_size):
        self.__batch_size = batch_size

    def set_time_offset(self, time_offset):
        self.__time_offset = time_offset

    def set_start_ms(self, start_ms):
        self.__start_ms = start_ms

    def set_input_video_directory(self, input_video_directory):
        self.__input_video_directory = input_video_directory

    def set_output_array_directory(self, output_array_directory):
        self.__output_array_directory = output_array_directory

    def set_output_array_subdirectory(self, output_array_subdirectory):
        self.__output_array_subdirectory = output_array_subdirectory

    def set_resolution(self, resolution):
        self.__resolution = resolution

    def set_lanes(self, lanes):
        self.__lanes = lanes

    # put the vehicle detections into a 2d array of size w_out x h_out
    def generate_data_frame(self, detections, w_in, h_in):
        x_scale = self.__w_out / w_in
        y_scale = self.__h_out / h_in

        frame = np.zeros((self.__w_out, self.__h_out), dtype=np.uint8)
        cars = 0
        trucks = 0

        for box in detections:
            car = videoutils.cartype(box)
            if car != 0:
                pt1, pt2 = (int(box[0] * x_scale), int(box[1] * y_scale)), (
                int(box[2] * x_scale), int(box[3] * y_scale))
                cv2.rectangle(frame, pt1, pt2, car, -1)
                if car == 1:  # might remove car and trucks
                    cars += 1
                else:
                    trucks += 1

        return frame, cars, trucks

    def generate_aerial_frame(self, detections, w_in):
        x_scale = self.__w_out / w_in

        frame = np.zeros((self.__w_out, self.__h_out), dtype=np.uint8)
        cars = 0
        trucks = 0
        # row 0 = combined
        # row 1 = car
        # row 2 = truck
        # col 0 = farthest, col 2 = closest
        simplified = np.zeros((3, 3))

        for box in detections:
            car = videoutils.cartype(box)
            if car != 0:
                if int(box[2]) < self.__trees[0]:
                    continue
                if int(box[0]) > self.__trees[1]:
                    continue

                if car == 1:  # might remove car and trucks
                    cars += 1
                    row = 1
                else:
                    trucks += 1
                    row = 2
                wheel = int(box[3])
                # cv2.rectangle(frame, (int(box[0]),int(box[1])),
                #               (int(box[2]), int(box[3])), color, 2
                y1 = -1
                y2 = -1
                # closest lane
                if wheel <= self.__lanes[0] and wheel > self.__lanes[1]:
                    y1 = 45
                    y2 = 61
                    simplified[0][2] += 1
                    simplified[row][2] += 1
                elif wheel <= self.__lanes[1] and wheel > self.__lanes[2]:
                    y1 = 24
                    y2 = 40
                    simplified[0][1] += 1
                    simplified[row][1] += 1
                elif wheel <= self.__lanes[2] and wheel > self.__lanes[3]:
                    y1 = 3
                    y2 = 19
                    simplified[0][0] += 1
                    simplified[row][0] += 1

                pt1 = (int(box[0] * x_scale),y1)
                pt2 = (int(box[2] * x_scale),y2)

                cv2.rectangle(frame, pt1, pt2, car, -1)
        frame = frame[0:64,round(64 * self.__trees[0]/1920):round(64 * self.__trees[1]/1920)]
        frame = cv2.resize(frame,(64,64))

        return frame, cars, trucks, simplified

    # turns the video into a 2d numpy array
    def video2array(self, video_name, oasd):
        start_time = str.split(video_name, "_")[1]
        dt = datetime.strptime(start_time, "%Y%m%d%H%M%S")

        if self.__start_ms == 0:
            self.__start_ms = int(dt.timestamp() * 1000) + int(self.__time_offset)

        # TICKET: update the output path to the next day when the day rolls over in the video
        output_dir = os.path.join(self.__output_array_directory, oasd)
        output_dir_simple = os.path.join(self.__output_array_directory_simple, oasd)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir_simple, exist_ok=True)

        input_video = os.path.join(self.__input_video_directory, video_name)

        print(input_video)
        cap = cv2.VideoCapture(input_video)

        if not cap.isOpened():
            print("Error opening video file")
            exit()

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ts = 0

        for i in tqdm(range(0, num_frames, self.__batch_size)):
            # if int(oasd) < 15:
            #     break
            # Read frame
            batch = []
            timestamps = []
            for j in range(self.__batch_size):
                if i + j < num_frames:
                    ret, frame = cap.read()

                    # tilt frame
                    if self.__tilt != 0:
                        M = cv2.getRotationMatrix2D((frame_width / 2, frame_height / 2), self.__tilt, 1)
                        frame = cv2.warpAffine(frame, M, (frame_width, frame_height))

                    ts = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if not ret:
                        print("Error reading frame " + str(i+j))
                        exit()
                    batch.append(frame)
                    timestamps.append(ts)

            output = self.__model(batch, verbose=False)

            for j, result in enumerate(output):
                if not result:
                    nothing = np.zeros((self.__w_out, self.__h_out), dtype=np.uint8)
                    nothing_simple = np.zeros((3, 3), dtype=np.uint8)
                    epoch_time = self.__start_ms + timestamps[j]
                    output_name = str(round(epoch_time)) + "_0_0.npy"
                    np.save(os.path.join(output_dir, output_name), nothing)
                    np.save(os.path.join(output_dir_simple, output_name), nothing_simple)
                    continue

                detections = result.boxes.data

                # data, cars, trucks = self.generate_data_frame(detections, frame_width, frame_height)
                data, cars, trucks, simplified = self.generate_aerial_frame(detections, frame_width)
                epoch_time = self.__start_ms + timestamps[j]
                output_name = str(round(epoch_time)) + "_" + str(cars) + "_" + str(trucks) + ".npy"

                np.save(os.path.join(output_dir, output_name), data)
                np.save(os.path.join(output_dir_simple, output_name), simplified)

        # used for skipping
        # if int(oasd) < 15:
        #     # Get the total number of frames in the video
        #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #
        #     # Set the video capture to the last frame
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        #
        #     # Read the last frame
        #     ret, last_frame = cap.read()
        #
        #     # Get the timestamp of the last frame
        #     last_frame_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        #     self.__start_ms += last_frame_timestamp + (1000 / cap.get(cv2.CAP_PROP_FPS))
        #     return

        self.__start_ms += ts + (1000 / cap.get(cv2.CAP_PROP_FPS))
        print("num_frames: " + str(num_frames))

    def wrapper(self, args):
        return self.video2array(*args)

    def video2array_directory(self, directory=None):
        if directory is not None:
            self.__input_video_directory = directory
        videos = glob.glob(os.path.join(self.__input_video_directory, "*.mp4"))

        for i, v in enumerate(videos):
            # self.__output_array_subdirectory = str(i)
            self.video2array(v, str(i))
            # inputs.append((v,str(i)))

    def video_with_cars(self, video_name, lanes):
        # TICKET: update the output path to the next day when the day rolls over in the video
        input_video = os.path.join(self.__input_video_directory, video_name)

        cap = cv2.VideoCapture(input_video)

        if not cap.isOpened():
            print("Error opening video file")
            exit()

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


        # 012: 0.75
        # 013: -0.55
        # 003: 0.4
        tilt_angle_degrees = 0.8

        output_path = "example.mp4"
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(output_path, fourcc, fps,
                                 (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        fr = 0
        with tqdm(total=num_frames) as pbar:
            while cap.isOpened():
                pbar.update(1)
                ret, frame = cap.read()
                # tilted_frame = frame

                M = cv2.getRotationMatrix2D((frame_width / 2, frame_height / 2), tilt_angle_degrees, 1)
                tilted_frame = cv2.warpAffine(frame, M, (frame_width, frame_height))

                fr += 1
                if fr == 200:
                    break

                if not ret:
                    break

                detections = self.__model(frame, verbose=False)

                for box in detections[0].boxes.data:
                    color = videoutils.isCar(box)
                    if color != False:
                        wheel=int(box[3])
                        cv2.rectangle(tilted_frame, (int(box[0]),int(box[1])),
                                      (int(box[2]), int(box[3])), color, 2)

                        # closest lane
                        if wheel <= self.__lanes[0] and wheel > self.__lanes[1]:
                            lane = 2
                        elif wheel <= self.__lanes[1] and wheel > self.__lanes[2]:
                            lane = 1
                        elif wheel <= self.__lanes[2] and wheel > self.__lanes[3]:
                            lane = 0
                        else:
                            lane = -1

                # 17th numbers [915,760,712,680,264,1635]
                # cv2.line(tilted_frame, (0, 680), (1920, 680), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (0, 712), (1920, 712), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (0, 760), (1920, 760), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (0, 915), (1920, 915), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (264, 0), (264, 1080), (255, 0, 0), 1)
                # cv2.line(tilted_frame, (1635, 0), (1635, 1080), (255, 0, 0), 1)

                # 18th numbers [775, 642, 597, 568, 364, 1716] -0.6
                # cv2.line(tilted_frame, (0, 568), (1920, 568), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (0, 597), (1920, 597), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (0, 642), (1920, 642), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (0, 775), (1920, 775), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (364, 0), (364, 1080), (255, 0, 0), 1)
                # cv2.line(tilted_frame, (1716, 0), (1716, 1080), (255, 0, 0), 1)

                # 19th numbers [860, 725, 680, 650, 295, 1650]
                # cv2.line(tilted_frame, (0, 650), (1920, 650), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (0, 680), (1920, 680), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (0, 725), (1920, 725), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (0, 860), (1920, 860), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (295, 0), (295, 1080), (255, 0, 0), 1)
                # cv2.line(tilted_frame, (1650, 0), (1650, 1080), (255, 0, 0), 1)

                # 20th numbers [773, 640, 595, 567, 265, 1625] -0.2 tilt
                # cv2.line(tilted_frame, (0, 567), (1920, 567), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (0, 595), (1920, 595), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (0, 640), (1920, 640), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (0, 773), (1920, 773), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (265, 0), (265, 1080), (255, 0, 0), 1)
                # cv2.line(tilted_frame, (1625, 0), (1625, 1080), (255, 0, 0), 1)

                # 12 numbers [775, 664, 621, 588, 346, 1574] 0.75
                # cv2.line(tilted_frame, (0, 588), (1920, 588), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (0, 621), (1920, 621), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (0, 664), (1920, 664), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (0, 775), (1920, 775), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (346, 0), (346, 1080), (255, 0, 0), 1)
                # cv2.line(tilted_frame, (1574, 0), (1574, 1080), (255, 0, 0), 1)
                # cv2.line(tilted_frame, (902, 660), (1516, 660), (0, 0, 255), 1)
                # 6m=614 // 1920/2=960 :: 960-614=346 to 960+614=1574

                # 13 numbers [815, 674, 624, 588, 344, 1576] -0.55
                # cv2.line(tilted_frame, (0, 588), (1920, 588), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (0, 624), (1920, 624), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (0, 674), (1920, 674), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (0, 815), (1920, 815), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (344, 0), (344, 1080), (255, 0, 0), 1)
                # cv2.line(tilted_frame, (1576, 0), (1576, 1080), (255, 0, 0), 1)
                # cv2.line(tilted_frame, (232, 660), (848, 660), (0, 0, 255), 1)
                # 6m=616 // 960-616=344 :: 960+616=1576

                # 3 numbers [844, 730, 690, 660, 360, 1560] 0.4
                # cv2.line(tilted_frame, (0, 660), (1920, 660), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (0, 690), (1920, 690), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (0, 730), (1920, 730), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (0, 844), (1920, 844), (0, 255, 0), 1)
                # cv2.line(tilted_frame, (360, 0), (360, 1080), (255, 0, 0), 1)
                # cv2.line(tilted_frame, (1560, 0), (1560, 1080), (255, 0, 0), 1)
                # cv2.line(tilted_frame, (570, 710), (1470, 710), (0, 0, 255), 1)
                # 9m=900 -> 6m=600 // 960-600=360 :: 960+600=1560

                # Write the modified frame to the output video
                output.write(tilted_frame)
        output.release()
        cap.release()
        cv2.destroyAllWindows()



conv = Converter(input_video_directory="F:/convert/35",
                 output_array_directory=r"F:\35\res_bird64",
                 output_array_directory_simple=r"F:\35\res_simple64",
                 batch_size=64,
                 time_offset=-87109)

#3 = 30572
#12 = -81822
#13 = -50513
#34 = -55423
#35 = -87109
#48 = -92904
#63 = 20047
if __name__ == "__main__":
    # conv.video_with_cars("DJI_20231121110314_0003_D.MP4", 0) #analyse video
    conv.video2array_directory() #generate label