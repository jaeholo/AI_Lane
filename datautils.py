import tqdm
import os
import shutil
import numpy
from tqdm import tqdm
import multiprocessing
import time

# def sort_data(basedir, baseoutput, bucket_size):
#     dir_num = 0
#     file_num = 0
#     curr_output_path = ""
#     for d in sorted(os.listdir(basedir),  key=lambda x: int(x)):
#         curr_source_path = os.path.join(basedir, d)
#         curr_output_path = os.path.join(baseoutput, str(dir_num))
#         os.makedirs(curr_output_path, exist_ok=True)
#         for f in tqdm(os.listdir(curr_source_path)):
#             if (file_num // bucket_size) > dir_num:
#                 dir_num += 1
#                 curr_output_path = os.path.join(baseoutput, str(dir_num))
#                 os.makedirs(curr_output_path, exist_ok=True)
#             shutil.move(os.path.join(curr_source_path, f), os.path.join(curr_output_path, f))
#             file_num += 1
#
#
# # sort_data("D:/Bafang/data/trainingdata/bird_saconly", "D:/Bafang/data/dalle/3+1/sac", 8192)
# # sort_data("D:/Bafang/data/trainingdata/mbird", "D:/Bafang/data/dalle/3+1/graph", 8192)
# # sort_data("D:/Bafang/data/dalle/original/e", "D:/Bafang/data/dalle/split/e", 8192)
# # sort_data("D:/Bafang/data/dalle/original/z", "D:/Bafang/data/dalle/split/z", 8192)
#
# def combine(d1, d2, d3, d4):
#     dir_name = os.listdir(d1)
#     for dir in tqdm(dir_name):
#         filenames = os.listdir(dir)
#         for file in filenames:
#             suffix = os.path.join(dir, file)
#             arr1 = numpy.load(os.path.join(d1, suffix))
#             arr2 = numpy.load(os.path.join(d2, suffix))
#             arr3 = numpy.load(os.path.join(d3, suffix))
#             arr4 = numpy.load(os.path.join(d4, suffix))
#             # out =
#
# def get_names(directory, out):
#     filenames = os.listdir(directory)
#     # random.shuffle(filenames)
#
#     with open(out, 'w') as file:
#         # Write each filename to a new line in the text file
#         for filename in tqdm(filenames):
#             file.write(filename + '\n')
#
# directory = "D:/Bafang/data/traindevtest/train2/m"
# outputfile = "train2.txt"
#
# # get_names(directory, outputfile)
#
# def moveup(dir):
#     subdirs = os.listdir(dir)
#     int = 0
#     for d in tqdm(subdirs):
#         subdir_path = os.path.join(dir, d)
#         files = os.listdir(subdir_path)
#         for f in files:
#             original = os.path.join(subdir_path, f)
#             new = os.path.join(dir, f)
#             shutil.move(original, new)


# moveup("D:/Bafang/data/hopez/train/17")
# moveup("D:/Bafang/data/hopez/train/17")
# moveup("D:/Bafang/data/hopez/train/17")


# moveup("D:/Bafang/data/dalle/3+1/graph/train")
# moveup("D:/Bafang/data/dalle/3+1/graph/dev")
# moveup("D:/Bafang/data/dalle/3+1/graph/test")
# moveup("D:/Bafang/data/dalle/3+1/sac/train")
# moveup("D:/Bafang/data/dalle/3+1/sac/dev")
# moveup("D:/Bafang/data/dalle/3+1/sac/test")
# combine()

def to_npz(npydir, npzdir, station):
    for d in sorted(os.listdir(dir),  key=lambda x: int(x)):
        path = os.path.join(dir, d)
        newpath = os.path.join(newdir, d)
        textpath = os.path.join(newdir, d + "_txt")
        os.makedirs(newpath, exist_ok=True)
        os.makedirs(textpath, exist_ok=True)
        files = os.listdir(path)
        # files = files[70:-70] # this is used when the input data is audio
        data = {}
        npz_num = 0
        i = 0
        textfile = open(os.path.join(textpath, f'{station}_{d}_{npz_num}.txt'), "w")
        for f in tqdm(files):
            textfile.write(str(i) + " " + f.split(".")[0] + "\n")
            arr = numpy.load(os.path.join(path, f))
            key = f'array{i}'
            data[key] = arr
            i += 1
            if i == 4096:
                numpy.savez(os.path.join(newpath, f'{station}_{d}_{npz_num}'), **data)
                i = 0
                data.clear()
                npz_num += 1
                textfile.close()
                textfile = open(os.path.join(textpath, f'{station}_{d}_{npz_num}.txt'), "w")
        numpy.savez(os.path.join(newpath, f'{station}_{d}_{npz_num}'), **data)
        textfile.close()


to_npz(r"F:\35\250nz64", r"F:\35\250nz64npz", "035")
# exit()
#
# inputa = [
#           "D:/Bafang/data/250nz/17",
#           "D:/Bafang/data/250nz/18",
#           # "D:/Bafang/data/hopen/17/250enz",
#           "D:/Bafang/data/250nz/20",
#           # "D:/Bafang/data/hope/17/1000nzbp",
#           # "D:/Bafang/data/hope/18/1000nzbp",
#           # # "D:/Bafang/data/hopen/17/250enzbp",
#           # "D:/Bafang/data/hope/20/1000nzbp",
#           ]
# inputb = [
#           "D:/Bafang/data/check250/17",
#           "D:/Bafang/data/check250/18",
#           # "D:/Bafang/data/250enz/n/17",
#           "D:/Bafang/data/check250/20",
#           # "D:/Bafang/data/1000nzbp/17",
#           # "D:/Bafang/data/1000nzbp/18",
#           # # "D:/Bafang/data/250enzbp/n/17",
#           # "D:/Bafang/data/1000nzbp/20",
#           ]
#
# # Wrapper function for calling your_function with two parameters
# def function_wrapper(params):
#     return to_npz(*params)
#
# if __name__ == '__main__':
#     # Create a list of parameter tuples
#     param_tuples = list(zip(inputa, inputb))
#
#     # Create a pool of worker processes
#     pool = multiprocessing.Pool()
#
#     # Run the function in parallel with different parameters
#     results = pool.map(function_wrapper, param_tuples)
#
#     # Close the pool of worker processes
#     pool.close()
#     pool.join()

# to_npz("D:/Bafang/data/dalle/1", "D:/Bafang/data/dallez/1", "E:/Bafang/dallez/1")
# to_npz("D:/Bafang/data/dalle/4", "D:/Bafang/data/dallez/4", "E:/Bafang/dallez/4")
# to_npz("D:/Bafang/data/dalle/3+1/graph", "D:/Bafang/data/dallez/3+1/graph", "E:/Bafang/dallez/3+1/graph")
# to_npz("D:/Bafang/data/dalle/3+1/sac", "D:/Bafang/data/dallez/3+1/sac", "E:/Bafang/dallez/3+1/sac")




