import os
import shutil
from tqdm import tqdm


def moveup(dir):
    subdirs = os.listdir(dir)
    int = 0
    for d in tqdm(subdirs):
        subdir_path = os.path.join(dir, d)
        files = os.listdir(subdir_path)
        for f in files:
            original = os.path.join(subdir_path, f)
            new = os.path.join(dir, f)
            shutil.move(original, new)


def rename(datedir, date):
    # Iterate over the files in the directory
    for subdir in os.listdir(datedir):
        addn = f'{date}_' + subdir.split("_")[0] + "_"
        dir = os.path.join(datedir, subdir)
        for f in os.listdir(dir):
            n, e = os.path.splitext(f)
            n = n.replace("array", "")
            newname = addn + n + e

            orig = os.path.join(dir, f)
            newn = os.path.join(dir, newname)
            # print(orig)
            # print(newn)
            os.rename(orig, newn)

# # Directory containing the files
# date = 20
# datedir = f'D:/Bafang/data/check250/{date}'
#
# rename(datedir, date)

moveup(r"F:\35\250nz64npz")
