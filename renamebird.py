import os

def rename_function(dir):
    for f in os.listdir(dir):
        # print(f)
        original_time, s1, s2 = f.split("_")
        # print(original_time,s1,s2)
        # n, e = os.path.splitext(f)
        # original_time = n.split("_")[0]
        # the_left = n.split("_")[1：]
        new_time = str(int(original_time)-55423*2)
        new_name = new_time+"_"+s1+"_"+s2
        os.rename(dir+f, dir+new_name)
        # print(new_name)
        # new_name = new_time + e
        # print(original_time)
        # print(the_left)

dir = "F:/34/res_bird/16/"

rename_function(dir)