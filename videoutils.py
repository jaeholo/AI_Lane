# 0: person
# 1: bicycle
# 2: car
# 3: motorcycle
# 4: airplane
# 5: bus
# 6: train
# 7: truck
# 8: boat
classes = [0,1,2,3,5,7]

def isCar(box):
    # print(box[5])
    color = {
        2: (0, 0, 255), # blue
        3: (0, 255, 0), # green
        5: (255, 255, 0), # yellow
        7: (255, 0, 0) # red
    }
    return color.get(int(box[5]), False)

def cartype(box):
    # print(box[5])
    # type = {
    #     2: 1, # car
    #     3: 4, # motorcycle
    #     5: 3, # bus
    #     7: 2 # truck
    # }
    type = {
        2: 1, # car
        3: 1, # motorcycle
        5: 2, # bus
        7: 2 # truck
    }
    return type.get(int(box[5]), 0)

def result2coords(box):
    return (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))