import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import os
from array import array

def rescale(erp_rgb_image_data, estimated_depthmap, persp_monodepth, idx):
    myMap = estimated_depthmap["frustum"]

    height = myMap.shape[0]
    width = myMap.shape[1]

    #print("The depth array is " + str(height) + " " + str(width))
    #print("The max depth is " + str(np.amax(myMap)))
    #print("The min depth is " + str(np.amin(myMap)))

    val_far = myMap[int(height/2)][int(width/2)]
    val_left = myMap[int(height/2)][int(width/4)]

    far_real = 1.8
    left_real = 1.095

    newDepthMap = []
    depthBuffer = []

    for row in myMap:
        myRow = []
        for col in row:
            depth_norm = (col - val_left) / (val_far - val_left)
            depth_real = depth_norm * (far_real - left_real) + left_real

            myRow.append(depth_real)
            depthBuffer.append(np.float32(depth_real))

        newDepthMap.append(myRow)

    return newDepthMap, depthBuffer

def visualize_real_map(output_folder, real_map):
    plt.imsave(os.path.join(output_folder, "360monodepth_new_A.png"), real_map, cmap="binary")


def list2bin(output_folder, depth_map):
    
    output_path = os.path.join(output_folder, "EVL_360_A.depth")
    #os.remove(output_path)

    output_file = open(output_path, "wb")
    float_array = array("f", depth_map)
    float_array.tofile(output_file)
    output_file.close()

    
