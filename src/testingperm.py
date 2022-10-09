from math import factorial as fac
import numpy as np

# A function to convert an image into a permutation
# still need to import images directly
def image_perm(image, perm):
    q0 = image[0:int(image_height/2), 0:int(image_width/2)]
    q1 = image[0:int(image_height/2), int(image_width/2):image_width]
    q2 = image[int(image_height/2):image_height, 0:int(image_width/2)]
    q3 = image[int(image_height/2):image_height, int(image_width/2):image_width]

    pre_perm = [q0, q1, q2, q3]

    q0 = pre_perm[perm[0]]
    q1 = pre_perm[perm[1]]
    q2 = pre_perm[perm[2]]
    q3 = pre_perm[perm[3]]

    res = np.concatenate((np.concatenate((q0, q1), 1), np.concatenate((q2, q3), 1)), 0)
    print(res.shape)
    return res

listofImages = np.zeros(24)

for i in range(24):
    # produce specific arrangement then assign to listofImage
    listofImages[i] = image_perm(image, perm)