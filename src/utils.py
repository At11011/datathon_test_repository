import numpy as np

perms = [
        [0,1,2,3], # 0
        [0,1,3,2], # 1
        [0,2,1,3], # 2
        [0,2,3,1], # 3
        [0,3,1,2], # 4
        [3,2,1,0], # 5
        [1,0,2,3], # 6
        [1,0,3,2], # 7
        [1,2,0,3], # 8
        [1,2,3,0], # 9
        [1,3,0,2], # 10
        [1,3,2,0], # 11
        [2,0,1,3], # 12
        [2,0,3,1], # 13
        [2,1,0,3], # 14
        [2,1,3,0], # 15
        [2,3,0,1], # 16
        [2,3,1,0], # 17
        [3,0,1,2], # 18
        [3,0,2,1], # 19
        [3,1,0,2], # 20
        [3,1,2,0], # 21
        [3,2,0,1], # 22
        [3,2,1,0]  # 23
        ]

# A function to convert an image into a permutation
def image_perm(image, perm):
    
    q0 = image[0:int(128/2), 0:int(128/2)]
    q1 = image[0:int(128/2), int(128/2):128]
    q2 = image[int(128/2):128, 0:int(128/2)]
    q3 = image[int(128/2):128, int(128/2):128]

    pre_perm = [q0, q1, q2, q3]

    q0 = pre_perm[perms[perm][0]]
    q1 = pre_perm[perms[perm][1]]
    q2 = pre_perm[perms[perm][2]]
    q3 = pre_perm[perms[perm][3]]

    res = np.concatenate((np.concatenate((q0, q1), 1), np.concatenate((q2, q3), 1)), 0)
    return res

if __name__ == "__main__":
    print("Hello")
    pos = np.zeros((24,128,128))
    print(perms[1][0])

def get_pieces(img, rows, cols, row_cut_size, col_cut_size):
    pieces = []
    for r in range(0, rows, row_cut_size):
        for c in range(0, cols, col_cut_size):
            pieces.append(img[r:r+row_cut_size, c:c+col_cut_size, :])
    return pieces

# Splits an image into uniformly sized puzzle pieces
def get_uniform_rectangular_split(img, puzzle_dim_x, puzzle_dim_y):
    rows = img.shape[0]
    cols = img.shape[1]
    if rows % puzzle_dim_y != 0 or cols % puzzle_dim_x != 0:
        print('Please ensure image dimensions are divisible by desired puzzle dimensions.')
    row_cut_size = rows // puzzle_dim_y
    col_cut_size = cols // puzzle_dim_x

    pieces = get_pieces(img, rows, cols, row_cut_size, col_cut_size)

    return pieces