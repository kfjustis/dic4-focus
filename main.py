import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.fftpack import dct, idct
from PIL import Image

def KLT(array):
    '''
    Source: https://sukhbinder.wordpress.com/2014/09/11/karhunen-loeve-
    transform-in-python/
    '''
    val, vec = np.linalg.eig(np.cov(array))
    klt = np.dot(vec, array)
    return klt, vec, val

def load_image_as_array(imgFile):
	img = Image.open(imgFile)
	imgArray = np.asarray(img)

	return imgArray

'''
Source: http://stackoverflow.com/questions/3680262/how-to-slice-a-2d-python-
array-fails-with-typeerror-list-indices-must-be-int
'''
def get_2d_list_slice(self, matrix, start_row, end_row, start_col, end_col):
    return [row[start_col:end_col] for row in matrix[start_row:end_row]]

'''
Source: http://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller
-2d-arrays
'''
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

'''
Source: http://stackoverflow.com/questions/16873441/form-a-big-2d-array-from-
multiple-smaller-2d-arrays/16873755#16873755
'''
def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))

def dct_loop(img_array, loop_times):
    i = 0

    for i in range(0, loop_times):
        # slice into 8x8 chunks
        blockArr = blockshaped(img_array, 8, 8)

        # run DCT on each chunk
        j = 0
        for j in range (0, 4096): #4096
            blockArr[j] = dct(np.array(blockArr[j]), 1)

        # reshape
        dct_arr = unblockshaped(blockArr, 512, 512)
        img_array = dct_arr

    return dct_arr

def idct_loop(img_array, loop_times):
    i = 0

    for i in range(0, loop_times):
        # slice into 8x8 chunks
        blockArr = blockshaped(img_array, 8, 8)

        # run DCT on each chunk
        j = 0
        for j in range (0, 4096):
            blockArr[j] = idct(np.array(blockArr[j]), 1) / 6


        # reshape
        dct_arr = unblockshaped(blockArr, 512, 512)
        img_array = dct_arr

    return dct_arr

def calculate_error(img1_arr, img2_arr):
    arr1 = img1_arr.ravel()
    arr2 = img2_arr.ravel()

    i = 0
    error = 0.0
    for i in range (0, 262144):
        error += math.exp(int(arr1[i]) - int(arr2[i])) / 262144
        #error = error / 262144

    return error

def main(argv):
    inputFile = ""

    # load file with command line args
    try:
        opts, args = getopt.getopt(argv,"i:")
    except getopt.GetoptError:
        print("USAGE: python3 main.py -i <file>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-i":
            inputFile = arg
        else:
            print("USAGE: python3 main.py -i <file>")
            sys.exit()

    if (inputFile is ""):
        print("USAGE: python3 main.py -i <file>")
        sys.exit()

    # load image as array
    print("Loading image...")
    imgArr = load_image_as_array(inputFile)
    print("Image loaded!")
    print()

    # do klt
    print("Determining KLT values...")
    ktm, vect, val = KLT(imgArr);
    print("Found KLT values!")
    print()

    '''
    Now we need to generate the debauchies images with debauchies filter bank
    '''

    # run it through DCT 8x8 at a time
    # 4096 8x8 blocks in 512 x 512 array
    # 64 blocks in row, col

    dctArr = dct_loop(imgArr, 1)
    img2 = Image.fromarray(dctArr, 'L')

    fixed = idct_loop(dctArr, 1)
    img2 = Image.fromarray(fixed, 'L')
    img2.show(img2)

    '''
    Calculate the error between the two images
    '''

    imgError = calculate_error(np.asarray(imgArr), np.asarray(img2))
    print (imgError)
    print ("{:.3f}%".format(imgError * 100))

if __name__ == "__main__":
	main(sys.argv[1:])
