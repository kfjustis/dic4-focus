import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt
import math
import pywt
from scipy.fftpack import dct, idct
from PIL import Image

'''
References:
    https://sukhbinder.wordpress.com/2014/09/11/karhunen-loeve-
    transform-in-python/

    http://stackoverflow.com/questions/3680262/how-to-slice-a-2d-python-
    array-fails-with-typeerror-list-indices-must-be-int

    http://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller
    -2d-arrays
'''

def KLT(array):

    val, vec = np.linalg.eig(np.cov(array))
    klt = np.dot(vec, array)
    return klt, vec, val

def load_image_as_array(imgFile):
	img = Image.open(imgFile)
	imgArray = np.asarray(img)

	return imgArray

def get_2d_list_slice(self, matrix, start_row, end_row, start_col, end_col):
    return [row[start_col:end_col] for row in matrix[start_row:end_row]]

"""
Return an array of shape (n, nrows, ncols) where
n * nrows * ncols = arr.size

If arr is a 2D array, the returned array should look like n subblocks with
each subblock preserving the "physical" layout of arr.
"""
def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

"""
Return an array of shape (h, w) where
h * w = arr.size

If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
then the returned array preserves the "physical" layout of the sublocks.
"""
def unblockshaped(arr, h, w):
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
        for j in range (0, 4096): # for all the 8x8 blocks
            blockArr[j] = dct(np.array(blockArr[j]), 2, norm='ortho')

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
            blockArr[j] = idct(np.array(blockArr[j]), 2, norm='ortho')


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
        error += math.exp(int(arr1[i]) - int(arr2[i]))
        error = error / 262144

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
    print("    Image loaded!")
    print()

    # do klt
    print("Determining KLT values...")
    ktm, vect, val = KLT(imgArr);
    print("    Found KLT values!")
    print()

    # run it through DCT 8x8 at a time
    # 4096 8x8 blocks in 512 x 512 array
    # 64 blocks in row, col

    print("Printing image array...")
    print(imgArr)
    print("    Done!")
    print()

    print("Applying one-level DWT...")
    coeffs = pywt.dwt2(imgArr, 'haar')
    cA, (cH, cV, cD) = coeffs
    print("    Done!")

    dwtCompList = [cA, cH, cV, cD]
    stitchArr = unblockshaped(np.array(dwtCompList), 512, 512)

    print("Displaying image...")
    #pass1 = Image.fromarray(stitchArr, 'L')
    pass1 = Image.fromarray(stitchArr)
    pass1.show()
    #pass1.save('DWT2D_transform.spi', format='SPIDER')
    print("    Done!")

    print("Applying inverse transform and displaying image...")
    finalArr = np.array((pywt.idwt2(coeffs, 'haar')), dtype=np.uint8)
    pass2 = Image.fromarray(finalArr)
    pass2.save('iDWT2D_reconstruction.bmp')
    print("    Done!")
    print()

    '''
    Calculate the error between the two images
    '''
    print("Calculating error between the original image and final...")
    imgError = calculate_error(np.asarray(imgArr), np.asarray(finalArr))
    print ("Mean squared error: ""{:.12}%".format(float(imgError * 100)))

if __name__ == "__main__":
	main(sys.argv[1:])
