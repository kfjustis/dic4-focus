import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt
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

def main(argv):
    inputFile = ""

    #load file with command line args
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

    #load image as array
    print("Loading image...")
    imgArr = load_image_as_array(inputFile)
    print("Image loaded!")

    #do klt
    ktm, vect, val = KLT(imgArr);
    '''
    i = 0
    for i in range(0, 16):
        plt.stem(val[i,:], linefmt='-', markerfmt='o')
        plt.savefig('vec'+str(i)+'.png', bbox_inches='tight')
        plt.clf()
    '''
    print(ktm)
    print(vect)
    print(val)

    '''
    Now we need to generate the debauchies images with debauchies filter bank
    '''

if __name__ == "__main__":
	main(sys.argv[1:])
