# USAGE
# python stitch.py --N 3 --PATH /01.png,/02.png,/03.png

# import the necessary packages
from pyimagesearch.panorama import Stitcher
import argparse
import imutils
import cv2
import time
start_time = time.time()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--N", required=True, help="Number of Images")
ap.add_argument("-p", "--PATH", required=True, help="path to the image")
ap.add_argument("-c", "--CENTER", required=True, help="Center of image")
ap.add_argument("-s", "--SHOW", required=False, help="Show")
args = vars(ap.parse_args())

# load the two images and resize them to have a width of 400 pixels
# (for faster processing)
N = int(args["N"])
imgP = args["PATH"].split(',')
C = int(args["CENTER"])
img = [None]*N
for i in range(0,N):
    img[i] = cv2.imread(imgP[i])
    # print(img[i][0,0])
    # cv2.imshow(("Image " + str(i)), img[i])
    img[i] = imutils.resize(img[i], width=300)

# stitch the images together to create a panorama
stitcher = Stitcher()
stitcher.stitch(N, img, C, showMatches=True)
#(result, vis) = stitcher.stitch(N, img, 2, showMatches=True)
print("--- %s seconds ---" %(time.time() - start_time))
'''
# show the images
for i in range(0,N):
    cv2.imshow(("Image " + str(i)), result[i])
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
'''
cv2.waitKey(0)

