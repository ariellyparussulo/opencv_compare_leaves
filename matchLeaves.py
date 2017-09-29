from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                    help='input leaf image')
ap.add_argument('-d', '--database', required=True,
                    help='input your database path')

args = vars(ap.parse_args())

image_name = args["image"]
image = cv2.imread(args["image"])
database_path = args["database"]

for file in os.listdir(database_path):
    file_path = database_path + '/' + file
    image_to_compare = cv2.imread(file_path)

    gray_Image_A = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_Image_B = cv2.cvtColor(image_to_compare, cv2.COLOR_RGB2GRAY)

    (score, diff) = compare_ssim(gray_Image_A, gray_Image_B, full=True)

    diff = (diff * 255).astype('uint8')

    print('%s = %s: %s perc' % (image_name, file_path, score * 100))
