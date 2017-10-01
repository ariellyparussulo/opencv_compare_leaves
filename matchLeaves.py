from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import os
from operator import itemgetter
import time

def rotateImage(image, degree):
    rows, cols = image.shape

    M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
    dst = cv2.warpAffine(image, M, (cols, rows))
    return dst

def compareImages(imageA, imageB, lastResult):
    (score, diff) = compare_ssim(imageA,
                                    imageB,
                                    full=True)

    diff = (diff * 255).astype('uint8')

    if lastResult < score:
        return score
    return lastResult

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                    help='input leaf image')
ap.add_argument('-d', '--database', required=True,
                    help='input your database path')

args = vars(ap.parse_args())

image_name = args["image"]
image = cv2.imread(args["image"])
database_path = args["database"]

results = []
score = 0
start_date = time.time() * 1000
for folder in os.listdir(database_path):
    folder_path = database_path + '/' + folder

    for file in os.listdir(folder_path):
        file_path = folder_path + '/' + file
        image_to_compare = cv2.imread(file_path)

        gray_Image_A = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_Image_B = cv2.cvtColor(image_to_compare, cv2.COLOR_RGB2GRAY)

        gray_resized_image_A = cv2.resize(gray_Image_A, (600, 400))
        gray_resized_image_B = cv2.resize(gray_Image_B, (600, 400))

        score = compareImages(gray_resized_image_A, gray_resized_image_B, score)

        gray_resized_image_A_rotated_45 = rotateImage(gray_resized_image_A, 45)
        score = compareImages(gray_resized_image_A_rotated_45, gray_resized_image_B, score)

        gray_resized_image_A_rotated_90 = rotateImage(gray_resized_image_A, 45)
        score = compareImages(gray_resized_image_A_rotated_90, gray_resized_image_B, score)

        gray_resized_image_A_rotated_135 = rotateImage(gray_resized_image_A, 45)
        score = compareImages(gray_resized_image_A_rotated_135, gray_resized_image_B, score)

        gray_resized_image_A_rotated_180 = rotateImage(gray_resized_image_A, 45)
        score = compareImages(gray_resized_image_A_rotated_180, gray_resized_image_B, score)

        gray_resized_image_A_rotated_225 = rotateImage(gray_resized_image_A, 45)
        score = compareImages(gray_resized_image_A_rotated_225, gray_resized_image_B, score)

        gray_resized_image_A_rotated_270 = rotateImage(gray_resized_image_A, 45)
        score = compareImages(gray_resized_image_A_rotated_270, gray_resized_image_B, score)

        gray_resized_image_A_rotated_315 = rotateImage(gray_resized_image_A, 45)
        score = compareImages(gray_resized_image_A_rotated_315, gray_resized_image_B, score)

        gray_resized_image_A_rotated_360 = rotateImage(gray_resized_image_A, 45)
        score = compareImages(gray_resized_image_A_rotated_360, gray_resized_image_B, score)

    results.append((folder, score))
    score = 0

end_date = time.time() * 1000
sorted_results = sorted(results,key=itemgetter(1))
for item in sorted_results:
    print('%s = %f perc' % (item[0], item[1] * 100))

print('time: %f s' % ((end_date - start_date)/1000))
