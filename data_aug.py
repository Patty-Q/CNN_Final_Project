"""
Main functions:
1. Add Gaussian noise.
2. Randomly change the brightness.
3. The image is flipped horizontally and vertically.
4. Adjust contrast.
"""
from PIL import Image,ImageEnhance
import os
import cv2
import random
from skimage.util import random_noise
from skimage import exposure

# The main function is to perform various transformation and enhancement operations on images and save the results to the specified path.
image_number = 0
# original
raw_path = "./data/train/"
# new
new_path = "./aug/train/"

# Add Gaussian noise
def addNoise(img):
    return random_noise(img, mode='gaussian', seed=13, clip=True)*255
# Randomly change brightness
def changeLight(img):
    rate = random.uniform(0.5, 1.5)
    img = exposure.adjust_gamma(img, rate) 
    return img

# Traverse the 59 folders under raw_train
for raw_dir_name in range(59):

    raw_dir_name = str(raw_dir_name)

    saved_image_path = new_path + raw_dir_name+"/"
    raw_image_path = raw_path + raw_dir_name+"/"

    if not os.path.exists(saved_image_path):
        os.mkdir(saved_image_path)

    raw_image_file_name = os.listdir(raw_image_path)

    raw_image_file_path = []

    for i in raw_image_file_name:

        raw_image_file_path.append(raw_image_path+i)

    for x in raw_image_file_path:

        img = Image.open(x)
        cv_image = cv2.imread(x)

        
        gau_image = addNoise(cv_image)
      
        light = changeLight(cv_image)
        light_and_gau = addNoise(light)

        cv2.imwrite(saved_image_path + "gau_" + os.path.basename(x),gau_image)
        cv2.imwrite(saved_image_path + "light_" + os.path.basename(x),light)
        cv2.imwrite(saved_image_path + "gau_light" + os.path.basename(x),light_and_gau)

        # 1. Flip
        # 1.1 Flip left and right
        img_flip_left_right = img.transpose(Image.FLIP_LEFT_RIGHT)
        # 1.2 上下翻转
        img_flip_top_bottom = img.transpose(Image.FLIP_TOP_BOTTOM)

        #2.Contrast
        enh_con = ImageEnhance.Contrast(img)

        contrast = 1.5

        image_contrasted = enh_con.enhance(contrast)

        #save

        img.save(saved_image_path + os.path.basename(x))

        img_flip_left_right.save(saved_image_path + "left_right_" + os.path.basename(x))

        img_flip_top_bottom.save(saved_image_path + "top_bottom_" + os.path.basename(x))

        image_contrasted.save(saved_image_path + "contrasted_" + os.path.basename(x))

        image_number += 1

        print("convert pictur" "es :%s size:%s mode:%s" % (image_number, img.size, img.mode))

 
