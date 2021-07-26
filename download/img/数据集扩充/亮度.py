import time

import numpy as np
import cv2
from PIL import Image
import shutil

'''
_brightness
    图像亮度增强
'''
def brightness(img):
    img = Image.fromarray(img)

    brightness = 1 + np.random.randint(1, 9) / 10
    brightness_img = img.point(lambda p: p * brightness)

    return Image.fromarray(np.uint8(brightness_img))

def saveBrightnessLabel(name):
    shutil.copyfile(name + ".txt", name + "_brightness.txt")

'''
_darkness
    图像亮度降低
'''
def darkness(img):
    darkness = np.random.randint(1, 9) / 10
    darkness_img = img * darkness
    return Image.fromarray(np.uint8(darkness_img))

def saveDarknessLabel(name):
    shutil.copyfile(name + ".txt", name + "_darkness.txt")

img=cv2.imread('Original.jpg')
send_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img2=brightness(send_image)
img3=darkness(send_image)

crop_im = cv2.cvtColor(np.array(img2), cv2.COLOR_RGBA2BGRA)
crop_im2 = cv2.cvtColor(np.array(img3), cv2.COLOR_RGBA2BGRA)

cv2.imwrite("bright.jpg", crop_im)
cv2.imwrite("dark.jpg", crop_im2)


# cv2.imshow('UDP 视频传输',crop_im)
# if cv2.waitKey(1) == 27:  # 按下“ESC”退出
#     a=1
# time.sleep(1)
#