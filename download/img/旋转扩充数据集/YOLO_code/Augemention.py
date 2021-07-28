#!/usr/bin/env python

import cv2
import math
import numpy as np
import os
import pdb
import xml.etree.ElementTree as ET
from Image_expansion import *

class ImgAugemention():
    def __init__(self):
        self.angle = 90


    def xml_to_txt(self,dw,dh,xmin,ymin ,xmax ,ymax):



        x = (xmin+ xmax) / 2.0
        y = (ymax + ymin) / 2.0
        w = xmax- xmin
        h = ymax - ymin

        x = x /dw
        w = w / dw
        y = y / dh
        h = h / dh

        return (x, y, w, h)


    def txt_to_xml(self,Pwidth,Pheight,x,y , wp, hp):



        xmin = int((x * Pwidth + 1) - wp * 0.5 * Pwidth)

        ymin = int((y * Pheight + 1) - hp * 0.5 * Pheight)

        xmax = int((x* Pwidth + 1) + wp * 0.5 * Pwidth)

        ymax = int((y * Pheight + 1) + hp * 0.5 * Pheight)


        return xmin,ymin ,xmax ,ymax

    # rotate_img
    def rotate_image(self, src, angle, scale=1.):
        w = src.shape[1]
        h = src.shape[0]

        # convet angle into rad
        rangle = np.deg2rad(angle)  # angle in radians
        # calculate new image width and height
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5, 0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # map
        return cv2.warpAffine(
            src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))),
            flags=cv2.INTER_LANCZOS4)

    def rotate_txt(self, src,src2, x, y, wp, hp, angle, scale=1.):

        w = src.shape[1]
        h = src.shape[0]
        w1 = src2.shape[1]
        h1 = src2.shape[0]
        print(x, y, wp, hp)

        xmin,ymin ,xmax ,ymax=self.txt_to_xml(w,h,x, y, wp, hp)
        # print(xmin,ymin ,xmax ,ymax)


        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        # get width and heigh of changed image
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5, 0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # rot_mat: the final rot matrix
        # get the four center of edges in the initial martix，and convert the coord
        point1 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymin, 1]))
        point2 = np.dot(rot_mat, np.array([xmax, (ymin+ymax)/2, 1]))
        point3 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymax, 1]))
        point4 = np.dot(rot_mat, np.array([xmin, (ymin+ymax)/2, 1]))
        # concat np.array
        concat = np.vstack((point1, point2, point3, point4))
        # change type
        concat = concat.astype(np.int32)
        # print(concat)
        rx, ry, rw, rh = cv2.boundingRect(concat)


        xmin=rx
        ymin=ry
        xmax=rx+rw
        ymax=ry+rh

        rx, ry, rw, rh=self.xml_to_txt(w1, h1,xmin,ymin ,xmax ,ymax)

        print(rx, ry, rw, rh)


        return str(rx), str(ry), str(rw), str(rh)

    def process_img(self,imgs_path, txt_path, img_save_path, txt_save_path, angle_list):
        # assign the rot angles
        for angle in angle_list:
            for img_name in os.listdir(imgs_path):
                # split filename and suffix
                n, s = os.path.splitext(img_name)
                # for the sake of use yolo model, only process '.jpg'
                if s == ".jpg":
                    img_path = os.path.join(imgs_path, img_name)
                    img = cv2.imread(img_path)
                    rotated_img = self.rotate_image(img, angle)
                    save_name = n + "_" + str(angle) + "d.jpg"
                    # 写入图像
                    cv2.imwrite(img_save_path + save_name, rotated_img)
                    txt_url = img_name.split('.')[0] + '.txt'
                    save_txt_name=n + "_" + str(angle) + "d.txt"

                    with open(txt_save_path + save_txt_name, "w") as outfile:
                        with open(txt_path+txt_url, "r") as infile:
                            for line in infile.readlines():
                                words = line.split(" ")
                                x, y, w, h = self.rotate_txt(img,rotated_img, float(words[1]), float(words[2]), float(words[3]), float(words[4]), angle)
                                outfile.write(words[0] + " " + x[0:7] + " " + y[0:7] + " " + w[0:7] + " " + h[0:7] + '\n')


    def process_img2(self,imgs_path, txt_path, img_save_path, txt_save_path):
        # assign the rot angles
        for img_name in os.listdir(imgs_path):
            # split filename and suffix
            n, s = os.path.splitext(img_name)
            # for the sake of use yolo model, only process '.jpg'
            if s == ".jpg":
                img_path = os.path.join(imgs_path, img_name)
                img = cv2.imread(img_path)
                send_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                txt_url = img_name.split('.')[0] + '.txt'

                ### 亮
                crop_im = brightness(send_image)
                brightimg = cv2.cvtColor(np.array(crop_im), cv2.COLOR_RGBA2BGRA)
                name="brightness"
                save_name = n + "_" + name + ".jpg"
                save_txt_name = n + "_" + name + ".txt"
                # 写入图像
                cv2.imwrite(img_save_path + save_name, brightimg)
                copyLabel(txt_path + txt_url,txt_save_path +save_txt_name)

                ### 暗
                crop_im=darkness(send_image)
                darknessimg = cv2.cvtColor(np.array(crop_im), cv2.COLOR_RGBA2BGRA)
                name="darkness"
                save_name = n + "_" + name + ".jpg"
                save_txt_name = n + "_" + name + ".txt"
                cv2.imwrite(img_save_path + save_name, darknessimg)
                copyLabel(txt_path + txt_url,txt_save_path +save_txt_name)

                ### 对比度
                crop_im=contrast(send_image)
                contrastimg = cv2.cvtColor(np.array(crop_im), cv2.COLOR_RGBA2BGRA)
                name="contrast"
                save_name = n + "_" + name + ".jpg"
                save_txt_name = n + "_" + name + ".txt"
                cv2.imwrite(img_save_path + save_name, contrastimg)
                copyLabel(txt_path + txt_url,txt_save_path +save_txt_name)


                ### 锐化
                crop_im=sharpen(send_image)
                sharpenimg = cv2.cvtColor(np.array(crop_im), cv2.COLOR_RGBA2BGRA)
                name="sharpen"
                save_name = n + "_" + name + ".jpg"
                save_txt_name = n + "_" + name + ".txt"
                cv2.imwrite(img_save_path + save_name, sharpenimg)
                copyLabel(txt_path + txt_url,txt_save_path +save_txt_name)


                ### 高斯模糊

                crop_im=blur(send_image)
                blurimg = cv2.cvtColor(np.array(crop_im), cv2.COLOR_RGBA2BGRA)
                name="blur"
                save_name = n + "_" + name + ".jpg"
                save_txt_name = n + "_" + name + ".txt"
                cv2.imwrite(img_save_path + save_name, blurimg)
                copyLabel(txt_path + txt_url,txt_save_path +save_txt_name)


                ### 透视变换
                ### 镜像翻转

                crop_im=flip(send_image)
                flipimg = cv2.cvtColor(np.array(crop_im), cv2.COLOR_RGBA2BGRA)
                name="flip"
                save_name = n + "_" + name + ".jpg"
                save_txt_name = n + "_" + name + ".txt"
                cv2.imwrite(img_save_path + save_name, flipimg)
                saveFlipLabel(txt_path + txt_url,txt_save_path +save_txt_name)



                ### 椒盐噪声
                crop_im=addNoise(send_image)
                addNoiseimg = cv2.cvtColor(np.array(crop_im), cv2.COLOR_RGBA2BGRA)
                name="addNoise"
                save_name = n + "_" + name + ".jpg"
                save_txt_name = n + "_" + name + ".txt"
                cv2.imwrite(img_save_path + save_name, addNoiseimg)
                copyLabel(txt_path + txt_url,txt_save_path +save_txt_name)


                ### 渐晕

                # img=vignetting(send_image)
                crop_im=vignetting(send_image)
                vignettingimg = cv2.cvtColor(np.array(crop_im), cv2.COLOR_RGBA2BGRA)
                name="vignetting"
                save_name = n + "_" + name + ".jpg"
                save_txt_name = n + "_" + name + ".txt"
                cv2.imwrite(img_save_path + save_name, vignettingimg)
                copyLabel(txt_path + txt_url,txt_save_path +save_txt_name)

                ### 随机丢包

                crop_im=cutout(send_image)
                cutoutimg = cv2.cvtColor(np.array(crop_im), cv2.COLOR_RGBA2BGRA)
                name="cutout"
                save_name = n + "_" + name + ".jpg"
                save_txt_name = n + "_" + name + ".txt"
                cv2.imwrite(img_save_path + save_name, cutoutimg)
                copyLabel(txt_path + txt_url,txt_save_path +save_txt_name)





if __name__ == '__main__':
    img_aug = ImgAugemention()
    imgs_path = './img/'
    txt_path = './label/'
    img_save_path = './images/'
    txt_save_path = './labels/'
    angle_list = [0,30,60, 90, 120, 150, 180, 270,330]
    # angle_list=[0,360]
    img_aug.process_img2(imgs_path, txt_path, img_save_path, txt_save_path)
