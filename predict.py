"""
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要保存，利用r_image.save("img.jpg")即可保存。
3、如果想要获得框的坐标，可以进入detect_image函数，读取top,left,bottom,right这四个值。
4、如果想要截取下目标，可以利用获取到的top,left,bottom,right这四个值在原图上利用矩阵的方式进行截取。
"""
from centernet import CenterNet
from PIL import Image
import numpy as np
import cv2
import os


def detect(model, img_path):
    image = cv2.imread(img_path)
    image = Image.fromarray(image.astype('uint8')).convert('RGB')
    r_image = np.array(model.detect_image(image))
    cv2.namedWindow("CenterNet", cv2.WINDOW_NORMAL)
    cv2.imshow('CenterNet', r_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    input_img = "./img"
    model = CenterNet()
    for img in os.listdir(input_img):
        img_path = os.path.join(input_img, img)
        detect(model, img_path)
