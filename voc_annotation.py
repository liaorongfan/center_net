# ---------------------------------------------#
#   运行前一定要修改classes
#   如果生成的2007_train.txt里面没有目标信息
#   那么就是因为classes没有设定正确
# ---------------------------------------------#
import xml.etree.ElementTree as ET
from os import getcwd

# sets=[('2012', 'train'), ('2012', 'val'), ('2012', 'test')]
sets = [('2012', 'train')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert_annotation(year, image_id, list_file):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml' % (year, image_id), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text

        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        # b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))

        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


# wd = getcwd()

for year, image_set in sets:
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt' % (year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('VOCdevkit/VOC%s/JPEGImages/%s.jpg' % (year, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()