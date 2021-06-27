import cv2
import numpy as np
import random
from pathlib import Path


def rotate(img, angle):  # 对图片执行旋转操作
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    matRotate = cv2.getRotationMatrix2D((width * 0.5, height * 0.5), angle, 1)
    dst = cv2.warpAffine(img, matRotate, (width, height))
    return dst


def resize_pic(img, scale):  # 对图片执行resize操作
    shape_img = img.shape
    dstHeight = int(shape_img[0] * scale)
    dstWeight = int(shape_img[1] * scale)

    dst = cv2.resize(img, (dstWeight, dstHeight))
    return dst


def cut(img, val):  # 对图片执行剪切操作
    shape_img = img.shape
    dst = img[int(shape_img[0] * val):shape_img[0] - int(shape_img[0] * val),
              int(shape_img[1] * val * 2):int(shape_img[1] - shape_img[1] * val * 2)]
    return dst


def shift(img, val):  # 对图片执行平移操作
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    dst = np.zeros(imgInfo, np.uint8)
    for i in range(height):
        for j in range(width - val):
            dst[i, j + val] = img[i, j]

    return dst


def gasuss_noise(image, mean=0, var=0.002):  # 对图片添加高斯噪声
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    # image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var**0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out


def sp_noise(image, prob):  # 对图片添加椒盐噪声
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def show_augmentation(demo_image):
    # demo_image = x_train[0]
    aug_dir = './augmentation'
    Path(aug_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(aug_dir + '/demo.png', demo_image)

    rot = rotate(demo_image, 30)
    cv2.imwrite(aug_dir + '/rotate.png', rot)

    rsz = resize_pic(demo_image, 0.5)
    cv2.imwrite(aug_dir + '/resize.png', rsz)

    cutt = cut(demo_image, 0.1)
    cv2.imwrite(aug_dir + '/cut.png', cutt)

    shft = shift(demo_image, 10)
    cv2.imwrite(aug_dir + '/shift.png', shft)

    gauss = gasuss_noise(demo_image)
    cv2.imwrite(aug_dir + '/gauss.png', gauss)

    sp_noise_img = sp_noise(demo_image, 0.01)
    cv2.imwrite(aug_dir + '/sp_noise.png', sp_noise_img)
