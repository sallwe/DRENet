"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import cv2
import numpy as np
import random


def randomHueSaturationValue(
    image,
    hue_shift_limit=(-30, 30),
    sat_shift_limit=(-5, 5),
    val_shift_limit=(-15, 15),
    u=0.5,
):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(
            hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(
    image,
    mask,
    shift_limit=(-0.1, 0.1),
    scale_limit=(-0.1, 0.1),
    aspect_limit=(-0.1, 0.1),
    rotate_limit=(-0, 0),
    borderMode=cv2.BORDER_CONSTANT,
    u=0.5,
):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height]])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array(
            [width / 2 + dx, height / 2 + dy]
        )

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(
            image,
            mat,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=borderMode,
            borderValue=(0, 0, 0),
        )
        mask = cv2.warpPerspective(
            mask,
            mat,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=borderMode,
            borderValue=(0, 0, 0),
        )

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask


def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask

def randomRotate180(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(np.rot90(image))
        mask = np.rot90(np.rot90(mask))
    return image, mask

def randomRotate270(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(np.rot90(np.rot90(image)))
        mask = np.rot90(np.rot90(np.rot90(mask)))
    return image, mask

def randomcrop(image, mask, u=0.5):
    crop_rate = np.random.uniform(0.7,0.9)
    height = np.int32(image.shape[0]*crop_rate)
    width = height
    if np.random.random() < u:
        h, w, c = image.shape
        y = np.random.randint(0, h-height+1)
        x = np.random.randint(0, w-width+1)
        image = image[y:y+height,x:x+width,:]
        image = cv2.resize(image,(h, w), interpolation = cv2.INTER_CUBIC)
        mask = mask[y:y+height,x:x+width]
        mask = cv2.resize(mask,(h, w), interpolation = cv2.INTER_CUBIC)
    return image, mask

def mmlabNormalize(img):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = (img - mean) / std
    return img
  
class RoadAugmentation:
    def __init__(self, size, random_ratio, patch_size):
        self.size = size
        self.random_ratio = random_ratio
        self.patch_size = patch_size

    def get_mask_coordinate(self, label):
        N = self.size[0] // self.patch_size[0]
        mask_coordinate = []
        for i in range(N):
            for j in range(N):
                top_left_x = j * self.patch_size[0]
                top_left_y = i * self.patch_size[0]

                center_x = top_left_x + self.patch_size[0] // 2
                center_y = top_left_y + self.patch_size[0] // 2
                if (label[center_x, center_y] == 255).all():
                    mask_coordinate.append([center_x, center_y])
        return mask_coordinate

    def get_random_mask_coord(self, label):
        mask_coord = self.get_mask_coordinate(label)
        coord_length = len(mask_coord)
        random.shuffle(mask_coord)
        random_ratio = random.uniform(self.random_ratio[0], self.random_ratio[1])
        random_length = int(random_ratio * coord_length)
        return mask_coord[:random_length]

    def set_mask(self, img, label):
        half_p = self.patch_size[0] // 2
        ran_mask_coord = self.get_random_mask_coord(label)
        for coord in ran_mask_coord:
            center_y, center_x = coord
            top_left_x = max(center_x - half_p, 0)
            top_left_y = max(center_y - half_p, 0)
            bottom_right_x = min(center_x + half_p, self.size[1] - 1)
            bottom_right_y = min(center_y + half_p, self.size[1] - 1)
            img[top_left_y:bottom_right_y+1, top_left_x:bottom_right_x+1] = 0
