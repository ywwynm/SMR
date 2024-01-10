import cv2
import numpy as np


def get_cropped_img(img_np, x, y, w, h):
  if img_np is None:
    return None

  img_h, img_w, c = img_np.shape

  if x < 0:
    x = 0
  if y < 0:
    y = 0

  x2 = x + w
  y2 = y + h
  if x2 > img_w:
    x2 = img_w
  if y2 > img_h:
    y2 = img_h

  x = int(x)
  y = int(y)
  x2 = int(x2)
  y2 = int(y2)

  if x2 == x:
    x2 += 1
  if y2 == y:
    y2 += 1

  return img_np[y:y2, x:x2, :]


def get_cropped_img_from_path(img_path: str, x, y, w, h):
  img = cv2.imread(img_path)
  if img is None:
    return None

  return get_cropped_img(img, x, y, w, h)