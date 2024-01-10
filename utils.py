import os
import datetime


def now_str():
  return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_img_size_from_seq(seq_name: str):
  size_str = seq_name.split('_')[-1]
  sizes = size_str.split('x')
  w = int(sizes[0])
  if w % 2 != 0:
    w += 1
  h = int(sizes[1])
  if h % 2 != 0:
    h += 1
  return w, h
