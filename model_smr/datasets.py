import os
import pickle

import img_utils

import cv2
import numpy as np

from tqdm import tqdm

from torch.utils.data import Dataset

from torchvision import transforms

tfms = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet
])


class COCOSMRClsDataset(Dataset):

  def __init__(self, dataset_dir, dataset_type_dir_name, smr_dict_path, codec, actual_qps, label_type='SMR-top1', resize=224):
    self.resize = resize
    self.tfms = tfms

    f = open(smr_dict_path, 'rb')
    smr_dict = pickle.load(f)
    f.close()

    img_paths = []
    img_infos = []
    for seq_with_bbox in tqdm(smr_dict.keys()):
      smr_info = smr_dict[seq_with_bbox]
      splits = seq_with_bbox.split('_')
      seq_without_bbox = splits[0] + '_' + splits[1]
      splits = splits[-1].split(',')
      x = float(splits[0])
      y = float(splits[1])
      w = float(splits[2])
      h = float(splits[3])

      ori_img_path = f'{dataset_dir}/{dataset_type_dir_name}/{seq_without_bbox}/{seq_without_bbox}.jpg'
      img_paths.append(np.array([ori_img_path, ori_img_path]).astype(np.string_))
      img_infos.append(np.array([x, y, w, h, 1.0], dtype=float))

      for i in range(len(actual_qps)):
        qp = actual_qps[i]
        qp_img_path = f'{dataset_dir}/{dataset_type_dir_name}/{seq_without_bbox}/{codec}/{qp}.jpg'
        if not os.path.exists(qp_img_path):
          qp_img_path = qp_img_path.replace('jpg', 'png')
        img_paths.append(np.array([ori_img_path, qp_img_path]).astype(np.string_))
        img_infos.append(np.array([x, y, w, h, smr_info[label_type][i]], dtype=float))

    self.img_paths_np = np.array(img_paths)
    self.img_infos_np = np.array(img_infos)

  def __len__(self):
    return len(self.img_paths_np)

  def __getitem__(self, index):
    img_paths = self.img_paths_np[index]
    img_infos = self.img_infos_np[index]
    ori_img_path = str(img_paths[0], encoding='utf-8')
    qp_img_path = str(img_paths[1], encoding='utf-8')
    x = img_infos[0]
    y = img_infos[1]
    w = img_infos[2]
    h = img_infos[3]
    label = img_infos[4]

    ori_img_np = img_utils.get_cropped_img_from_path(ori_img_path, x, y, w, h)
    ori_img_np = cv2.resize(ori_img_np, (self.resize, self.resize))
    ori_img_np = cv2.cvtColor(ori_img_np, cv2.COLOR_BGR2RGB)
    qp_img_np = img_utils.get_cropped_img_from_path(qp_img_path, x, y, w, h)
    qp_img_np = cv2.resize(qp_img_np, (self.resize, self.resize))
    qp_img_np = cv2.cvtColor(qp_img_np, cv2.COLOR_BGR2RGB)

    return self.tfms(ori_img_np), self.tfms(qp_img_np), label


class COCOSMRDetDataset(Dataset):

  def __init__(self, dataset_dir, dataset_type_dir_name, smr_dict_path, codec, actual_qps,
               label_T_conf=0.3, label_T_IOU=0.5, label_T_S=0.5, resize=512):
    self.resize = resize
    self.tfms = tfms

    f = open(smr_dict_path, 'rb')
    smr_dict = pickle.load(f)
    f.close()

    valid_cnt_type = f'valid_machine_cnt-{label_T_conf}-{label_T_IOU}'
    self.label_type = f'SMR-{label_T_conf}-({label_T_IOU},{label_T_S})'

    img_paths = []
    labels = []
    for seq in tqdm(smr_dict.keys()):
      if smr_dict[seq][valid_cnt_type] < 20:
        continue

      ori_img_path = '%s/%s/%s/%s.jpg' % (dataset_dir, dataset_type_dir_name, seq, seq)
      img_path_np = np.array([ori_img_path, ori_img_path]).astype(np.string_)
      img_paths.append(img_path_np)
      labels.append(1.0)

      for i in range(len(actual_qps)):
        qp = actual_qps[i]
        qp_img_path = '%s/%s/%s/%s/%d.jpg' % (dataset_dir, dataset_type_dir_name, seq, codec, qp)
        if not os.path.exists(qp_img_path):
          qp_img_path = qp_img_path.replace('jpg', 'png')

        img_path_np = np.array([ori_img_path, qp_img_path]).astype(np.string_)
        img_paths.append(img_path_np)
        labels.append(smr_dict[seq][self.label_type][i])

    self.img_paths_np = np.array(img_paths)
    self.labels_np = np.array(labels)

  def __len__(self):
    return len(self.img_paths_np)

  def __getitem__(self, index):
    img_paths = self.img_paths_np[index]
    label = self.labels_np[index]
    ori_img_path = str(img_paths[0], encoding='utf-8')
    qp_img_path = str(img_paths[1], encoding='utf-8')

    ori_img_np = cv2.imread(ori_img_path)
    ori_img_np = cv2.resize(ori_img_np, (self.resize, self.resize))
    ori_img_np = cv2.cvtColor(ori_img_np, cv2.COLOR_BGR2RGB)

    qp_img_np = cv2.imread(qp_img_path)
    qp_img_np = cv2.resize(qp_img_np, (self.resize, self.resize))
    qp_img_np = cv2.cvtColor(qp_img_np, cv2.COLOR_BGR2RGB)

    return tfms(ori_img_np), tfms(qp_img_np), label