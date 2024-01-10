import os
import time

from datasets import COCOSMRClsDataset, COCOSMRDetDataset
from models import SMRBaselineModel, SMRDiffBasedModel

import utils

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


def worker_init_task_affinity(worker_id):
  os.system('taskset -p 0xffffffff %d > /dev/null' % os.getpid())


def train_cls(ckpt_path=None):
  dataset_dir = '/path/to/coco_smr/'
  actual_qps = [11, 13, 15, 17, 19] + list(range(21, 52))

  dataset_type_dir_name = 'train2017'
  smr_dict_path = '../assets/smr_annotations_coco_hevc_cls_train.pkl'
  codec = 'hevc'
  ds_train = COCOSMRClsDataset(dataset_dir, dataset_type_dir_name, smr_dict_path, codec, actual_qps, label_type='SMR-top1')

  dataset_type_dir_name = 'val2017'
  smr_dict_path = '../assets/smr_annotations_coco_hevc_cls_val.pkl'
  codec = 'hevc'
  ds_val = COCOSMRClsDataset(dataset_dir, dataset_type_dir_name, smr_dict_path, codec, actual_qps, label_type='SMR-top1')

  # batch size 92 is an appropriate batch size that can be fit into 24GB GPU memory when the precision is "16-mixed"
  train_loader = DataLoader(ds_train, num_workers=14, batch_size=92, shuffle=False, pin_memory=True, worker_init_fn=worker_init_task_affinity)
  val_loader = DataLoader(ds_val, num_workers=14, batch_size=92, shuffle=False, pin_memory=False, worker_init_fn=worker_init_task_affinity)

  # SMRDiffBasedModel can be loaded in the same way
  if ckpt_path is None:
    model = SMRBaselineModel(lr=1e-4, d=4096)
  else:
    model = SMRBaselineModel.load_from_checkpoint(ckpt_path, lr=1e-4, d=4096)

  lr_monitor = LearningRateMonitor(logging_interval='epoch')
  ckpt_callback = ModelCheckpoint(save_top_k=2, monitor='val_loss')
  trainer = pl.Trainer(strategy='ddp_find_unused_parameters_false', accelerator='gpu', devices=[0, 1, 2],
                       precision='16-mixed', max_epochs=-1, check_val_every_n_epoch=1, callbacks=[lr_monitor, ckpt_callback])
  trainer.fit(model, train_loader, val_loader)


def train_det(ckpt_path=None):
  dataset_dir = '/path/to/coco_smr/'
  actual_qps = [11, 13, 15, 17, 19] + list(range(21, 52))

  T_conf = 0.3
  T_IOU = 0.5
  T_S = 0.5

  dataset_type_dir_name = 'train2017'
  smr_dict_path = '../assets/smr_annotations_coco_hevc_det_train.pkl'
  codec = 'hevc'
  ds_train = COCOSMRDetDataset(dataset_dir, dataset_type_dir_name, smr_dict_path, codec, actual_qps,
                               label_T_conf=T_conf, label_T_IOU=T_IOU, label_T_S=T_S)

  dataset_type_dir_name = 'val2017'
  smr_dict_path = '../assets/smr_annotations_coco_hevc_det_val.pkl'
  codec = 'hevc'
  ds_val = COCOSMRDetDataset(dataset_dir, dataset_type_dir_name, smr_dict_path, codec, actual_qps,
                             label_T_conf=T_conf, label_T_IOU=T_IOU, label_T_S=T_S)

  # batch size 18 is an appropriate batch size that can be fit into 24GB GPU memory when the precision is "16-mixed", because the input resolution is 512*512 for object detection
  train_loader = DataLoader(ds_train, num_workers=14, batch_size=18, shuffle=False, pin_memory=True,
                            worker_init_fn=worker_init_task_affinity)
  val_loader = DataLoader(ds_val, num_workers=14, batch_size=18, shuffle=False, pin_memory=False,
                          worker_init_fn=worker_init_task_affinity)

  # SMRDiffBasedModel can be loaded in the same way
  # There is one exception: When using our trained model Q to predict SMR for (T_IOU=0.5, T_S=0.5), d=3072 is used instead of d=4096
  # For other SMR types and model G, d=4096 is used consistently
  if ckpt_path is None:
    model = SMRBaselineModel(lr=1e-4, d=4096)
  else:
    model = SMRBaselineModel.load_from_checkpoint(ckpt_path, lr=1e-4, d=4096)

  lr_monitor = LearningRateMonitor(logging_interval='epoch')
  ckpt_callback = ModelCheckpoint(save_top_k=2, monitor='val_loss')
  trainer = pl.Trainer(strategy='ddp_find_unused_parameters_false', accelerator='gpu', devices=[0, 1, 2],
                       precision='16-mixed', max_epochs=-1, check_val_every_n_epoch=1,
                       callbacks=[lr_monitor, ckpt_callback])
  trainer.fit(model, train_loader, val_loader)


def main():
  start_time = time.time()
  start_time_str = utils.now_str()
  print('The program starts at %s' % start_time_str)

  train_cls()
  # train_det()

  print('The program starts at %s, and ends at %s, time cost: %.2fs' %
        (start_time_str, utils.now_str(), time.time() - start_time))


if __name__ == '__main__':
  main()
