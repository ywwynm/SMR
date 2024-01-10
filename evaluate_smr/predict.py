import time

import utils

import cv2

from model_smr.datasets import tfms
from model_smr.models import SMRBaselineModel, SMRDiffBasedModel


def predict(ckpt_path=None, device='cuda:0'):
  model = SMRBaselineModel.load_from_checkpoint(ckpt_path, lr=1e-4, d=4096)
  model.eval()
  model.to(device)

  ori_img_path = '../assets/sample_imgs/raw.jpg'
  qp_img_path = '../assets/sample_imgs/qp42.jpg'

  ori_img_np = cv2.imread(ori_img_path)
  ori_img_np = cv2.resize(ori_img_np, (224, 224))
  ori_img_np = cv2.cvtColor(ori_img_np, cv2.COLOR_BGR2RGB)

  qp_img_np = cv2.imread(qp_img_path)
  qp_img_np = cv2.resize(qp_img_np, (224, 224))
  qp_img_np = cv2.cvtColor(qp_img_np, cv2.COLOR_BGR2RGB)

  ori_img_tensor = tfms(ori_img_np).unsqueeze(0).to(device)
  qp_img_tensor = tfms(qp_img_np).unsqueeze(0).to(device)
  smr_predicted = model(ori_img_tensor, qp_img_tensor).detach().cpu().numpy()
  print(smr_predicted)  # For the sample image, SMR values: QP 30 -> 0.66450727, QP 42 -> 0.27713755, QP 48 -> 0.13697474


def main():
  start_time = time.time()
  start_time_str = utils.now_str()
  print('The program starts at %s' % start_time_str)

  predict(ckpt_path='../assets/smr_model_weights/cls_top1_modelG_mae0.0463.ckpt')

  print('The program starts at %s, and ends at %s, time cost: %.2fs' %
        (start_time_str, utils.now_str(), time.time() - start_time))


if __name__ == '__main__':
  main()

