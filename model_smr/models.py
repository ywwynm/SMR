import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import *

import pytorch_lightning as pl
from torchvision.models.feature_extraction import create_feature_extractor


# Model G
class SMRBaselineModel(pl.LightningModule):

  def __init__(self, lr, d):
    super().__init__()
    self.lr = lr

    base_model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
    extractor = create_feature_extractor(base_model, ['flatten'])
    self.extractor = extractor
    self.final = nn.Sequential(
      nn.Linear(base_model.classifier[-1].in_features * 2, d),
      nn.ReLU(True),
      nn.Linear(d, d),
      nn.ReLU(True),
      nn.Linear(d, 1)
    )


  def forward(self, x1, x2):
    f1 = self.extractor(x1)['flatten']
    f2 = self.extractor(x2)['flatten']
    fc_input = torch.cat([f1, f2], dim=1)
    return self.final(fc_input)


  def predict_step(self, batch, batch_idx, **kwargs):
    x_ori, x_qp, y = batch
    return self(x_ori, x_qp)


  def training_step(self, batch, batch_idx):
    x1, x2, y = batch
    y = y.float()
    y = y[:, None]
    y_hat = self(x1, x2)
    loss = F.l1_loss(y_hat, y)
    self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
    return loss


  def validation_step(self, batch, batch_idx):
    x1, x2, y = batch
    y = y.float()
    y = y[:, None]
    y_hat = self(x1, x2)
    loss = F.l1_loss(y_hat, y)
    self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
    return loss


  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.lr)


# Model Q
class SMRDiffBasedModel(pl.LightningModule):

  def __init__(self, lr, d):
    super().__init__()
    self.lr = lr

    base_model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
    extractor = create_feature_extractor(base_model, ['flatten'])
    self.extractor = extractor
    self.final = nn.Sequential(
      nn.Linear(base_model.classifier[-1].in_features * 2, d),
      nn.ReLU(True),
      nn.Linear(d, d),
      nn.ReLU(True),
      nn.Linear(d, 1)
    )


  def forward(self, x1, x2):
    f1 = self.extractor(x1)['flatten']
    f2 = self.extractor(x2)['flatten']
    fc_input = torch.cat([f1, f2], dim=1)
    return 1 - self.final(fc_input)


  def predict_step(self, batch, batch_idx, **kwargs):
    x_ori, x_qp, y = batch
    return self(x_ori, x_qp)


  def training_step(self, batch, batch_idx):
    x1, x2, y = batch
    y = y.float()
    y = y[:, None]
    y_hat = self(x1, x2)
    loss = F.l1_loss(y_hat, y)
    self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
    return loss


  def validation_step(self, batch, batch_idx):
    x1, x2, y = batch
    y = y.float()
    y = y[:, None]
    y_hat = self(x1, x2)
    loss = F.l1_loss(1 - y_hat, y)
    self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
    return loss


  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.lr)