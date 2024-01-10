# SMR
The code of our paper [Perceptual Video Coding for Machines via Satisfied Machine Ratio Modeling](https://arxiv.org/abs/2211.06797).

Video Coding for Machines (VCM) aims to compress visual signals for machine analysis. However, existing methods only consider a few machines, neglecting the majority. Moreover, the machine's perceptual characteristics are not leveraged effectively, resulting in suboptimal compression efficiency. To overcome these limitations, this paper introduces Satisfied Machine Ratio (SMR), a metric that statistically evaluates the perceptual quality of compressed images and videos for machines by aggregating satisfaction scores from them. Each score is derived from machine perceptual differences between original and compressed images. Targeting image classification and object detection tasks, we build two representative machine libraries for SMR annotation and create a large-scale SMR dataset to facilitate SMR studies. We then propose an SMR prediction model based on the correlation between deep feature differences and SMR. Furthermore, we introduce an auxiliary task to increase the prediction accuracy by predicting the SMR difference between two images in different quality. Extensive experiments demonstrate that SMR models significantly improve compression performance for machines and exhibit robust generalizability on unseen machines, codecs, datasets, and frame types. SMR enables perceptual coding for machines and propels VCM from specificity to generality.

## Requirements

Easy.

**OS**: Windows, Linux.

**PyTorch+CUDA**: any version should be OK.

**Other requirements**:

```
pip install pytorch-lightning opencv-python tqdm
```

Some models are trained on `pytorch-lightning==1.7.4`. However, they should work well on newer versions (tested on 2.1.0).

## Predict

```
# Change paths in this file at first.
python evaluate_smr/predict.py
```

## Train

```
# Change paths in this file at first.
python model_smr/train.py
```

## TODO

- [ ] Upload pre-trained model weights (~5.36GB).
- [ ] Upload SMR annotations (~9.24GB).
- [ ] Upload SMR dataset (>600GB).
- [ ] Add the reference code for compressing dataset.
- [ ] Add the reference code for annotating SMR using constructed machine libraries.
- [ ] Add the code for evaluation.

