# Few-Shot Learning applied to Speech Emotion Recognition (SER)

This repository contemplate a study on Few-Shot applied to SER task for the Representation Learning course at University of Campinas (2021).

## Introduction

Few-shot learning is an open challenge in many tasks. It
is particularly important in the speech emotion recognition context due to the costly recording and annotation of emotion labels
in speech signals. Several techniques were proposed in order to
deal with few-shot problems, data augmentation, transfer learning, contrastive learning and meta-learning. This work focuses
on meta-learning strategies to build a few-shot emotion classifier. In particular, we performed experiments based on ModelAgnostic Meta-Learning (MAML) algorithm augmented with
both Multi-Step Loss and Contrastive loss. Our results show
that both augmentations have the potential to improve MAML
performance. However, we observed that MAML is easy to
overfit, performing very well in training but not generalizing
for the out-of-sample test set. Despite that, several works have
presented similar behaviour trying to apply MAML in image
classification. Consequently, many improvements were proposed over the years to stabilize MAML training and help fastadaptation performance.

## Datasets

- Emofilm dataset (you must require it in https://zenodo.org/record/1326428#.X8sFStgzZPY)


## How to run

Due to lack of GPU's all our runs were performed using google colab. The only necessary packages are pyworlds, faiss-gpu and pytorch metric learning.

```
# Install pyworld for melspec extraction and pytorch metric learning for contrastive loss
!pip install pyworld
!pip install pytorch-metric-learning
!pip install faiss-gpu
```

All other default packages used are default colab environment.

To run any experiment, set your local path and training and testing csv files locating wav_path and emotion label in /src/bin/train_x.py files.


## Authors
- Lucas Hideki Ueda (lucashueda@gmail.com)
- Leonardo B. de M. M. Marques (leonardoboulitreau@gmail.com)

## Github references

- https://github.com/dragen1860/MAML-Pytorch
- https://github.com/AnugunjNaman/Fixed-MAML
- https://github.com/mozilla/TTS