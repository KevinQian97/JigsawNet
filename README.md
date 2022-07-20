# JigsawNet
Authors: Yijun Qian, Lijun Yu, Wenhe Liu, Alexander G Hauptmann

Email: yijunqia@andrew.cmu.edu

## Overview
This is the official implementation of Rethinking Zero-shot Action Recognition: Learning from Latent Atomic Actions (ECCV22). For more information please refer to our accepted paper in ECCV 2022.

## Prerequisites
The code is built with many libraries.
If you encounter problems about the dependencies, please resort to their official sites for help.
We have prepared the environment config file and suggest build the environment through ANACONDA.
```
cd JigsawNet
conda env create -f environment.yml 
conda activate zsl
```

## Data Preparation
To run our code, please download Kinetics Dataset and place `Kinetics400` and `Kinetics600` under the same `$Data Path`.
Make sure the data structure is like:
* $Data Path
  * Kinetics400
      * train
      * validate
      * test
  * Kinetics600
      * videos
          * trn
          * val
          * tst
Meanwhile, please download the annotation folder and place them under `./datasets`
[KineticsZSAR-TRAIN]()
[KineticsZSAR-TEST]()

## Inference
To quickly evaluate our model, please download the [weight]().
Then follow these steps:
```
mkdir logs
mkdir exps
python eval.py --dataset KEVAL \
     --arch R2plus1D-34 --text_model bert \
     --epochs 1 --batch-size 48 -j 48 --dropout 0.5 --gpus 0 1 2 3 \
     --npb --lr_scheduler --vmz_tune_last_k_layer 4 \
     --freeze_text_to 10 --bert_pooling avg --consist_loss --attn \
     --root_model ./exps --root_log ./logs --video_path $Data Path \
     --tune_from $PATH-TO-MODEL
```

## Training
To train the model, please follow these steps:
```
mkdir logs
mkdir exps
python train.py --dataset KineticsZSAR \
     --arch R2plus1D-34 --text_model bert \
     --gd 20 --lr 1e-5 --lr_steps 2 6 9 12  --epochs 20 \
     --batch-size 28 -j 28 --dropout 0.5 --gpus 0 1 2 3 \
     --root_model ./exps --root_log ./logs --video_path /mnt/data \
     --npb --lr_scheduler --vmz_tune_last_k_layer 4 \
     --freeze_text_to 10 --bert_pooling avg  --attn --consist_loss
```

## License
Please read the `LICENSE` before using.