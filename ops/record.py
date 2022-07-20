import torch.utils.data as data
from PIL import Image
import os
import torch
import numpy as np
from numpy.random import randint
import cv2
import json
import decord
from decord import VideoReader
from decord import cpu, gpu
import torchvision
import gc
import math
decord.bridge.set_bridge('torch')
from torchvision.io import read_video,read_video_timestamps

class VideoRecord_KineticsZSAR(object):
    def __init__(self, dic, name, root_path):
        self._data = dic
        self._root_path = root_path
        self._path = os.path.join(root_path,name)
        self.trans = torchvision.transforms.ToPILImage(mode='RGB')
        self.zip_rat = 3
        if "$Neg" in self._path:
            self.neg = True
            self._path = name.split("$Neg")[0]
        else:
            self.neg = False

    def frames(self):
        try:
            vr = VideoReader(self._path)
        except:
            print(self._path)
        if len(vr)==0:
            raise RuntimeError("Can't load video {} correctly".format(self._path))
        num_frames = len(vr)
        images = []
        if num_frames<100:
            for i in range(num_frames):
                img = vr[i].permute(2,0,1)
                images.append(self.trans(img).convert('RGB'))
        else:
            zip_rat = num_frames // 64
            for i in range(0,num_frames,zip_rat):
                img = vr[i].permute(2,0,1)
                images.append(self.trans(img).convert('RGB'))
            images = images[:64]

        del(vr)
        # self._data["nframes"] = len(images)
        return images

    @property
    def path(self):
        return self._path

    @property
    def name(self):
        return self._path.split("/")[-1].split(".")[0]

    @property
    def num_frames(self):
        if self._data["nframes"]<100:
            return self._data["nframes"]
        else:
            return 64


    @property
    def label(self):
        return self._data["label_id"]

    @property
    def target(self):
        if self.neg:
            return -1
        else:
            return 1

    @property
    def text(self):
        return self._data["label"]

    @property
    def start(self):
        return 0
