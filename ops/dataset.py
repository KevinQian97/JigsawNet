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
from ops.record import VideoRecord_KineticsZSAR as VideoRecord
decord.bridge.set_bridge('torch')


def collate_fn(batch,if_glove = False,if_clip=False):
    """
    need to modify collate_fn since the size of act_tokens are variable
    input:
    list of tuples [(vision,act_tokens,obj_tokens,obj_clf_tokens,act_clf_label,obj_clf_labels)]
    vision: tensor(video_candidates X 3 X num_segments X H X W)
    act_tokens: {'input_ids':tensor(M X N), 'token_type_ids':tensor(M X N), 'attention_mask':tensor(M X N)}
    obj_tokens: {'input_ids':tensor(video_candidates X N), 'token_type_ids':tensor(video_candidates X N), 'attention_mask':tensor(video_candidates X N)}
    obj_clf_tokens: {'input_ids':tensor(20 X N), 'token_type_ids':tensor(20 X N), 'attention_mask':tensor(20 X N)}
    act_clf_label: int
    obj_clf_labels: tensor(20)

    output:
    vids: tensor(B*video_candidates X 3 X num_segments X H X W)
    texts: {'input_ids':tensor(sum(B,M) X N), 'token_type_ids':tensor(sum(B,M) X N), 'attention_mask':tensor(sum(B,M) X N)}
    indices: tensor(B)
    objs: {'input_ids':tensor(B*video_candidates X N), 'token_type_ids':tensor(B*video_candidates X N), 'attention_mask':tensor(B*video_candidates X N)}
    obj_clf: {'input_ids':tensor(B*20 X N), 'token_type_ids':tensor(B*20 X N), 'attention_mask':tensor(B*20 X N)}
    act_label: tensor(B)
    obj_label: tensor(B X 20)
    """
    vision,act_tokens,obj_tokens,obj_clf_tokens,act_clf_label,obj_clf_labels,frame_ids = zip(*batch)
    if if_clip:
        vids = []
        for vid in vision:
            vids.extend(vid.chunk(vid.size(1),1))
        vids = torch.stack(vids,0)
        vids = vids.squeeze()
    else:
        if len(vision[0].size())==5:
            vids = torch.cat(vision,0)
        else:
            vids = torch.stack(vision,0)
    if if_glove:
        texts = torch.cat(act_tokens,0)
        indices = []
        for act_token in act_tokens:
            indices.append(act_token.size(0))
        indices = torch.tensor(indices)
    else:
        keys = list(act_tokens[0].keys())
        texts = {}
        for key in keys:
            texts[key] = torch.cat([act_token[key] for act_token in act_tokens],0)
        indices = []
        for act_token in act_tokens:
            indices.append(act_token[keys[0]].size(0))
        indices = torch.tensor(indices)

    objs = {}
    keys = list(obj_tokens[0].keys())

    if len(obj_tokens[0][keys[0]].size())==2:
        for key in keys:
            objs[key] = torch.cat([obj_token[key] for obj_token in obj_tokens],0)
    else:
        for key in keys:
            objs[key] = torch.stack([obj_token[key] for obj_token in obj_tokens],0)

    obj_clf = {}
    for key in keys:
        obj_clf[key] = torch.cat([obj_clf_token[key] for obj_clf_token in obj_clf_tokens],0)
    
    act_label = torch.tensor([label for label in act_clf_label])
    obj_label = torch.stack([label for label in obj_clf_labels])
    
    frame_ids = torch.stack(frame_ids,0)
    return vids,texts,objs,obj_clf,indices,act_label,obj_label,frame_ids



class ZSARDataset(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 transform=None,
                 random_shift=True, 
                 test_mode=False,
                 remove_missing=False,  
                 video_candidates = 4,
                 if_attn = False,
                 video_path = ""):

        self.root_path = root_path
        self.video_path = video_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.obj_tokens = torch.load(os.path.join(root_path,"obj_tokens.pt"))
        self.act_tokens = torch.load(os.path.join(root_path,"act_tokens.pt"))
        self.video_candidates = video_candidates
        self.if_attn = if_attn
        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff
        self._parse_list()
        

# also modify it to video loader
    def _load_image(self, frames, directory, idx, start):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                if idx >= len(frames):
                    print("exceeding frame access happened")
                    idx = len(frames)-1
                return [frames[idx]]
            except Exception:
                print('error loading video:{} whose idx is {}'.format(os.path.join(self.root_path, directory),start+idx))
                return [frames[0]]

# parse_list for json input
    def _parse_list(self):
        tmp = json.load(open(self.list_file,"r"))["database"]
        items = tmp.keys()
        self.video_list = [VideoRecord(tmp[item],item,self.video_path) for item in items]
        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        if self.video_candidates > 1:
            average_clip = record.num_frames // self.video_candidates
            average_duration = average_clip // self.num_segments
            if average_duration> 0:
                offsets = np.multiply(list(range(self.num_segments*self.video_candidates)),average_duration)+ randint(average_duration,size=self.num_segments*self.video_candidates)
            elif record.num_frames > self.num_segments * self.video_candidates:
                offsets = np.sort(randint(record.num_frames,size=self.num_segments*self.video_candidates))
            else:
                offsets = np.zeros((self.num_segments*self.video_candidates))
            return offsets

        else:  # normal sample
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments))
            return offsets

    def _get_val_indices(self, record):
        if self.video_candidates >1:
            average_clip = record.num_frames // self.video_candidates
            average_duration = average_clip // self.num_segments
            if average_duration> 0:
                offsets = np.multiply(list(range(self.num_segments*self.video_candidates)),average_duration)+ randint(average_duration,size=self.num_segments*self.video_candidates)
            elif record.num_frames > self.num_segments * self.video_candidates:
                offsets = np.sort(randint(record.num_frames,size=self.num_segments*self.video_candidates))
            else:
                offsets = np.zeros((self.num_segments*self.video_candidates))
            return offsets
        else:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets

    def _get_test_indices(self, record):
        if self.video_candidates >1:
            average_clip = record.num_frames // self.video_candidates
            average_duration = average_clip // self.num_segments
            if average_duration> 0:
                offsets = np.multiply(list(range(self.num_segments*self.video_candidates)),average_duration)+ randint(average_duration,size=self.num_segments*self.video_candidates)
            elif record.num_frames > self.num_segments * self.video_candidates:
                offsets = np.sort(randint(record.num_frames,size=self.num_segments*self.video_candidates))
            else:
                offsets = np.zeros((self.num_segments*self.video_candidates))
            return offsets
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x)-1 for x in range(self.num_segments)])
            return offsets

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)


# get for json input
    def get(self, record, indices):
        # print(indices)
        images = list()
        frames = record.frames()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(frames,record.name,p,record.start)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1
        vision = self.transform(images)
        if self.video_candidates>1:
            vision = vision.chunk(self.video_candidates,1)
            vision = torch.stack(vision,0)

        if record.name in self.obj_tokens:
            obj_tokens = self.obj_tokens[record.name]["tokens"]
            obj_clf_tokens = self.obj_tokens[record.name]["obj_clf_tokens"]
            obj_clf_labels = self.obj_tokens[record.name]["obj_clf_labels"]
        elif record.path in self.obj_tokens:
            obj_tokens = self.obj_tokens[record.path]["tokens"]
            obj_clf_tokens = self.obj_tokens[record.path]["obj_clf_tokens"]
            obj_clf_labels = self.obj_tokens[record.path]["obj_clf_labels"]
        if self.video_candidates>1:
            for k,v in obj_tokens.items():
                obj_tokens[k] = v.repeat(self.video_candidates,1)
        elif self.video_candidates==0:
            for k,v in obj_tokens.items():
                if len(obj_tokens[k].size())==1:
                    obj_tokens[k] = v.repeat(self.num_segments,1)
        elif not self.if_attn:
            for k,v in obj_tokens.items():
                obj_tokens[k] = v.repeat(4,1)
        act_tokens = self.act_tokens[record.text]
        act_clf_label = record.label
        
        return vision,act_tokens,obj_tokens,obj_clf_tokens,act_clf_label,obj_clf_labels,torch.tensor(indices)

    def __len__(self):
        return len(self.video_list)
