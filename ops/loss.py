import torch
from torch import nn
import torch.nn.functional as F
import json
class MultiCosLoss(nn.Module):
    def __init__(self,):
        super(MultiCosLoss,self).__init__()
        self.pool = nn.AdaptiveMaxPool1d(1)
    
    def forward(self,vision_feats,text_feats):
        assert len(vision_feats)==len(text_feats)
        loss = torch.tensor(0.0,requires_grad=True)
        for v,t in zip(vision_feats,text_feats):
            sim = F.normalize(v.squeeze()).mm(F.normalize(t).t())
            sim = self.pool(sim.unsqueeze(0))
            loss = loss + 1-torch.mean(sim.squeeze())
        loss = loss/len(vision_feats)
        return loss


class MultiCosLosswithCache(nn.Module):
    def __init__(self,):
        super(MultiCosLoss,self).__init__(cache)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.cache = cache
    
    def forward(self,vision_feats,text_feats):
        assert len(vision_feats)==len(text_feats)
        loss = torch.tensor(0.0,requires_grad=True)
        for v,t in zip(vision_feats,text_feats):
            sim = F.normalize(v.squeeze()).mm(F.normalize(t).t())
            sim = self.pool(sim.unsqueeze(0))
            loss = loss + 1-torch.mean(sim.squeeze())
        loss = loss/len(vision_feats)
        return loss
        


class WeightedBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self,class_weight,pos_weight):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.weight = class_weight
        self.pos_weight = pos_weight

    def forward(self, input, target):
        binary_target = (target > 0).type(target.dtype)
        weight = target + 1 - binary_target
        if self.weight is not None:
            weight *= self.weight
        # return F.binary_cross_entropy_with_logits(
        #     input, binary_target, weight, pos_weight=self.pos_weight,
        #     reduction=self.reduction)
        return F.binary_cross_entropy_with_logits(
            input, binary_target,weight, pos_weight=self.pos_weight,
            reduction=self.reduction)

def prep_weight(train_list):
    js = json.load(open(train_list,"r"))
    database = js["database"]
    labels = js["labels"]
    props = list(database.keys())
    pos_score = [0]*(len(labels)-1)
    neg_score = [0]*(len(labels)-1)
    num_score = [0]*(len(labels)-1)
    for prop in props:
        for i in range(len(database[prop]["annotations"]["conf"])):
            pos = database[prop]["annotations"]["conf"][i]
            pos_score[i] += pos
            if pos == 0:
                neg_score[i]+=1
            if pos >0:
                num_score[i]+=1
    sum_class = len(props)

    class_weight = []
    for i in neg_score:
        if i ==0:
            class_weight.append(0)
        else:
            class_weight.append(sum_class/(2*i))
    class_weight = torch.Tensor(class_weight).cuda()
    class_weight = class_weight / class_weight.sum() * class_weight.shape[0]
    pos_weight = []
    for i,j in zip(pos_score,neg_score):
        if i==0:
            pos_weight.append(0)
        else:
            pos_weight.append(j/i)
    pos_weight = torch.Tensor(pos_weight).cuda()
    # pos_weight = torch.Tensor([j/max(1,i) for i,j in zip(pos_score,neg_score)]).cuda()
    pos_weight = torch.clamp(pos_weight,0,100)
    print(class_weight)
    print(pos_weight)
    # pos_weight -= pos_weight.min()
    # pos_weight /= pos_weight.max()
    return class_weight,pos_weight