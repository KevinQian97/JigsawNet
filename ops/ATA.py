from torch import nn

from ops.basic_ops import ConsensusModule
from ops.transforms import *
from torch.nn.init import normal_, constant_
import os
import copy
from transformers import AutoTokenizer,AutoModel,BertModel,GPT2Model,BartModel,CLIPVisionModel,CLIPTextModel,CLIPModel
from torch.nn import functional as F

def get_fine_tuning_parameters(model, args):
    r21d34_max_index = 4
    vit_max_index = 11
    bert_max_index = 11
    gpt_max_index = 11
    bart_max_index = 5

    tune_last_k_layer = args.vmz_tune_last_k_layer
    freeze_text_to = args.freeze_text_to
    
    if args.arch != "clip":
        ft_begin_index = r21d34_max_index - tune_last_k_layer + 1
    else:
        ft_begin_index = vit_max_index - tune_last_k_layer + 1

    ft_module_names = []
    ft_module_names.append("module.act_gate")
    ft_module_names.append("module.obj_gate")
    ft_module_names.append("module.attn_model")
    ft_module_names.append("module.act_model.1")
    ft_module_names.append("module.act_model.3")
    ft_module_names.append("module.base_model.vision_model.embeddings")
    ft_module_names.append("module.base_model.vision_model.pre_layrnorm")
    ft_module_names.append("module.base_model.vision_model.post_layernorm")
    if args.arch != "clip":
        for i in range(ft_begin_index, r21d34_max_index + 1):
            ft_module_names.append('module.base_model.layer{}'.format(i))
            ft_module_names.append('module.base_model.stem.{}'.format(i))
    else:
        for i in range(ft_begin_index, vit_max_index + 1):
            ft_module_names.append('module.base_model.vision_model.encoder.layers.{}'.format(i))

    if "bert" in args.text_pretrain:
        for i in range(freeze_text_to,bert_max_index+1):
            ft_module_names.append("module.text_model.encoder.layer.{}".format(i))
        ft_module_names.append("module.text_model.pooler")
    elif "gpt" in args.text_pretrain:
        for i in range(freeze_text_to,gpt_max_index+1):
            ft_module_names.append("module.text_model.h.{}".format(i))
        ft_module_names.append("module.text_model.ln_f")
    elif "bart" in args.text_pretrain:
        for i in range(freeze_text_to,bart_max_index+1):
            ft_module_names.append("module.text_model.decoder.layers.{}".format(i))
        ft_module_names.append("module.text_model.decoder.layernorm_embedding")
    elif "openai" in args.text_pretrain:
        for i in range(freeze_text_to,bert_max_index+1):
            ft_module_names.append("module.text_model.text_model.encoder.layers.{}".format(i))
        ft_module_names.append("module.text_model.text_model.final_layer_norm")
        ft_module_names.append("module.text_proj")
        ft_module_names.append("module.text_model.text_model.embeddings")
    if args.dropout > 0:    
        ft_module_names.append('module.new_fc')
        ft_module_names.append("module.ft_attn")
    else:
        ft_module_names.append("module.base_model.fc")
    

    parameters = []
    freeze = []
    tune = []
    names = []
    # for k, v in model.named_parameters():
    #     print(k)
    # raise RuntimeError("stop")
    for k, v in model.named_parameters():
        names.append(k)
        no_grad = True
        for ft_module in ft_module_names:
            if k.startswith(ft_module):
                parameters.append({'params': v})
                tune.append(k)
                no_grad = False
                break
        if no_grad:
            v.requires_grad = False
            freeze.append(k)
    print('fine_tune:', len(tune), tune)
    print('freeze', len(freeze), freeze)
    print('all', len(names))
    print('param', len(parameters))
    return parameters

class ZSAR(nn.Module):
    def __init__(self, num_class, num_segments,base_model='resnet101', 
    dropout=0.8, feature_dim=768, modality = "RGB",text_feature_dim = 768,
    partial_bn=False, print_spec=True, pretrain='imagenet',
    fc_lr5=False,
    cfg_file = None,
    text_pretrain = None,
    bert_pooling = "avg",
    video_candidates = 4,
    attn = False,
    nhead=8,
    num_layers = 1,
    cache = False,
    emb_path = None
    ):
        super(ZSAR, self).__init__()
        self.num_segments = num_segments
        self.reshape = True
        self.dropout = dropout
        self.feature_dim = feature_dim  # the dimension of the CNN feature to represent each frame
        self.pretrain = pretrain
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.cfg_file = cfg_file
        self.bert_pooling = bert_pooling
        self.video_candidates = video_candidates
        self.text_pretrain = text_pretrain
        self.attn = attn
        self.emb_path = emb_path

        if "gpt" in text_pretrain:
            self.text_model = GPT2Model.from_pretrained(text_pretrain)
        elif "bart" in text_pretrain:
            self.text_model = BartModel.from_pretrained("facebook/bart-base")
        elif "glove" in text_pretrain:
            embs = np.load(self.emb_path)
            self.act_model = nn.Sequential(
                nn.Embedding.from_pretrained(torch.from_numpy(embs).float()),
                nn.Linear(300,self.feature_dim),
                nn.ReLU(),
                # nn.TransformerEncoderLayer(nhead=nhead,d_model=self.feature_dim)
            )
            self.ft_attn = nn.Sequential(
            nn.Linear(self.feature_dim,self.feature_dim//2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.feature_dim//2,1)
            )
            self.text_model =  BertModel.from_pretrained("bert-base-uncased")
        elif "bert" in text_pretrain:
            self.text_model = BertModel.from_pretrained(text_pretrain)
        elif "openai" in text_pretrain:
            self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            self.text_proj = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").text_projection
        if self.bert_pooling == "attn":
            self.ft_attn = nn.Sequential(
            nn.Linear(self.feature_dim,self.feature_dim//2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.feature_dim//2,1)
            )

        if self.attn:
            self.attn_model = nn.TransformerEncoderLayer(nhead=nhead,d_model=self.feature_dim)
            # layers = nn.TransformerEncoderLayer(nhead=nhead,d_model=self.feature_dim)
            # self.attn_model = nn.TransformerEncoder(layers,num_layers=num_layers)
        self.act_gate_fc = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, self.feature_dim), nn.Sigmoid())
        self.obj_gate_fc = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, self.feature_dim), nn.Sigmoid())
        self.modality = modality

        if print_spec:
            print(("""
            Initializing ZSAR with base model: {}.
            Vision Configurations:
            num_segments:       {}
            dropout_ratio:      {}
            feature_dim:    {}
            video_candidates: {}
            text_pretrain: {}
            """.format(self.base_model_name, self.num_segments, 
            self.dropout, self.feature_dim,self.video_candidates,
            self.text_pretrain)))

        self._prepare_base_model(self.base_model_name)

        feature_dim = self._prepare_clf(num_class)

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_clf(self, num_class):
        if self.base_model_name == "X3D":
            feature_dim = getattr(self.base_model.head, self.base_model.last_layer_name).in_features
        elif self.base_model_name == "clip":
            feature_dim = 768
        else:
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            if self.base_model_name == "X3D":
                setattr(self.base_model.head, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            elif self.base_model_name != "clip":
                setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            if self.base_model_name == "X3D":
                setattr(self.base_model.head, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            elif self.base_model_name == "clip":
                self.vision_fc = nn.Dropout(p=self.dropout)
            else:
                setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
           
            self.new_fc = nn.Linear(feature_dim,num_class)

        std = 0.001
        if self.new_fc is None:
            if self.base_model_name == "X3D":
                normal_(getattr(self.base_model.head, self.base_model.last_layer_name).weight, 0, std)
                constant_(getattr(self.base_model.head, self.base_model.last_layer_name).bias, 0)
            elif self.base_model_name != "clip":
                normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
                constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))

        if 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True if self.pretrain == 'imagenet' else False)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
                    
        elif base_model == "R2plus1D-34":
            from archs.R2plus1D import r2plus1d_34
            if "kinetics" in self.pretrain:
                self.base_model = r2plus1d_34(self.pretrain,num_classes=400)
            elif "ig65m" in self.pretrain:
                if "_8_" in self.pretrain:
                    self.base_model = r2plus1d_34(self.pretrain,num_classes=487)
                else:
                    self.base_model = r2plus1d_34(self.pretrain,num_classes=359)
            else:
                self.base_model = r2plus1d_34(self.pretrain)

            self.input_size = 112
            self.input_mean =[0.43216, 0.394666, 0.37645]
            self.input_std = [0.22803, 0.22145, 0.216989]
            self.base_model.last_layer_name = 'fc'

        elif base_model == "R2plus1D-152":
            from archs.R2plus1D import r2plus1d_152
            if "kinetics" in self.pretrain:
                self.base_model = r2plus1d_152(self.pretrain,num_classes=400)
            elif "ig65m" in self.pretrain:
                self.base_model = r2plus1d_152(self.pretrain,num_classes=359)
            else:
                self.base_model = r2plus1d_152(self.pretrain)
                
            self.input_size = 112
            self.input_mean =[0.43216, 0.394666, 0.37645]
            self.input_std = [0.22803, 0.22145, 0.216989]
            self.base_model.last_layer_name = 'fc'
        
        elif base_model == "IP-CSN":
            from archs.CSN import ip_csn_152
            if "kinetics" in self.pretrain:
                self.base_model = ip_csn_152(self.pretrain,num_classes=400)
            elif "sports1m" in self.pretrain:
                self.base_model = ip_csn_152(self.pretrain,num_classes=487)
            elif "ig65m" in self.pretrain:
                self.base_model = ip_csn_152(self.pretrain,num_classes=359)
            else:
                self.base_model = ip_csn_152()
            self.input_size = 224
            self.input_mean =[0.43216, 0.394666, 0.37645]
            self.input_std = [0.22803, 0.22145, 0.216989]
            self.base_model.last_layer_name = 'fc'
        
        elif base_model == "IR-CSN":
            from archs.CSN import ir_csn_152
            if "kinetics" in self.pretrain:
                self.base_model = ir_csn_152(self.pretrain,num_classes=400)
            elif "sports1m" in self.pretrain:
                self.base_model = ir_csn_152(self.pretrain,num_classes=487)
            elif "ig65m" in self.pretrain:
                self.base_model = ir_csn_152(self.pretrain,num_classes=359)
            else:
                self.base_model = ir_csn_152()
            self.input_size = 224
            self.input_mean =[0.43216, 0.394666, 0.37645]
            self.input_std = [0.22803, 0.22145, 0.216989]
            self.base_model.last_layer_name = 'fc'

        elif base_model == "X3D":
            from archs.X3D import X3D,build_model
            self.base_model = build_model(self.cfg_file)
            checkpoint = torch.load("/data/yijunq/models/TSA/x3d_l.pyth")
            self.base_model.load_state_dict(checkpoint["model_state"])
            self.input_size = 312
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            self.base_model.last_layer_name = 'projection'

        elif base_model == "clip":
            self.base_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            self.vision_fc = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").visual_projection
            self.input_size = 224
            self.input_mean = [0.48145466,0.4578275,0.40821073]
            self.input_std = [0.26862954,0.26130258,0.27577711]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def init_cache(self,act_mat):
        if "glove" in self.text_pretrain:
            act_emb = self.act_model(act_mat)
            num_words = act_emb.size(1)
            act_emb = act_emb.mean(1)
            act_emb = F.normalize(act_emb,dim=1,p=2)
            print("Num of Atomic Actions:{}".format(act_emb.size(0)))
            print(act_emb.size())
            self.act_cache = act_emb.data
        else:
            act_emb = self.text_model(**act_mat)["last_hidden_state"]
            masks = act_mat["attention_mask"]
            act_emb = torch.sum(act_emb.masked_fill(masks.unsqueeze(2)==0,0),dim=1)
            act_emb = act_emb/torch.sum(masks,1,keepdim=True)
            act_emb = F.normalize(act_emb,dim=1,p=2)
            num_atomic_acts = act_emb.size(0)
            print("Num of Atomic Actions:{}".format(num_atomic_acts))
            self.act_cache = act_emb.data
        self.vis_cache = {}



    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(ZSAR, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
            
    def forward(self, vids,texts,objs,obj_clf):
        '''
        forward procedure of ATA module

        output:
        obj_clf_emb: tensor(B*20XD))
        cls_emb: tensor(sum(B,M) X D)
        obj_emb: tensor(B*video_candidates X D)
        base_out: tensor(B*video_candidates X D)
        '''
        # print(objs.keys())
        # print(objs["attention_mask"].size())
        if "glove" not in self.text_pretrain:
            cls_emb = self.text_model(**texts)["last_hidden_state"]
        else:
            cls_emb = self.act_model(texts)
        obj_emb = self.text_model(**objs)["last_hidden_state"]
        obj_clf_emb = self.text_model(**obj_clf)["last_hidden_state"]
        
        batch_size = vids.size(0)
        # print(vids.size())
        
        if self.bert_pooling == "first":
            cls_emb = cls_emb[:,0]
            obj_emb = obj_emb[:,0]
            obj_clf_emb = obj_clf_emb[:,0]
            # cls_emb = self.text_encoder(cls_emb)
        elif self.bert_pooling == "avg":
            # cls_emb = self.text_dp(cls_emb)
            if "glove" not in self.text_pretrain:
                masks = texts["attention_mask"]
                cls_emb = torch.sum(cls_emb.masked_fill(masks.unsqueeze(2)==0,0),dim=1)
                cls_emb = cls_emb/torch.sum(masks,1,keepdim=True)
            else:
                weights = self.ft_attn(cls_emb).squeeze(2)
                weights = torch.softmax(weights, dim=1)
                cls_emb = torch.sum(cls_emb*weights.unsqueeze(2),1)

            masks = objs["attention_mask"]
            obj_emb = torch.sum(obj_emb.masked_fill(masks.unsqueeze(2)==0,0),dim=1)
            obj_emb = obj_emb/torch.sum(masks,1,keepdim=True)

            masks = obj_clf["attention_mask"]
            obj_clf_emb = torch.sum(obj_clf_emb.masked_fill(masks.unsqueeze(2)==0,0),dim=1)
            obj_clf_emb = obj_clf_emb/torch.sum(masks,1,keepdim=True)


        if self.base_model_name == "X3D":
            # print(input.size())
            base_out = self.base_model([vids])
        elif self.base_model_name == "clip":
            base_out = self.base_model(vids)
            base_out = base_out["pooler_output"]
            base_out = self.vision_fc(base_out)
        else:
            base_out = self.base_model(vids) 
        
        if self.video_candidates == 1:
            base_out = base_out.permute(0,2,1)
            if not self.attn:
                base_out = base_out.chunk(batch_size,0)
                base_out = [feat.squeeze(0) for feat in base_out]
                base_out = torch.cat(base_out,0)

        if self.dropout > 0:
            base_out = self.new_fc(base_out) #768
            
        # base_out = F.normalize(base_out)
        

        if self.attn:
            if self.video_candidates>1:
                base_out = torch.stack(base_out.chunk(int(batch_size/self.video_candidates),0),0)
                obj_emb = torch.stack(obj_emb.chunk(int(batch_size/self.video_candidates),0),0)
            else:
                obj_emb = obj_emb.unsqueeze(1)
            
            mix_feat = torch.cat([base_out,obj_emb],1)
            mix_feat = self.attn_model(mix_feat)
            if self.video_candidates>1:
                [base_out,obj_emb] = mix_feat.chunk(2,1)
                base_out = base_out.chunk(int(batch_size/self.video_candidates),0)
                base_out = [feat.squeeze() for feat in base_out]
                base_out = F.normalize(torch.cat(base_out,0),dim=1,p=2)
                obj_emb = obj_emb.chunk(int(batch_size/self.video_candidates),0)
                obj_emb = [feat.squeeze() for feat in obj_emb]
                obj_emb = F.normalize(torch.cat(obj_emb,0),dim=1,p=2)
            else:
                [base_out,obj_emb] = mix_feat.split([4,1],1)
                base_out = base_out.chunk(int(batch_size),0)
                base_out = [feat.squeeze() for feat in base_out]
                base_out = F.normalize(torch.cat(base_out,0),dim=1,p=2)
                obj_emb = obj_emb.repeat(1,4,1)
                obj_emb = obj_emb.chunk(int(batch_size),0)
                obj_emb = [feat.squeeze() for feat in obj_emb]
                obj_emb = F.normalize(torch.cat(obj_emb,0),dim=1,p=2)
        # print(base_out.size())
        # print(obj_emb.size())
        act_gates = self.act_gate_fc(torch.cat([base_out, obj_emb], 1))
        obj_gates = self.obj_gate_fc(torch.cat([base_out, obj_emb], 1))
        base_out = F.normalize(act_gates*base_out,dim=1,p=2)
        obj_emb = F.normalize(obj_gates*obj_emb,dim=1,p=2)
        cls_emb = F.normalize(cls_emb,dim=1,p=2)
        return {"vision":base_out,"cls_emb":cls_emb,"obj_emb":obj_emb,"obj_clf_emb":obj_clf_emb}

    def partialBN(self, enable):
        self._enable_pbn = enable

    @property
    def crop_size(self):
        return [self.input_size,self.input_size]

    @property
    def scale_size(self):
        if self.base_model_name == "X3D":
            return [356, 446]
        elif "R2plus1D" in self.base_model_name:
            return [128,171]
        elif "clip" in self.base_model_name:
            return [256,256]
        elif "CSN" in self.base_model_name:
            return [256,256]
        else:
            raise RuntimeError("not inmplemented base model:{}".format(self.base_model_name))
        # else:
        #     return self.input_size * 256 // self.input_size


    def get_augmentation(self, flip=True):
        if self.modality == 'RGB':
            if flip:
                print("Simple center crop")
                return torchvision.transforms.Compose([GroupScale(self.scale_size),GroupMultiScaleCrop(self.input_size, [1, .875, 0.75,0.66]),
                                                GroupRandomHorizontalFlip(is_flow=False)])
            else:
                print("Simple center crop without flip")
                return torchvision.transforms.Compose([GroupScale(self.scale_size),GroupMultiScaleCrop(self.input_size, [1, .875, 0.75])])
            
        elif self.modality == "TwoStream":
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                        GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
