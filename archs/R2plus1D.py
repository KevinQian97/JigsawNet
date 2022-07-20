import torch.hub
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce
from torchvision.models.video.resnet import VideoResNet, BasicBlock, R2Plus1dStem, Conv2Plus1D, Bottleneck
from archs.vmz import _generic_resnet, R2Plus1dStem_Pool, Conv2Plus1D, model_urls
# from vmz import _generic_resnet, R2Plus1dStem_Pool, Conv2Plus1D, model_urls

# model_urls = {
#     "r2plus1d_34_8_ig65m": "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip8_ig65m_from_scratch-9bae36ae.pth",
#     "r2plus1d_34_32_ig65m": "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ig65m_from_scratch-449a7af9.pth",
#     "r2plus1d_34_8_kinetics": "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip8_ft_kinetics_from_ig65m-0aa0550b.pth",
#     "r2plus1d_34_32_kinetics": "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth",
# }


class VideoResNetWithFeatureReturn(VideoResNet):
    def __init__(self,block,conv_makers,layers,stem):
        super().__init__(block=block,conv_makers=conv_makers,layers=layers,stem=stem)
        self.pool_spatial = Reduce("n c t h w -> n c t", reduction="mean")
        self.pool_temporal = Reduce("n c t -> n c", reduction="mean")
    
    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool_spatial(x)
        # x = self.pool_temporal(x)
        # x = self.avgpool(x)
        # Flatten the layer to fc
        # print(x.size())
        # x = x.flatten(1)
        # x = self.fc(x)

        return x


def r2plus1d_34(pretrain=None,num_classes=400):
    model = VideoResNetWithFeatureReturn(block=BasicBlock,
                                         conv_makers=[Conv2Plus1D] * 4,
                                         layers=[3, 4, 6, 3],
                                         stem=R2Plus1dStem)

    # model = VideoResNet(block=BasicBlock,conv_makers=[Conv2Plus1D] * 4,layers=[3, 4, 6, 3],stem=R2Plus1dStem)

    # model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)

    # Fix difference in PyTorch vs Caffe2 architecture
    # https://github.com/facebookresearch/VMZ/issues/89
    # https://github.com/pytorch/vision/issues/1265
    model.layer2[0].conv2[0] = Conv2Plus1D(128, 128, 288)
    model.layer3[0].conv2[0] = Conv2Plus1D(256, 256, 576)
    model.layer4[0].conv2[0] = Conv2Plus1D(512, 512, 1152)
    model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)
    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9
    if pretrain:
        state_dict = torch.hub.load_state_dict_from_url(model_urls[pretrain],
                                                        progress=True)
        model.load_state_dict(state_dict)
        # print()

    return model

def r2plus1d_152(pretraining="", use_pool1=True, progress=False, **kwargs):
    avail_pretrainings = [
        "ig65m_32frms",
        "ig_ft_kinetics_32frms",
        "sports1m_32frms",
        "sports1m_ft_kinetics_32frms",
    ]
    if pretraining in avail_pretrainings:
        arch = "r2plus1d_152_" + pretraining
        pretrained = True
    else:
        # warnings.warn(
        #     f"Unrecognized pretraining dataset, continuing with randomly initialized network."
        #     " Available pretrainings: {avail_pretrainings}",
        #     UserWarning,
        # )
        arch = "r2plus1d_34"
        pretrained = False

    model = _generic_resnet(
        arch,
        pretrained,
        progress,
        block=Bottleneck,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[3, 8, 36, 3],
        stem=R2Plus1dStem_Pool if use_pool1 else R2Plus1dStem,
        **kwargs
    )
    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls[arch], progress=progress
        )
        model.load_state_dict(state_dict)

    return model


if __name__ == "__main__":
    base_model = r2plus1d_34("r2plus1d_34_32_kinetics").cuda()
    # print(base_model)
    input = torch.randn((4,3,32,112,112)).cuda()
    out = base_model(input)
    print(out.size())