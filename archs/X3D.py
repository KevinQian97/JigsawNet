import math
import torch
import torch.nn as nn
import sys
sys.path.append("../")
from ops import head_helper, resnet_helper, stem_helper
from configs.defaults import get_cfg
import configs.weight_init_helper as init_helper


# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d_nopool": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "i3d_nopool": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
    "x3d": [
        [[5]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels.
        [[3]],  # res3 temporal kernels.
        [[3]],  # res4 temporal kernels.
        [[3]],  # res5 temporal kernels.
    ],
}

_POOL1 = {
    "c2d": [[2, 1, 1]],
    "c2d_nopool": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "i3d_nopool": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
    "x3d": [[1, 1, 1]],
}

def get_norm(cfg):
    """
    Args:
        cfg (CfgNode): model building configs, details are in the comments of
            the config file.
    Returns:
        nn.Module: the normalization layer.
    """
    if cfg.BN.NORM_TYPE == "batchnorm":
        return nn.BatchNorm3d
    # elif cfg.BN.NORM_TYPE == "sub_batchnorm":
    #     return partial(SubBatchNorm3d, num_splits=cfg.BN.NUM_SPLITS)
    # elif cfg.BN.NORM_TYPE == "sync_batchnorm":
    #     return partial(
    #         NaiveSyncBatchNorm3d, num_sync_devices=cfg.BN.NUM_SYNC_DEVICES
    #     )
    else:
        raise NotImplementedError(
            "Norm type {} is not supported".format(cfg.BN.NORM_TYPE)
        )

def build_model(cfg_file, gpu_id=None):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    # Construct the model
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    name = cfg.MODEL.MODEL_NAME
    # if torch.cuda.is_available():
    #     assert (
    #         cfg.NUM_GPUS <= torch.cuda.device_count()
    #     ), "Cannot use more GPU devices than available"
    # else:
    #     assert (
    #         cfg.NUM_GPUS == 0
    #     ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."
    model = X3D(cfg)

    # if cfg.NUM_GPUS:
    #     if gpu_id is None:
    #         # Determine the GPU used by the current process
    #         cur_device = torch.cuda.current_device()
    #     else:
    #         cur_device = gpu_id
    #     # Transfer the model to the current GPU device
    #     model = model.cuda(device=cur_device)
    # # Use multi-process data parallel model in the multi-gpu setting
    # if cfg.NUM_GPUS > 1:
    #     # Make model replica operate on the current device
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         module=model, device_ids=[cur_device], output_device=cur_device
    #     )
    return model

class X3D(nn.Module):
    """
    X3D model builder. It builds a X3D network backbone, which is a ResNet.

    Christoph Feichtenhofer.
    "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(X3D, self).__init__()

        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1

        exp_stage = 2.0
        self.dim_c1 = cfg.X3D.DIM_C1

        self.dim_res2 = (
            self._round_width(self.dim_c1, exp_stage, divisor=8)
            if cfg.X3D.SCALE_RES2
            else self.dim_c1
        )
        self.dim_res3 = self._round_width(self.dim_res2, exp_stage, divisor=8)
        self.dim_res4 = self._round_width(self.dim_res3, exp_stage, divisor=8)
        self.dim_res5 = self._round_width(self.dim_res4, exp_stage, divisor=8)

        self.block_basis = [
            # blocks, c, stride
            [1, self.dim_res2, 2],
            [2, self.dim_res3, 2],
            [5, self.dim_res4, 2],
            [3, self.dim_res5, 2],
        ]
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _round_width(self, width, multiplier, min_depth=8, divisor=8):
        """Round width of filters based on width multiplier."""
        if not multiplier:
            return width

        width *= multiplier
        min_depth = min_depth or divisor
        new_filters = max(
            min_depth, int(width + divisor / 2) // divisor * divisor
        )
        if new_filters < 0.9 * width:
            new_filters += divisor
        return int(new_filters)

    def _round_repeats(self, repeats, multiplier):
        """Round number of layers based on depth multiplier."""
        multiplier = multiplier
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    def _construct_network(self, cfg):
        """
        Builds a single pathway X3D model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        w_mul = cfg.X3D.WIDTH_FACTOR
        d_mul = cfg.X3D.DEPTH_FACTOR
        dim_res1 = self._round_width(self.dim_c1, w_mul)

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[dim_res1],
            kernel=[temp_kernel[0][0] + [3, 3]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 1, 1]],
            norm_module=self.norm_module,
            stem_func_name="x3d_stem",
        )

        # blob_in = s1
        dim_in = dim_res1
        for stage, block in enumerate(self.block_basis):
            dim_out = self._round_width(block[1], w_mul)
            dim_inner = int(cfg.X3D.BOTTLENECK_FACTOR * dim_out)

            n_rep = self._round_repeats(block[0], d_mul)
            prefix = "s{}".format(
                stage + 2
            )  # start w res2 to follow convention

            s = resnet_helper.ResStage(
                dim_in=[dim_in],
                dim_out=[dim_out],
                dim_inner=[dim_inner],
                temp_kernel_sizes=temp_kernel[1],
                stride=[block[2]],
                num_blocks=[n_rep],
                num_groups=[dim_inner]
                if cfg.X3D.CHANNELWISE_3x3x3
                else [num_groups],
                num_block_temp_kernel=[n_rep],
                nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
                nonlocal_group=cfg.NONLOCAL.GROUP[0],
                nonlocal_pool=cfg.NONLOCAL.POOL[0],
                instantiation=cfg.NONLOCAL.INSTANTIATION,
                trans_func_name=cfg.RESNET.TRANS_FUNC,
                stride_1x1=cfg.RESNET.STRIDE_1X1,
                norm_module=self.norm_module,
                dilation=cfg.RESNET.SPATIAL_DILATIONS[stage],
                drop_connect_rate=cfg.MODEL.DROPCONNECT_RATE * (
                    stage + 2) / (len(self.block_basis) + 1)
            )
            dim_in = dim_out
            self.add_module(prefix, s)

        if self.enable_detection:
            NotImplementedError
        else:
            spat_sz = int(math.ceil(cfg.DATA.TRAIN_CROP_SIZE / 32.0))
            self.head = head_helper.X3DHead(
                dim_in=dim_out,
                dim_inner=dim_inner,
                dim_out=cfg.X3D.DIM_C5,
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[cfg.DATA.NUM_FRAMES, spat_sz, spat_sz],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                bn_lin5_on=cfg.X3D.BN_LIN5,
            )

    def forward(self, x, bboxes=None):
        for module in self.children():
            x = module(x)
            # print(module)
        return x

if __name__ == "__main__":
    model = build_model("../configs/Kinetics/X3D_L.yaml")
    checkpoint = torch.load("/data/yijunq/models/TSA/x3d_l.pyth")
    model.load_state_dict(checkpoint["model_state"])
    print(model.head.projection)
    model.cuda()
    print(model)
    input = [torch.randn(3,3,16,312,312).cuda()]
    out = model(input)
    print(out.size())
    