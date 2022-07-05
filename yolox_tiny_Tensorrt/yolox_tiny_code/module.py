
import torch
from torch.nn import *
class FxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.load(r'yolox_tiny_code/model.pt') # Module(   (backbone): Module(     (stem): Module(       (conv): Module(         (conv): Conv2d(12, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)         (bn): BatchNorm2d(24, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)       )     )     (stage1): Module(       (0): Module(         (conv): Conv2d(24, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)         (bn): BatchNorm2d(48, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)       )       (1): Module(         (short_conv): Module(           (conv): Conv2d(48, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(24, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )         (main_conv): Module(           (conv): Conv2d(48, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(24, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )         (blocks): Module(           (0): Module(             (conv1): Module(               (conv): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)               (bn): BatchNorm2d(24, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)             )             (conv2): Module(               (conv): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)               (bn): BatchNorm2d(24, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)             )           )         )         (final_conv): Module(           (conv): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(48, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )       )     )     (stage2): Module(       (0): Module(         (conv): Conv2d(48, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)         (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)       )       (1): Module(         (short_conv): Module(           (conv): Conv2d(96, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(48, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )         (main_conv): Module(           (conv): Conv2d(96, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(48, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )         (blocks): Module(           (0): Module(             (conv1): Module(               (conv): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)               (bn): BatchNorm2d(48, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)             )             (conv2): Module(               (conv): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)               (bn): BatchNorm2d(48, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)             )           )           (1): Module(             (conv1): Module(               (conv): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)               (bn): BatchNorm2d(48, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)             )             (conv2): Module(               (conv): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)               (bn): BatchNorm2d(48, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)             )           )           (2): Module(             (conv1): Module(               (conv): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)               (bn): BatchNorm2d(48, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)             )             (conv2): Module(               (conv): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)               (bn): BatchNorm2d(48, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)             )           )         )         (final_conv): Module(           (conv): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )       )     )     (stage3): Module(       (0): Module(         (conv): Conv2d(96, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)         (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)       )       (1): Module(         (short_conv): Module(           (conv): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )         (main_conv): Module(           (conv): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )         (blocks): Module(           (0): Module(             (conv1): Module(               (conv): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)               (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)             )             (conv2): Module(               (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)               (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)             )           )           (1): Module(             (conv1): Module(               (conv): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)               (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)             )             (conv2): Module(               (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)               (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)             )           )           (2): Module(             (conv1): Module(               (conv): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)               (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)             )             (conv2): Module(               (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)               (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)             )           )         )         (final_conv): Module(           (conv): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )       )     )     (stage4): Module(       (0): Module(         (conv): Conv2d(192, 384, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)         (bn): BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)       )       (1): Module(         (conv1): Module(           (conv): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )         (poolings): Module(           (0): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)           (1): MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False)           (2): MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)         )         (conv2): Module(           (conv): Conv2d(768, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )       )       (2): Module(         (short_conv): Module(           (conv): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )         (main_conv): Module(           (conv): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )         (blocks): Module(           (0): Module(             (conv1): Module(               (conv): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)               (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)             )             (conv2): Module(               (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)               (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)             )           )         )         (final_conv): Module(           (conv): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )       )     )   )   (neck): Module(     (reduce_layers): Module(       (0): Module(         (conv): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)         (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)       )       (1): Module(         (conv): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)         (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)       )     )     (upsample): Upsample(scale_factor=2.0, mode=nearest)     (top_down_blocks): Module(       (0): Module(         (short_conv): Module(           (conv): Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )         (main_conv): Module(           (conv): Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )         (blocks): Module(           (0): Module(             (conv1): Module(               (conv): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)               (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)             )             (conv2): Module(               (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)               (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)             )           )         )         (final_conv): Module(           (conv): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )       )       (1): Module(         (short_conv): Module(           (conv): Conv2d(192, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(48, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )         (main_conv): Module(           (conv): Conv2d(192, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(48, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )         (blocks): Module(           (0): Module(             (conv1): Module(               (conv): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)               (bn): BatchNorm2d(48, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)             )             (conv2): Module(               (conv): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)               (bn): BatchNorm2d(48, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)             )           )         )         (final_conv): Module(           (conv): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )       )     )     (downsamples): Module(       (0): Module(         (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)         (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)       )       (1): Module(         (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)         (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)       )     )     (bottom_up_blocks): Module(       (0): Module(         (short_conv): Module(           (conv): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )         (main_conv): Module(           (conv): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )         (blocks): Module(           (0): Module(             (conv1): Module(               (conv): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)               (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)             )             (conv2): Module(               (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)               (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)             )           )         )         (final_conv): Module(           (conv): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )       )       (1): Module(         (short_conv): Module(           (conv): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )         (main_conv): Module(           (conv): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )         (blocks): Module(           (0): Module(             (conv1): Module(               (conv): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)               (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)             )             (conv2): Module(               (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)               (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)             )           )         )         (final_conv): Module(           (conv): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)           (bn): BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )       )     )     (out_convs): Module(       (0): Module(         (conv): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)         (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)       )       (1): Module(         (conv): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)         (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)       )       (2): Module(         (conv): Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)         (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)       )     )   )   (bbox_head): Module(     (multi_level_cls_convs): Module(       (0): Module(         (0): Module(           (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)           (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )         (1): Module(           (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)           (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )       )       (1): Module(         (0): Module(           (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)           (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )         (1): Module(           (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)           (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )       )       (2): Module(         (0): Module(           (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)           (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )         (1): Module(           (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)           (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )       )     )     (multi_level_reg_convs): Module(       (0): Module(         (0): Module(           (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)           (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )         (1): Module(           (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)           (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )       )       (1): Module(         (0): Module(           (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)           (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )         (1): Module(           (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)           (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )       )       (2): Module(         (0): Module(           (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)           (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )         (1): Module(           (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)           (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)         )       )     )     (multi_level_conv_cls): Module(       (0): Conv2d(96, 80, kernel_size=(1, 1), stride=(1, 1))       (1): Conv2d(96, 80, kernel_size=(1, 1), stride=(1, 1))       (2): Conv2d(96, 80, kernel_size=(1, 1), stride=(1, 1))     )     (multi_level_conv_reg): Module(       (0): Conv2d(96, 4, kernel_size=(1, 1), stride=(1, 1))       (1): Conv2d(96, 4, kernel_size=(1, 1), stride=(1, 1))       (2): Conv2d(96, 4, kernel_size=(1, 1), stride=(1, 1))     )     (multi_level_conv_obj): Module(       (0): Conv2d(96, 1, kernel_size=(1, 1), stride=(1, 1))       (1): Conv2d(96, 1, kernel_size=(1, 1), stride=(1, 1))       (2): Conv2d(96, 1, kernel_size=(1, 1), stride=(1, 1))     )   ) )
        self.load_state_dict(torch.load(r'yolox_tiny_code/state_dict.pt'))

    def forward(self, x):
        getitem = x[(Ellipsis, slice(None, None, 2), slice(None, None, 2))]
        getitem_1 = x[(Ellipsis, slice(None, None, 2), slice(1, None, 2))]
        getitem_2 = x[(Ellipsis, slice(1, None, 2), slice(None, None, 2))]
        getitem_3 = x[(Ellipsis, slice(1, None, 2), slice(1, None, 2))];  x = None
        cat_1 = torch.cat((getitem, getitem_2, getitem_1, getitem_3), dim = 1);  getitem = getitem_2 = getitem_1 = getitem_3 = None
        model_backbone_stem_conv_conv = self.model.backbone.stem.conv.conv(cat_1);  cat_1 = None
        model_backbone_stem_conv_bn = self.model.backbone.stem.conv.bn(model_backbone_stem_conv_conv);  model_backbone_stem_conv_conv = None
        sigmoid_1 = torch.sigmoid(model_backbone_stem_conv_bn)
        mul_1 = model_backbone_stem_conv_bn * sigmoid_1;  model_backbone_stem_conv_bn = sigmoid_1 = None
        model_backbone_stage1_0_conv = getattr(self.model.backbone.stage1, "0").conv(mul_1);  mul_1 = None
        model_backbone_stage1_0_bn = getattr(self.model.backbone.stage1, "0").bn(model_backbone_stage1_0_conv);  model_backbone_stage1_0_conv = None
        sigmoid_2 = torch.sigmoid(model_backbone_stage1_0_bn)
        mul_2 = model_backbone_stage1_0_bn * sigmoid_2;  model_backbone_stage1_0_bn = sigmoid_2 = None
        model_backbone_stage1_1_short_conv_conv = getattr(self.model.backbone.stage1, "1").short_conv.conv(mul_2)
        model_backbone_stage1_1_short_conv_bn = getattr(self.model.backbone.stage1, "1").short_conv.bn(model_backbone_stage1_1_short_conv_conv);  model_backbone_stage1_1_short_conv_conv = None
        sigmoid_3 = torch.sigmoid(model_backbone_stage1_1_short_conv_bn)
        mul_3 = model_backbone_stage1_1_short_conv_bn * sigmoid_3;  model_backbone_stage1_1_short_conv_bn = sigmoid_3 = None
        model_backbone_stage1_1_main_conv_conv = getattr(self.model.backbone.stage1, "1").main_conv.conv(mul_2);  mul_2 = None
        model_backbone_stage1_1_main_conv_bn = getattr(self.model.backbone.stage1, "1").main_conv.bn(model_backbone_stage1_1_main_conv_conv);  model_backbone_stage1_1_main_conv_conv = None
        sigmoid_4 = torch.sigmoid(model_backbone_stage1_1_main_conv_bn)
        mul_4 = model_backbone_stage1_1_main_conv_bn * sigmoid_4;  model_backbone_stage1_1_main_conv_bn = sigmoid_4 = None
        model_backbone_stage1_1_blocks_0_conv1_conv = getattr(getattr(self.model.backbone.stage1, "1").blocks, "0").conv1.conv(mul_4)
        model_backbone_stage1_1_blocks_0_conv1_bn = getattr(getattr(self.model.backbone.stage1, "1").blocks, "0").conv1.bn(model_backbone_stage1_1_blocks_0_conv1_conv);  model_backbone_stage1_1_blocks_0_conv1_conv = None
        sigmoid_5 = torch.sigmoid(model_backbone_stage1_1_blocks_0_conv1_bn)
        mul_5 = model_backbone_stage1_1_blocks_0_conv1_bn * sigmoid_5;  model_backbone_stage1_1_blocks_0_conv1_bn = sigmoid_5 = None
        model_backbone_stage1_1_blocks_0_conv2_conv = getattr(getattr(self.model.backbone.stage1, "1").blocks, "0").conv2.conv(mul_5);  mul_5 = None
        model_backbone_stage1_1_blocks_0_conv2_bn = getattr(getattr(self.model.backbone.stage1, "1").blocks, "0").conv2.bn(model_backbone_stage1_1_blocks_0_conv2_conv);  model_backbone_stage1_1_blocks_0_conv2_conv = None
        sigmoid_6 = torch.sigmoid(model_backbone_stage1_1_blocks_0_conv2_bn)
        mul_6 = model_backbone_stage1_1_blocks_0_conv2_bn * sigmoid_6;  model_backbone_stage1_1_blocks_0_conv2_bn = sigmoid_6 = None
        add_1 = mul_6 + mul_4;  mul_6 = mul_4 = None
        cat_2 = torch.cat((add_1, mul_3), dim = 1);  add_1 = mul_3 = None
        model_backbone_stage1_1_final_conv_conv = getattr(self.model.backbone.stage1, "1").final_conv.conv(cat_2);  cat_2 = None
        model_backbone_stage1_1_final_conv_bn = getattr(self.model.backbone.stage1, "1").final_conv.bn(model_backbone_stage1_1_final_conv_conv);  model_backbone_stage1_1_final_conv_conv = None
        sigmoid_7 = torch.sigmoid(model_backbone_stage1_1_final_conv_bn)
        mul_7 = model_backbone_stage1_1_final_conv_bn * sigmoid_7;  model_backbone_stage1_1_final_conv_bn = sigmoid_7 = None
        model_backbone_stage2_0_conv = getattr(self.model.backbone.stage2, "0").conv(mul_7);  mul_7 = None
        model_backbone_stage2_0_bn = getattr(self.model.backbone.stage2, "0").bn(model_backbone_stage2_0_conv);  model_backbone_stage2_0_conv = None
        sigmoid_8 = torch.sigmoid(model_backbone_stage2_0_bn)
        mul_8 = model_backbone_stage2_0_bn * sigmoid_8;  model_backbone_stage2_0_bn = sigmoid_8 = None
        model_backbone_stage2_1_short_conv_conv = getattr(self.model.backbone.stage2, "1").short_conv.conv(mul_8)
        model_backbone_stage2_1_short_conv_bn = getattr(self.model.backbone.stage2, "1").short_conv.bn(model_backbone_stage2_1_short_conv_conv);  model_backbone_stage2_1_short_conv_conv = None
        sigmoid_9 = torch.sigmoid(model_backbone_stage2_1_short_conv_bn)
        mul_9 = model_backbone_stage2_1_short_conv_bn * sigmoid_9;  model_backbone_stage2_1_short_conv_bn = sigmoid_9 = None
        model_backbone_stage2_1_main_conv_conv = getattr(self.model.backbone.stage2, "1").main_conv.conv(mul_8);  mul_8 = None
        model_backbone_stage2_1_main_conv_bn = getattr(self.model.backbone.stage2, "1").main_conv.bn(model_backbone_stage2_1_main_conv_conv);  model_backbone_stage2_1_main_conv_conv = None
        sigmoid_10 = torch.sigmoid(model_backbone_stage2_1_main_conv_bn)
        mul_10 = model_backbone_stage2_1_main_conv_bn * sigmoid_10;  model_backbone_stage2_1_main_conv_bn = sigmoid_10 = None
        model_backbone_stage2_1_blocks_0_conv1_conv = getattr(getattr(self.model.backbone.stage2, "1").blocks, "0").conv1.conv(mul_10)
        model_backbone_stage2_1_blocks_0_conv1_bn = getattr(getattr(self.model.backbone.stage2, "1").blocks, "0").conv1.bn(model_backbone_stage2_1_blocks_0_conv1_conv);  model_backbone_stage2_1_blocks_0_conv1_conv = None
        sigmoid_11 = torch.sigmoid(model_backbone_stage2_1_blocks_0_conv1_bn)
        mul_11 = model_backbone_stage2_1_blocks_0_conv1_bn * sigmoid_11;  model_backbone_stage2_1_blocks_0_conv1_bn = sigmoid_11 = None
        model_backbone_stage2_1_blocks_0_conv2_conv = getattr(getattr(self.model.backbone.stage2, "1").blocks, "0").conv2.conv(mul_11);  mul_11 = None
        model_backbone_stage2_1_blocks_0_conv2_bn = getattr(getattr(self.model.backbone.stage2, "1").blocks, "0").conv2.bn(model_backbone_stage2_1_blocks_0_conv2_conv);  model_backbone_stage2_1_blocks_0_conv2_conv = None
        sigmoid_12 = torch.sigmoid(model_backbone_stage2_1_blocks_0_conv2_bn)
        mul_12 = model_backbone_stage2_1_blocks_0_conv2_bn * sigmoid_12;  model_backbone_stage2_1_blocks_0_conv2_bn = sigmoid_12 = None
        add_2 = mul_12 + mul_10;  mul_12 = mul_10 = None
        model_backbone_stage2_1_blocks_1_conv1_conv = getattr(getattr(self.model.backbone.stage2, "1").blocks, "1").conv1.conv(add_2)
        model_backbone_stage2_1_blocks_1_conv1_bn = getattr(getattr(self.model.backbone.stage2, "1").blocks, "1").conv1.bn(model_backbone_stage2_1_blocks_1_conv1_conv);  model_backbone_stage2_1_blocks_1_conv1_conv = None
        sigmoid_13 = torch.sigmoid(model_backbone_stage2_1_blocks_1_conv1_bn)
        mul_13 = model_backbone_stage2_1_blocks_1_conv1_bn * sigmoid_13;  model_backbone_stage2_1_blocks_1_conv1_bn = sigmoid_13 = None
        model_backbone_stage2_1_blocks_1_conv2_conv = getattr(getattr(self.model.backbone.stage2, "1").blocks, "1").conv2.conv(mul_13);  mul_13 = None
        model_backbone_stage2_1_blocks_1_conv2_bn = getattr(getattr(self.model.backbone.stage2, "1").blocks, "1").conv2.bn(model_backbone_stage2_1_blocks_1_conv2_conv);  model_backbone_stage2_1_blocks_1_conv2_conv = None
        sigmoid_14 = torch.sigmoid(model_backbone_stage2_1_blocks_1_conv2_bn)
        mul_14 = model_backbone_stage2_1_blocks_1_conv2_bn * sigmoid_14;  model_backbone_stage2_1_blocks_1_conv2_bn = sigmoid_14 = None
        add_3 = mul_14 + add_2;  mul_14 = add_2 = None
        model_backbone_stage2_1_blocks_2_conv1_conv = getattr(getattr(self.model.backbone.stage2, "1").blocks, "2").conv1.conv(add_3)
        model_backbone_stage2_1_blocks_2_conv1_bn = getattr(getattr(self.model.backbone.stage2, "1").blocks, "2").conv1.bn(model_backbone_stage2_1_blocks_2_conv1_conv);  model_backbone_stage2_1_blocks_2_conv1_conv = None
        sigmoid_15 = torch.sigmoid(model_backbone_stage2_1_blocks_2_conv1_bn)
        mul_15 = model_backbone_stage2_1_blocks_2_conv1_bn * sigmoid_15;  model_backbone_stage2_1_blocks_2_conv1_bn = sigmoid_15 = None
        model_backbone_stage2_1_blocks_2_conv2_conv = getattr(getattr(self.model.backbone.stage2, "1").blocks, "2").conv2.conv(mul_15);  mul_15 = None
        model_backbone_stage2_1_blocks_2_conv2_bn = getattr(getattr(self.model.backbone.stage2, "1").blocks, "2").conv2.bn(model_backbone_stage2_1_blocks_2_conv2_conv);  model_backbone_stage2_1_blocks_2_conv2_conv = None
        sigmoid_16 = torch.sigmoid(model_backbone_stage2_1_blocks_2_conv2_bn)
        mul_16 = model_backbone_stage2_1_blocks_2_conv2_bn * sigmoid_16;  model_backbone_stage2_1_blocks_2_conv2_bn = sigmoid_16 = None
        add_4 = mul_16 + add_3;  mul_16 = add_3 = None
        cat_3 = torch.cat((add_4, mul_9), dim = 1);  add_4 = mul_9 = None
        model_backbone_stage2_1_final_conv_conv = getattr(self.model.backbone.stage2, "1").final_conv.conv(cat_3);  cat_3 = None
        model_backbone_stage2_1_final_conv_bn = getattr(self.model.backbone.stage2, "1").final_conv.bn(model_backbone_stage2_1_final_conv_conv);  model_backbone_stage2_1_final_conv_conv = None
        sigmoid_17 = torch.sigmoid(model_backbone_stage2_1_final_conv_bn)
        mul_17 = model_backbone_stage2_1_final_conv_bn * sigmoid_17;  model_backbone_stage2_1_final_conv_bn = sigmoid_17 = None
        model_backbone_stage3_0_conv = getattr(self.model.backbone.stage3, "0").conv(mul_17)
        model_backbone_stage3_0_bn = getattr(self.model.backbone.stage3, "0").bn(model_backbone_stage3_0_conv);  model_backbone_stage3_0_conv = None
        sigmoid_18 = torch.sigmoid(model_backbone_stage3_0_bn)
        mul_18 = model_backbone_stage3_0_bn * sigmoid_18;  model_backbone_stage3_0_bn = sigmoid_18 = None
        model_backbone_stage3_1_short_conv_conv = getattr(self.model.backbone.stage3, "1").short_conv.conv(mul_18)
        model_backbone_stage3_1_short_conv_bn = getattr(self.model.backbone.stage3, "1").short_conv.bn(model_backbone_stage3_1_short_conv_conv);  model_backbone_stage3_1_short_conv_conv = None
        sigmoid_19 = torch.sigmoid(model_backbone_stage3_1_short_conv_bn)
        mul_19 = model_backbone_stage3_1_short_conv_bn * sigmoid_19;  model_backbone_stage3_1_short_conv_bn = sigmoid_19 = None
        model_backbone_stage3_1_main_conv_conv = getattr(self.model.backbone.stage3, "1").main_conv.conv(mul_18);  mul_18 = None
        model_backbone_stage3_1_main_conv_bn = getattr(self.model.backbone.stage3, "1").main_conv.bn(model_backbone_stage3_1_main_conv_conv);  model_backbone_stage3_1_main_conv_conv = None
        sigmoid_20 = torch.sigmoid(model_backbone_stage3_1_main_conv_bn)
        mul_20 = model_backbone_stage3_1_main_conv_bn * sigmoid_20;  model_backbone_stage3_1_main_conv_bn = sigmoid_20 = None
        model_backbone_stage3_1_blocks_0_conv1_conv = getattr(getattr(self.model.backbone.stage3, "1").blocks, "0").conv1.conv(mul_20)
        model_backbone_stage3_1_blocks_0_conv1_bn = getattr(getattr(self.model.backbone.stage3, "1").blocks, "0").conv1.bn(model_backbone_stage3_1_blocks_0_conv1_conv);  model_backbone_stage3_1_blocks_0_conv1_conv = None
        sigmoid_21 = torch.sigmoid(model_backbone_stage3_1_blocks_0_conv1_bn)
        mul_21 = model_backbone_stage3_1_blocks_0_conv1_bn * sigmoid_21;  model_backbone_stage3_1_blocks_0_conv1_bn = sigmoid_21 = None
        model_backbone_stage3_1_blocks_0_conv2_conv = getattr(getattr(self.model.backbone.stage3, "1").blocks, "0").conv2.conv(mul_21);  mul_21 = None
        model_backbone_stage3_1_blocks_0_conv2_bn = getattr(getattr(self.model.backbone.stage3, "1").blocks, "0").conv2.bn(model_backbone_stage3_1_blocks_0_conv2_conv);  model_backbone_stage3_1_blocks_0_conv2_conv = None
        sigmoid_22 = torch.sigmoid(model_backbone_stage3_1_blocks_0_conv2_bn)
        mul_22 = model_backbone_stage3_1_blocks_0_conv2_bn * sigmoid_22;  model_backbone_stage3_1_blocks_0_conv2_bn = sigmoid_22 = None
        add_5 = mul_22 + mul_20;  mul_22 = mul_20 = None
        model_backbone_stage3_1_blocks_1_conv1_conv = getattr(getattr(self.model.backbone.stage3, "1").blocks, "1").conv1.conv(add_5)
        model_backbone_stage3_1_blocks_1_conv1_bn = getattr(getattr(self.model.backbone.stage3, "1").blocks, "1").conv1.bn(model_backbone_stage3_1_blocks_1_conv1_conv);  model_backbone_stage3_1_blocks_1_conv1_conv = None
        sigmoid_23 = torch.sigmoid(model_backbone_stage3_1_blocks_1_conv1_bn)
        mul_23 = model_backbone_stage3_1_blocks_1_conv1_bn * sigmoid_23;  model_backbone_stage3_1_blocks_1_conv1_bn = sigmoid_23 = None
        model_backbone_stage3_1_blocks_1_conv2_conv = getattr(getattr(self.model.backbone.stage3, "1").blocks, "1").conv2.conv(mul_23);  mul_23 = None
        model_backbone_stage3_1_blocks_1_conv2_bn = getattr(getattr(self.model.backbone.stage3, "1").blocks, "1").conv2.bn(model_backbone_stage3_1_blocks_1_conv2_conv);  model_backbone_stage3_1_blocks_1_conv2_conv = None
        sigmoid_24 = torch.sigmoid(model_backbone_stage3_1_blocks_1_conv2_bn)
        mul_24 = model_backbone_stage3_1_blocks_1_conv2_bn * sigmoid_24;  model_backbone_stage3_1_blocks_1_conv2_bn = sigmoid_24 = None
        add_6 = mul_24 + add_5;  mul_24 = add_5 = None
        model_backbone_stage3_1_blocks_2_conv1_conv = getattr(getattr(self.model.backbone.stage3, "1").blocks, "2").conv1.conv(add_6)
        model_backbone_stage3_1_blocks_2_conv1_bn = getattr(getattr(self.model.backbone.stage3, "1").blocks, "2").conv1.bn(model_backbone_stage3_1_blocks_2_conv1_conv);  model_backbone_stage3_1_blocks_2_conv1_conv = None
        sigmoid_25 = torch.sigmoid(model_backbone_stage3_1_blocks_2_conv1_bn)
        mul_25 = model_backbone_stage3_1_blocks_2_conv1_bn * sigmoid_25;  model_backbone_stage3_1_blocks_2_conv1_bn = sigmoid_25 = None
        model_backbone_stage3_1_blocks_2_conv2_conv = getattr(getattr(self.model.backbone.stage3, "1").blocks, "2").conv2.conv(mul_25);  mul_25 = None
        model_backbone_stage3_1_blocks_2_conv2_bn = getattr(getattr(self.model.backbone.stage3, "1").blocks, "2").conv2.bn(model_backbone_stage3_1_blocks_2_conv2_conv);  model_backbone_stage3_1_blocks_2_conv2_conv = None
        sigmoid_26 = torch.sigmoid(model_backbone_stage3_1_blocks_2_conv2_bn)
        mul_26 = model_backbone_stage3_1_blocks_2_conv2_bn * sigmoid_26;  model_backbone_stage3_1_blocks_2_conv2_bn = sigmoid_26 = None
        add_7 = mul_26 + add_6;  mul_26 = add_6 = None
        cat_4 = torch.cat((add_7, mul_19), dim = 1);  add_7 = mul_19 = None
        model_backbone_stage3_1_final_conv_conv = getattr(self.model.backbone.stage3, "1").final_conv.conv(cat_4);  cat_4 = None
        model_backbone_stage3_1_final_conv_bn = getattr(self.model.backbone.stage3, "1").final_conv.bn(model_backbone_stage3_1_final_conv_conv);  model_backbone_stage3_1_final_conv_conv = None
        sigmoid_27 = torch.sigmoid(model_backbone_stage3_1_final_conv_bn)
        mul_27 = model_backbone_stage3_1_final_conv_bn * sigmoid_27;  model_backbone_stage3_1_final_conv_bn = sigmoid_27 = None
        model_backbone_stage4_0_conv = getattr(self.model.backbone.stage4, "0").conv(mul_27)
        model_backbone_stage4_0_bn = getattr(self.model.backbone.stage4, "0").bn(model_backbone_stage4_0_conv);  model_backbone_stage4_0_conv = None
        sigmoid_28 = torch.sigmoid(model_backbone_stage4_0_bn)
        mul_28 = model_backbone_stage4_0_bn * sigmoid_28;  model_backbone_stage4_0_bn = sigmoid_28 = None
        model_backbone_stage4_1_conv1_conv = getattr(self.model.backbone.stage4, "1").conv1.conv(mul_28);  mul_28 = None
        model_backbone_stage4_1_conv1_bn = getattr(self.model.backbone.stage4, "1").conv1.bn(model_backbone_stage4_1_conv1_conv);  model_backbone_stage4_1_conv1_conv = None
        sigmoid_29 = torch.sigmoid(model_backbone_stage4_1_conv1_bn)
        mul_29 = model_backbone_stage4_1_conv1_bn * sigmoid_29;  model_backbone_stage4_1_conv1_bn = sigmoid_29 = None
        model_backbone_stage4_1_poolings_0 = getattr(getattr(self.model.backbone.stage4, "1").poolings, "0")(mul_29)
        model_backbone_stage4_1_poolings_1 = getattr(getattr(self.model.backbone.stage4, "1").poolings, "1")(mul_29)
        model_backbone_stage4_1_poolings_2 = getattr(getattr(self.model.backbone.stage4, "1").poolings, "2")(mul_29)
        cat_5 = torch.cat([mul_29, model_backbone_stage4_1_poolings_0, model_backbone_stage4_1_poolings_1, model_backbone_stage4_1_poolings_2], dim = 1);  mul_29 = model_backbone_stage4_1_poolings_0 = model_backbone_stage4_1_poolings_1 = model_backbone_stage4_1_poolings_2 = None
        model_backbone_stage4_1_conv2_conv = getattr(self.model.backbone.stage4, "1").conv2.conv(cat_5);  cat_5 = None
        model_backbone_stage4_1_conv2_bn = getattr(self.model.backbone.stage4, "1").conv2.bn(model_backbone_stage4_1_conv2_conv);  model_backbone_stage4_1_conv2_conv = None
        sigmoid_30 = torch.sigmoid(model_backbone_stage4_1_conv2_bn)
        mul_30 = model_backbone_stage4_1_conv2_bn * sigmoid_30;  model_backbone_stage4_1_conv2_bn = sigmoid_30 = None
        model_backbone_stage4_2_short_conv_conv = getattr(self.model.backbone.stage4, "2").short_conv.conv(mul_30)
        model_backbone_stage4_2_short_conv_bn = getattr(self.model.backbone.stage4, "2").short_conv.bn(model_backbone_stage4_2_short_conv_conv);  model_backbone_stage4_2_short_conv_conv = None
        sigmoid_31 = torch.sigmoid(model_backbone_stage4_2_short_conv_bn)
        mul_31 = model_backbone_stage4_2_short_conv_bn * sigmoid_31;  model_backbone_stage4_2_short_conv_bn = sigmoid_31 = None
        model_backbone_stage4_2_main_conv_conv = getattr(self.model.backbone.stage4, "2").main_conv.conv(mul_30);  mul_30 = None
        model_backbone_stage4_2_main_conv_bn = getattr(self.model.backbone.stage4, "2").main_conv.bn(model_backbone_stage4_2_main_conv_conv);  model_backbone_stage4_2_main_conv_conv = None
        sigmoid_32 = torch.sigmoid(model_backbone_stage4_2_main_conv_bn)
        mul_32 = model_backbone_stage4_2_main_conv_bn * sigmoid_32;  model_backbone_stage4_2_main_conv_bn = sigmoid_32 = None
        model_backbone_stage4_2_blocks_0_conv1_conv = getattr(getattr(self.model.backbone.stage4, "2").blocks, "0").conv1.conv(mul_32);  mul_32 = None
        model_backbone_stage4_2_blocks_0_conv1_bn = getattr(getattr(self.model.backbone.stage4, "2").blocks, "0").conv1.bn(model_backbone_stage4_2_blocks_0_conv1_conv);  model_backbone_stage4_2_blocks_0_conv1_conv = None
        sigmoid_33 = torch.sigmoid(model_backbone_stage4_2_blocks_0_conv1_bn)
        mul_33 = model_backbone_stage4_2_blocks_0_conv1_bn * sigmoid_33;  model_backbone_stage4_2_blocks_0_conv1_bn = sigmoid_33 = None
        model_backbone_stage4_2_blocks_0_conv2_conv = getattr(getattr(self.model.backbone.stage4, "2").blocks, "0").conv2.conv(mul_33);  mul_33 = None
        model_backbone_stage4_2_blocks_0_conv2_bn = getattr(getattr(self.model.backbone.stage4, "2").blocks, "0").conv2.bn(model_backbone_stage4_2_blocks_0_conv2_conv);  model_backbone_stage4_2_blocks_0_conv2_conv = None
        sigmoid_34 = torch.sigmoid(model_backbone_stage4_2_blocks_0_conv2_bn)
        mul_34 = model_backbone_stage4_2_blocks_0_conv2_bn * sigmoid_34;  model_backbone_stage4_2_blocks_0_conv2_bn = sigmoid_34 = None
        cat_6 = torch.cat((mul_34, mul_31), dim = 1);  mul_34 = mul_31 = None
        model_backbone_stage4_2_final_conv_conv = getattr(self.model.backbone.stage4, "2").final_conv.conv(cat_6);  cat_6 = None
        model_backbone_stage4_2_final_conv_bn = getattr(self.model.backbone.stage4, "2").final_conv.bn(model_backbone_stage4_2_final_conv_conv);  model_backbone_stage4_2_final_conv_conv = None
        sigmoid_35 = torch.sigmoid(model_backbone_stage4_2_final_conv_bn)
        mul_35 = model_backbone_stage4_2_final_conv_bn * sigmoid_35;  model_backbone_stage4_2_final_conv_bn = sigmoid_35 = None
        model_neck_reduce_layers_0_conv = getattr(self.model.neck.reduce_layers, "0").conv(mul_35);  mul_35 = None
        model_neck_reduce_layers_0_bn = getattr(self.model.neck.reduce_layers, "0").bn(model_neck_reduce_layers_0_conv);  model_neck_reduce_layers_0_conv = None
        sigmoid_36 = torch.sigmoid(model_neck_reduce_layers_0_bn)
        mul_36 = model_neck_reduce_layers_0_bn * sigmoid_36;  model_neck_reduce_layers_0_bn = sigmoid_36 = None
        model_neck_upsample = self.model.neck.upsample(mul_36)
        cat_7 = torch.cat([model_neck_upsample, mul_27], 1);  model_neck_upsample = mul_27 = None
        model_neck_top_down_blocks_0_short_conv_conv = getattr(self.model.neck.top_down_blocks, "0").short_conv.conv(cat_7)
        model_neck_top_down_blocks_0_short_conv_bn = getattr(self.model.neck.top_down_blocks, "0").short_conv.bn(model_neck_top_down_blocks_0_short_conv_conv);  model_neck_top_down_blocks_0_short_conv_conv = None
        sigmoid_37 = torch.sigmoid(model_neck_top_down_blocks_0_short_conv_bn)
        mul_37 = model_neck_top_down_blocks_0_short_conv_bn * sigmoid_37;  model_neck_top_down_blocks_0_short_conv_bn = sigmoid_37 = None
        model_neck_top_down_blocks_0_main_conv_conv = getattr(self.model.neck.top_down_blocks, "0").main_conv.conv(cat_7);  cat_7 = None
        model_neck_top_down_blocks_0_main_conv_bn = getattr(self.model.neck.top_down_blocks, "0").main_conv.bn(model_neck_top_down_blocks_0_main_conv_conv);  model_neck_top_down_blocks_0_main_conv_conv = None
        sigmoid_38 = torch.sigmoid(model_neck_top_down_blocks_0_main_conv_bn)
        mul_38 = model_neck_top_down_blocks_0_main_conv_bn * sigmoid_38;  model_neck_top_down_blocks_0_main_conv_bn = sigmoid_38 = None
        model_neck_top_down_blocks_0_blocks_0_conv1_conv = getattr(getattr(self.model.neck.top_down_blocks, "0").blocks, "0").conv1.conv(mul_38);  mul_38 = None
        model_neck_top_down_blocks_0_blocks_0_conv1_bn = getattr(getattr(self.model.neck.top_down_blocks, "0").blocks, "0").conv1.bn(model_neck_top_down_blocks_0_blocks_0_conv1_conv);  model_neck_top_down_blocks_0_blocks_0_conv1_conv = None
        sigmoid_39 = torch.sigmoid(model_neck_top_down_blocks_0_blocks_0_conv1_bn)
        mul_39 = model_neck_top_down_blocks_0_blocks_0_conv1_bn * sigmoid_39;  model_neck_top_down_blocks_0_blocks_0_conv1_bn = sigmoid_39 = None
        model_neck_top_down_blocks_0_blocks_0_conv2_conv = getattr(getattr(self.model.neck.top_down_blocks, "0").blocks, "0").conv2.conv(mul_39);  mul_39 = None
        model_neck_top_down_blocks_0_blocks_0_conv2_bn = getattr(getattr(self.model.neck.top_down_blocks, "0").blocks, "0").conv2.bn(model_neck_top_down_blocks_0_blocks_0_conv2_conv);  model_neck_top_down_blocks_0_blocks_0_conv2_conv = None
        sigmoid_40 = torch.sigmoid(model_neck_top_down_blocks_0_blocks_0_conv2_bn)
        mul_40 = model_neck_top_down_blocks_0_blocks_0_conv2_bn * sigmoid_40;  model_neck_top_down_blocks_0_blocks_0_conv2_bn = sigmoid_40 = None
        cat_8 = torch.cat((mul_40, mul_37), dim = 1);  mul_40 = mul_37 = None
        model_neck_top_down_blocks_0_final_conv_conv = getattr(self.model.neck.top_down_blocks, "0").final_conv.conv(cat_8);  cat_8 = None
        model_neck_top_down_blocks_0_final_conv_bn = getattr(self.model.neck.top_down_blocks, "0").final_conv.bn(model_neck_top_down_blocks_0_final_conv_conv);  model_neck_top_down_blocks_0_final_conv_conv = None
        sigmoid_41 = torch.sigmoid(model_neck_top_down_blocks_0_final_conv_bn)
        mul_41 = model_neck_top_down_blocks_0_final_conv_bn * sigmoid_41;  model_neck_top_down_blocks_0_final_conv_bn = sigmoid_41 = None
        model_neck_reduce_layers_1_conv = getattr(self.model.neck.reduce_layers, "1").conv(mul_41);  mul_41 = None
        model_neck_reduce_layers_1_bn = getattr(self.model.neck.reduce_layers, "1").bn(model_neck_reduce_layers_1_conv);  model_neck_reduce_layers_1_conv = None
        sigmoid_42 = torch.sigmoid(model_neck_reduce_layers_1_bn)
        mul_42 = model_neck_reduce_layers_1_bn * sigmoid_42;  model_neck_reduce_layers_1_bn = sigmoid_42 = None
        model_neck_upsample_1 = self.model.neck.upsample(mul_42)
        cat_9 = torch.cat([model_neck_upsample_1, mul_17], 1);  model_neck_upsample_1 = mul_17 = None
        model_neck_top_down_blocks_1_short_conv_conv = getattr(self.model.neck.top_down_blocks, "1").short_conv.conv(cat_9)
        model_neck_top_down_blocks_1_short_conv_bn = getattr(self.model.neck.top_down_blocks, "1").short_conv.bn(model_neck_top_down_blocks_1_short_conv_conv);  model_neck_top_down_blocks_1_short_conv_conv = None
        sigmoid_43 = torch.sigmoid(model_neck_top_down_blocks_1_short_conv_bn)
        mul_43 = model_neck_top_down_blocks_1_short_conv_bn * sigmoid_43;  model_neck_top_down_blocks_1_short_conv_bn = sigmoid_43 = None
        model_neck_top_down_blocks_1_main_conv_conv = getattr(self.model.neck.top_down_blocks, "1").main_conv.conv(cat_9);  cat_9 = None
        model_neck_top_down_blocks_1_main_conv_bn = getattr(self.model.neck.top_down_blocks, "1").main_conv.bn(model_neck_top_down_blocks_1_main_conv_conv);  model_neck_top_down_blocks_1_main_conv_conv = None
        sigmoid_44 = torch.sigmoid(model_neck_top_down_blocks_1_main_conv_bn)
        mul_44 = model_neck_top_down_blocks_1_main_conv_bn * sigmoid_44;  model_neck_top_down_blocks_1_main_conv_bn = sigmoid_44 = None
        model_neck_top_down_blocks_1_blocks_0_conv1_conv = getattr(getattr(self.model.neck.top_down_blocks, "1").blocks, "0").conv1.conv(mul_44);  mul_44 = None
        model_neck_top_down_blocks_1_blocks_0_conv1_bn = getattr(getattr(self.model.neck.top_down_blocks, "1").blocks, "0").conv1.bn(model_neck_top_down_blocks_1_blocks_0_conv1_conv);  model_neck_top_down_blocks_1_blocks_0_conv1_conv = None
        sigmoid_45 = torch.sigmoid(model_neck_top_down_blocks_1_blocks_0_conv1_bn)
        mul_45 = model_neck_top_down_blocks_1_blocks_0_conv1_bn * sigmoid_45;  model_neck_top_down_blocks_1_blocks_0_conv1_bn = sigmoid_45 = None
        model_neck_top_down_blocks_1_blocks_0_conv2_conv = getattr(getattr(self.model.neck.top_down_blocks, "1").blocks, "0").conv2.conv(mul_45);  mul_45 = None
        model_neck_top_down_blocks_1_blocks_0_conv2_bn = getattr(getattr(self.model.neck.top_down_blocks, "1").blocks, "0").conv2.bn(model_neck_top_down_blocks_1_blocks_0_conv2_conv);  model_neck_top_down_blocks_1_blocks_0_conv2_conv = None
        sigmoid_46 = torch.sigmoid(model_neck_top_down_blocks_1_blocks_0_conv2_bn)
        mul_46 = model_neck_top_down_blocks_1_blocks_0_conv2_bn * sigmoid_46;  model_neck_top_down_blocks_1_blocks_0_conv2_bn = sigmoid_46 = None
        cat_10 = torch.cat((mul_46, mul_43), dim = 1);  mul_46 = mul_43 = None
        model_neck_top_down_blocks_1_final_conv_conv = getattr(self.model.neck.top_down_blocks, "1").final_conv.conv(cat_10);  cat_10 = None
        model_neck_top_down_blocks_1_final_conv_bn = getattr(self.model.neck.top_down_blocks, "1").final_conv.bn(model_neck_top_down_blocks_1_final_conv_conv);  model_neck_top_down_blocks_1_final_conv_conv = None
        sigmoid_47 = torch.sigmoid(model_neck_top_down_blocks_1_final_conv_bn)
        mul_47 = model_neck_top_down_blocks_1_final_conv_bn * sigmoid_47;  model_neck_top_down_blocks_1_final_conv_bn = sigmoid_47 = None
        model_neck_downsamples_0_conv = getattr(self.model.neck.downsamples, "0").conv(mul_47)
        model_neck_downsamples_0_bn = getattr(self.model.neck.downsamples, "0").bn(model_neck_downsamples_0_conv);  model_neck_downsamples_0_conv = None
        sigmoid_48 = torch.sigmoid(model_neck_downsamples_0_bn)
        mul_48 = model_neck_downsamples_0_bn * sigmoid_48;  model_neck_downsamples_0_bn = sigmoid_48 = None
        cat_11 = torch.cat([mul_48, mul_42], 1);  mul_48 = mul_42 = None
        model_neck_bottom_up_blocks_0_short_conv_conv = getattr(self.model.neck.bottom_up_blocks, "0").short_conv.conv(cat_11)
        model_neck_bottom_up_blocks_0_short_conv_bn = getattr(self.model.neck.bottom_up_blocks, "0").short_conv.bn(model_neck_bottom_up_blocks_0_short_conv_conv);  model_neck_bottom_up_blocks_0_short_conv_conv = None
        sigmoid_49 = torch.sigmoid(model_neck_bottom_up_blocks_0_short_conv_bn)
        mul_49 = model_neck_bottom_up_blocks_0_short_conv_bn * sigmoid_49;  model_neck_bottom_up_blocks_0_short_conv_bn = sigmoid_49 = None
        model_neck_bottom_up_blocks_0_main_conv_conv = getattr(self.model.neck.bottom_up_blocks, "0").main_conv.conv(cat_11);  cat_11 = None
        model_neck_bottom_up_blocks_0_main_conv_bn = getattr(self.model.neck.bottom_up_blocks, "0").main_conv.bn(model_neck_bottom_up_blocks_0_main_conv_conv);  model_neck_bottom_up_blocks_0_main_conv_conv = None
        sigmoid_50 = torch.sigmoid(model_neck_bottom_up_blocks_0_main_conv_bn)
        mul_50 = model_neck_bottom_up_blocks_0_main_conv_bn * sigmoid_50;  model_neck_bottom_up_blocks_0_main_conv_bn = sigmoid_50 = None
        model_neck_bottom_up_blocks_0_blocks_0_conv1_conv = getattr(getattr(self.model.neck.bottom_up_blocks, "0").blocks, "0").conv1.conv(mul_50);  mul_50 = None
        model_neck_bottom_up_blocks_0_blocks_0_conv1_bn = getattr(getattr(self.model.neck.bottom_up_blocks, "0").blocks, "0").conv1.bn(model_neck_bottom_up_blocks_0_blocks_0_conv1_conv);  model_neck_bottom_up_blocks_0_blocks_0_conv1_conv = None
        sigmoid_51 = torch.sigmoid(model_neck_bottom_up_blocks_0_blocks_0_conv1_bn)
        mul_51 = model_neck_bottom_up_blocks_0_blocks_0_conv1_bn * sigmoid_51;  model_neck_bottom_up_blocks_0_blocks_0_conv1_bn = sigmoid_51 = None
        model_neck_bottom_up_blocks_0_blocks_0_conv2_conv = getattr(getattr(self.model.neck.bottom_up_blocks, "0").blocks, "0").conv2.conv(mul_51);  mul_51 = None
        model_neck_bottom_up_blocks_0_blocks_0_conv2_bn = getattr(getattr(self.model.neck.bottom_up_blocks, "0").blocks, "0").conv2.bn(model_neck_bottom_up_blocks_0_blocks_0_conv2_conv);  model_neck_bottom_up_blocks_0_blocks_0_conv2_conv = None
        sigmoid_52 = torch.sigmoid(model_neck_bottom_up_blocks_0_blocks_0_conv2_bn)
        mul_52 = model_neck_bottom_up_blocks_0_blocks_0_conv2_bn * sigmoid_52;  model_neck_bottom_up_blocks_0_blocks_0_conv2_bn = sigmoid_52 = None
        cat_12 = torch.cat((mul_52, mul_49), dim = 1);  mul_52 = mul_49 = None
        model_neck_bottom_up_blocks_0_final_conv_conv = getattr(self.model.neck.bottom_up_blocks, "0").final_conv.conv(cat_12);  cat_12 = None
        model_neck_bottom_up_blocks_0_final_conv_bn = getattr(self.model.neck.bottom_up_blocks, "0").final_conv.bn(model_neck_bottom_up_blocks_0_final_conv_conv);  model_neck_bottom_up_blocks_0_final_conv_conv = None
        sigmoid_53 = torch.sigmoid(model_neck_bottom_up_blocks_0_final_conv_bn)
        mul_53 = model_neck_bottom_up_blocks_0_final_conv_bn * sigmoid_53;  model_neck_bottom_up_blocks_0_final_conv_bn = sigmoid_53 = None
        model_neck_downsamples_1_conv = getattr(self.model.neck.downsamples, "1").conv(mul_53)
        model_neck_downsamples_1_bn = getattr(self.model.neck.downsamples, "1").bn(model_neck_downsamples_1_conv);  model_neck_downsamples_1_conv = None
        sigmoid_54 = torch.sigmoid(model_neck_downsamples_1_bn)
        mul_54 = model_neck_downsamples_1_bn * sigmoid_54;  model_neck_downsamples_1_bn = sigmoid_54 = None
        cat_13 = torch.cat([mul_54, mul_36], 1);  mul_54 = mul_36 = None
        model_neck_bottom_up_blocks_1_short_conv_conv = getattr(self.model.neck.bottom_up_blocks, "1").short_conv.conv(cat_13)
        model_neck_bottom_up_blocks_1_short_conv_bn = getattr(self.model.neck.bottom_up_blocks, "1").short_conv.bn(model_neck_bottom_up_blocks_1_short_conv_conv);  model_neck_bottom_up_blocks_1_short_conv_conv = None
        sigmoid_55 = torch.sigmoid(model_neck_bottom_up_blocks_1_short_conv_bn)
        mul_55 = model_neck_bottom_up_blocks_1_short_conv_bn * sigmoid_55;  model_neck_bottom_up_blocks_1_short_conv_bn = sigmoid_55 = None
        model_neck_bottom_up_blocks_1_main_conv_conv = getattr(self.model.neck.bottom_up_blocks, "1").main_conv.conv(cat_13);  cat_13 = None
        model_neck_bottom_up_blocks_1_main_conv_bn = getattr(self.model.neck.bottom_up_blocks, "1").main_conv.bn(model_neck_bottom_up_blocks_1_main_conv_conv);  model_neck_bottom_up_blocks_1_main_conv_conv = None
        sigmoid_56 = torch.sigmoid(model_neck_bottom_up_blocks_1_main_conv_bn)
        mul_56 = model_neck_bottom_up_blocks_1_main_conv_bn * sigmoid_56;  model_neck_bottom_up_blocks_1_main_conv_bn = sigmoid_56 = None
        model_neck_bottom_up_blocks_1_blocks_0_conv1_conv = getattr(getattr(self.model.neck.bottom_up_blocks, "1").blocks, "0").conv1.conv(mul_56);  mul_56 = None
        model_neck_bottom_up_blocks_1_blocks_0_conv1_bn = getattr(getattr(self.model.neck.bottom_up_blocks, "1").blocks, "0").conv1.bn(model_neck_bottom_up_blocks_1_blocks_0_conv1_conv);  model_neck_bottom_up_blocks_1_blocks_0_conv1_conv = None
        sigmoid_57 = torch.sigmoid(model_neck_bottom_up_blocks_1_blocks_0_conv1_bn)
        mul_57 = model_neck_bottom_up_blocks_1_blocks_0_conv1_bn * sigmoid_57;  model_neck_bottom_up_blocks_1_blocks_0_conv1_bn = sigmoid_57 = None
        model_neck_bottom_up_blocks_1_blocks_0_conv2_conv = getattr(getattr(self.model.neck.bottom_up_blocks, "1").blocks, "0").conv2.conv(mul_57);  mul_57 = None
        model_neck_bottom_up_blocks_1_blocks_0_conv2_bn = getattr(getattr(self.model.neck.bottom_up_blocks, "1").blocks, "0").conv2.bn(model_neck_bottom_up_blocks_1_blocks_0_conv2_conv);  model_neck_bottom_up_blocks_1_blocks_0_conv2_conv = None
        sigmoid_58 = torch.sigmoid(model_neck_bottom_up_blocks_1_blocks_0_conv2_bn)
        mul_58 = model_neck_bottom_up_blocks_1_blocks_0_conv2_bn * sigmoid_58;  model_neck_bottom_up_blocks_1_blocks_0_conv2_bn = sigmoid_58 = None
        cat_14 = torch.cat((mul_58, mul_55), dim = 1);  mul_58 = mul_55 = None
        model_neck_bottom_up_blocks_1_final_conv_conv = getattr(self.model.neck.bottom_up_blocks, "1").final_conv.conv(cat_14);  cat_14 = None
        model_neck_bottom_up_blocks_1_final_conv_bn = getattr(self.model.neck.bottom_up_blocks, "1").final_conv.bn(model_neck_bottom_up_blocks_1_final_conv_conv);  model_neck_bottom_up_blocks_1_final_conv_conv = None
        sigmoid_59 = torch.sigmoid(model_neck_bottom_up_blocks_1_final_conv_bn)
        mul_59 = model_neck_bottom_up_blocks_1_final_conv_bn * sigmoid_59;  model_neck_bottom_up_blocks_1_final_conv_bn = sigmoid_59 = None
        model_neck_out_convs_0_conv = getattr(self.model.neck.out_convs, "0").conv(mul_47);  mul_47 = None
        model_neck_out_convs_0_bn = getattr(self.model.neck.out_convs, "0").bn(model_neck_out_convs_0_conv);  model_neck_out_convs_0_conv = None
        sigmoid_60 = torch.sigmoid(model_neck_out_convs_0_bn)
        mul_60 = model_neck_out_convs_0_bn * sigmoid_60;  model_neck_out_convs_0_bn = sigmoid_60 = None
        model_neck_out_convs_1_conv = getattr(self.model.neck.out_convs, "1").conv(mul_53);  mul_53 = None
        model_neck_out_convs_1_bn = getattr(self.model.neck.out_convs, "1").bn(model_neck_out_convs_1_conv);  model_neck_out_convs_1_conv = None
        sigmoid_61 = torch.sigmoid(model_neck_out_convs_1_bn)
        mul_61 = model_neck_out_convs_1_bn * sigmoid_61;  model_neck_out_convs_1_bn = sigmoid_61 = None
        model_neck_out_convs_2_conv = getattr(self.model.neck.out_convs, "2").conv(mul_59);  mul_59 = None
        model_neck_out_convs_2_bn = getattr(self.model.neck.out_convs, "2").bn(model_neck_out_convs_2_conv);  model_neck_out_convs_2_conv = None
        sigmoid_62 = torch.sigmoid(model_neck_out_convs_2_bn)
        mul_62 = model_neck_out_convs_2_bn * sigmoid_62;  model_neck_out_convs_2_bn = sigmoid_62 = None
        model_bbox_head_multi_level_cls_convs_0_0_conv = getattr(getattr(self.model.bbox_head.multi_level_cls_convs, "0"), "0").conv(mul_60)
        model_bbox_head_multi_level_cls_convs_0_0_bn = getattr(getattr(self.model.bbox_head.multi_level_cls_convs, "0"), "0").bn(model_bbox_head_multi_level_cls_convs_0_0_conv);  model_bbox_head_multi_level_cls_convs_0_0_conv = None
        sigmoid_63 = torch.sigmoid(model_bbox_head_multi_level_cls_convs_0_0_bn)
        mul_63 = model_bbox_head_multi_level_cls_convs_0_0_bn * sigmoid_63;  model_bbox_head_multi_level_cls_convs_0_0_bn = sigmoid_63 = None
        model_bbox_head_multi_level_cls_convs_0_1_conv = getattr(getattr(self.model.bbox_head.multi_level_cls_convs, "0"), "1").conv(mul_63);  mul_63 = None
        model_bbox_head_multi_level_cls_convs_0_1_bn = getattr(getattr(self.model.bbox_head.multi_level_cls_convs, "0"), "1").bn(model_bbox_head_multi_level_cls_convs_0_1_conv);  model_bbox_head_multi_level_cls_convs_0_1_conv = None
        sigmoid_64 = torch.sigmoid(model_bbox_head_multi_level_cls_convs_0_1_bn)
        mul_64 = model_bbox_head_multi_level_cls_convs_0_1_bn * sigmoid_64;  model_bbox_head_multi_level_cls_convs_0_1_bn = sigmoid_64 = None
        model_bbox_head_multi_level_reg_convs_0_0_conv = getattr(getattr(self.model.bbox_head.multi_level_reg_convs, "0"), "0").conv(mul_60);  mul_60 = None
        model_bbox_head_multi_level_reg_convs_0_0_bn = getattr(getattr(self.model.bbox_head.multi_level_reg_convs, "0"), "0").bn(model_bbox_head_multi_level_reg_convs_0_0_conv);  model_bbox_head_multi_level_reg_convs_0_0_conv = None
        sigmoid_65 = torch.sigmoid(model_bbox_head_multi_level_reg_convs_0_0_bn)
        mul_65 = model_bbox_head_multi_level_reg_convs_0_0_bn * sigmoid_65;  model_bbox_head_multi_level_reg_convs_0_0_bn = sigmoid_65 = None
        model_bbox_head_multi_level_reg_convs_0_1_conv = getattr(getattr(self.model.bbox_head.multi_level_reg_convs, "0"), "1").conv(mul_65);  mul_65 = None
        model_bbox_head_multi_level_reg_convs_0_1_bn = getattr(getattr(self.model.bbox_head.multi_level_reg_convs, "0"), "1").bn(model_bbox_head_multi_level_reg_convs_0_1_conv);  model_bbox_head_multi_level_reg_convs_0_1_conv = None
        sigmoid_66 = torch.sigmoid(model_bbox_head_multi_level_reg_convs_0_1_bn)
        mul_66 = model_bbox_head_multi_level_reg_convs_0_1_bn * sigmoid_66;  model_bbox_head_multi_level_reg_convs_0_1_bn = sigmoid_66 = None
        model_bbox_head_multi_level_conv_cls_0 = getattr(self.model.bbox_head.multi_level_conv_cls, "0")(mul_64);  mul_64 = None
        model_bbox_head_multi_level_conv_reg_0 = getattr(self.model.bbox_head.multi_level_conv_reg, "0")(mul_66)
        model_bbox_head_multi_level_conv_obj_0 = getattr(self.model.bbox_head.multi_level_conv_obj, "0")(mul_66);  mul_66 = None
        model_bbox_head_multi_level_cls_convs_1_0_conv = getattr(getattr(self.model.bbox_head.multi_level_cls_convs, "1"), "0").conv(mul_61)
        model_bbox_head_multi_level_cls_convs_1_0_bn = getattr(getattr(self.model.bbox_head.multi_level_cls_convs, "1"), "0").bn(model_bbox_head_multi_level_cls_convs_1_0_conv);  model_bbox_head_multi_level_cls_convs_1_0_conv = None
        sigmoid_67 = torch.sigmoid(model_bbox_head_multi_level_cls_convs_1_0_bn)
        mul_67 = model_bbox_head_multi_level_cls_convs_1_0_bn * sigmoid_67;  model_bbox_head_multi_level_cls_convs_1_0_bn = sigmoid_67 = None
        model_bbox_head_multi_level_cls_convs_1_1_conv = getattr(getattr(self.model.bbox_head.multi_level_cls_convs, "1"), "1").conv(mul_67);  mul_67 = None
        model_bbox_head_multi_level_cls_convs_1_1_bn = getattr(getattr(self.model.bbox_head.multi_level_cls_convs, "1"), "1").bn(model_bbox_head_multi_level_cls_convs_1_1_conv);  model_bbox_head_multi_level_cls_convs_1_1_conv = None
        sigmoid_68 = torch.sigmoid(model_bbox_head_multi_level_cls_convs_1_1_bn)
        mul_68 = model_bbox_head_multi_level_cls_convs_1_1_bn * sigmoid_68;  model_bbox_head_multi_level_cls_convs_1_1_bn = sigmoid_68 = None
        model_bbox_head_multi_level_reg_convs_1_0_conv = getattr(getattr(self.model.bbox_head.multi_level_reg_convs, "1"), "0").conv(mul_61);  mul_61 = None
        model_bbox_head_multi_level_reg_convs_1_0_bn = getattr(getattr(self.model.bbox_head.multi_level_reg_convs, "1"), "0").bn(model_bbox_head_multi_level_reg_convs_1_0_conv);  model_bbox_head_multi_level_reg_convs_1_0_conv = None
        sigmoid_69 = torch.sigmoid(model_bbox_head_multi_level_reg_convs_1_0_bn)
        mul_69 = model_bbox_head_multi_level_reg_convs_1_0_bn * sigmoid_69;  model_bbox_head_multi_level_reg_convs_1_0_bn = sigmoid_69 = None
        model_bbox_head_multi_level_reg_convs_1_1_conv = getattr(getattr(self.model.bbox_head.multi_level_reg_convs, "1"), "1").conv(mul_69);  mul_69 = None
        model_bbox_head_multi_level_reg_convs_1_1_bn = getattr(getattr(self.model.bbox_head.multi_level_reg_convs, "1"), "1").bn(model_bbox_head_multi_level_reg_convs_1_1_conv);  model_bbox_head_multi_level_reg_convs_1_1_conv = None
        sigmoid_70 = torch.sigmoid(model_bbox_head_multi_level_reg_convs_1_1_bn)
        mul_70 = model_bbox_head_multi_level_reg_convs_1_1_bn * sigmoid_70;  model_bbox_head_multi_level_reg_convs_1_1_bn = sigmoid_70 = None
        model_bbox_head_multi_level_conv_cls_1 = getattr(self.model.bbox_head.multi_level_conv_cls, "1")(mul_68);  mul_68 = None
        model_bbox_head_multi_level_conv_reg_1 = getattr(self.model.bbox_head.multi_level_conv_reg, "1")(mul_70)
        model_bbox_head_multi_level_conv_obj_1 = getattr(self.model.bbox_head.multi_level_conv_obj, "1")(mul_70);  mul_70 = None
        model_bbox_head_multi_level_cls_convs_2_0_conv = getattr(getattr(self.model.bbox_head.multi_level_cls_convs, "2"), "0").conv(mul_62)
        model_bbox_head_multi_level_cls_convs_2_0_bn = getattr(getattr(self.model.bbox_head.multi_level_cls_convs, "2"), "0").bn(model_bbox_head_multi_level_cls_convs_2_0_conv);  model_bbox_head_multi_level_cls_convs_2_0_conv = None
        sigmoid_71 = torch.sigmoid(model_bbox_head_multi_level_cls_convs_2_0_bn)
        mul_71 = model_bbox_head_multi_level_cls_convs_2_0_bn * sigmoid_71;  model_bbox_head_multi_level_cls_convs_2_0_bn = sigmoid_71 = None
        model_bbox_head_multi_level_cls_convs_2_1_conv = getattr(getattr(self.model.bbox_head.multi_level_cls_convs, "2"), "1").conv(mul_71);  mul_71 = None
        model_bbox_head_multi_level_cls_convs_2_1_bn = getattr(getattr(self.model.bbox_head.multi_level_cls_convs, "2"), "1").bn(model_bbox_head_multi_level_cls_convs_2_1_conv);  model_bbox_head_multi_level_cls_convs_2_1_conv = None
        sigmoid_72 = torch.sigmoid(model_bbox_head_multi_level_cls_convs_2_1_bn)
        mul_72 = model_bbox_head_multi_level_cls_convs_2_1_bn * sigmoid_72;  model_bbox_head_multi_level_cls_convs_2_1_bn = sigmoid_72 = None
        model_bbox_head_multi_level_reg_convs_2_0_conv = getattr(getattr(self.model.bbox_head.multi_level_reg_convs, "2"), "0").conv(mul_62);  mul_62 = None
        model_bbox_head_multi_level_reg_convs_2_0_bn = getattr(getattr(self.model.bbox_head.multi_level_reg_convs, "2"), "0").bn(model_bbox_head_multi_level_reg_convs_2_0_conv);  model_bbox_head_multi_level_reg_convs_2_0_conv = None
        sigmoid_73 = torch.sigmoid(model_bbox_head_multi_level_reg_convs_2_0_bn)
        mul_73 = model_bbox_head_multi_level_reg_convs_2_0_bn * sigmoid_73;  model_bbox_head_multi_level_reg_convs_2_0_bn = sigmoid_73 = None
        model_bbox_head_multi_level_reg_convs_2_1_conv = getattr(getattr(self.model.bbox_head.multi_level_reg_convs, "2"), "1").conv(mul_73);  mul_73 = None
        model_bbox_head_multi_level_reg_convs_2_1_bn = getattr(getattr(self.model.bbox_head.multi_level_reg_convs, "2"), "1").bn(model_bbox_head_multi_level_reg_convs_2_1_conv);  model_bbox_head_multi_level_reg_convs_2_1_conv = None
        sigmoid_74 = torch.sigmoid(model_bbox_head_multi_level_reg_convs_2_1_bn)
        mul_74 = model_bbox_head_multi_level_reg_convs_2_1_bn * sigmoid_74;  model_bbox_head_multi_level_reg_convs_2_1_bn = sigmoid_74 = None
        model_bbox_head_multi_level_conv_cls_2 = getattr(self.model.bbox_head.multi_level_conv_cls, "2")(mul_72);  mul_72 = None
        model_bbox_head_multi_level_conv_reg_2 = getattr(self.model.bbox_head.multi_level_conv_reg, "2")(mul_74)
        model_bbox_head_multi_level_conv_obj_2 = getattr(self.model.bbox_head.multi_level_conv_obj, "2")(mul_74);  mul_74 = None
        return ([model_bbox_head_multi_level_conv_cls_0, model_bbox_head_multi_level_conv_cls_1, model_bbox_head_multi_level_conv_cls_2], [model_bbox_head_multi_level_conv_reg_0, model_bbox_head_multi_level_conv_reg_1, model_bbox_head_multi_level_conv_reg_2], [model_bbox_head_multi_level_conv_obj_0, model_bbox_head_multi_level_conv_obj_1, model_bbox_head_multi_level_conv_obj_2])
        
