import cv2 as cv
import numpy as np
import onnx
import torch
import math
import mmcv
import onnxruntime as rt
from mmdet.apis import init_detector
from torch.fx import symbolic_trace

def path_check(file_path):
    import os
    if os.path.exists(file_path):
        os.remove(file_path)
        print(file_path + " has been deleted sucessfully!")

config = '../configs/yolox/yolox_tiny_8x8_300e_coco.py'
checkpoint = '../checkpoints/yolox_tiny_8x8_300e_coco.pth'
output_file = 'yolox_tiny_8x8_300e_coco.onnx'

device = 'cuda:0'

input_img = '../demo/demo.jpg'
input_shape = (1, 3, 416, 416)
one_img = mmcv.imread(input_img)
print("one_img.shape:", one_img.shape)

ratio = min(input_shape[2] / one_img.shape[0], input_shape[3] / one_img.shape[1])
print("ratio:", ratio)
padded_img = np.ones((input_shape[2], input_shape[3], 3), dtype=np.uint8) * 114
padded_img[:,:,1:] = 0
resized_img = cv.resize(
    one_img,
    (math.ceil(one_img.shape[1] * ratio), math.ceil(one_img.shape[0] * ratio)),
    interpolation=cv.INTER_LINEAR
).astype(np.uint8)
print("resized_img.shape:", resized_img.shape)
padded_img[: math.ceil(one_img.shape[0] * ratio), : math.ceil(one_img.shape[1] * ratio)] = resized_img
one_img = padded_img
mean = [123.675, 116.28, 103.53]
std = [58.395, 57.12, 57.375]
mean = np.array(mean, dtype=np.float32)
std = np.array(std, dtype=np.float32)
one_img = mmcv.imnormalize(one_img, mean, std, to_rgb=True)
one_img = one_img.transpose(2, 0, 1)
one_img = torch.from_numpy(one_img).unsqueeze(0).float().requires_grad_(True)
one_img = one_img.to(device)

one_img = torch.ones(input_shape).cuda()

class Yolox_tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = init_detector(config, checkpoint, device = device)
    
    def forward(self, x):
        x = self.model.backbone(x)
        x = self.model.neck(x)
        x = self.model.bbox_head(x)
        return x 
model = Yolox_tiny()
print("model:\n", model)
model = model.eval()
pytorch_result_origin = model(one_img)
print("pytorch_result_origin:\n", pytorch_result_origin)

# # 符号追踪这个模块
symbolic_traced : torch.fx.GraphModule = symbolic_trace(model)

# # 推理
print("result:\n", symbolic_traced(one_img))

# # 中间表示
yolox_tiny_graph = symbolic_traced.graph
print("yolox_tiny_graph:\n", yolox_tiny_graph)

# # 生成代码 str格式
yolox_tiny_model_code = symbolic_traced.code
print("yolox_tiny_model_code:\n", yolox_tiny_model_code)

# # 打印graph的所有node
gm = torch.fx.symbolic_trace(model)
gm.graph.print_tabular()

symbolic_traced.to_folder("yolox_tiny_code")

# from yolox_tiny_code import FxModule
# model = FxModule().eval()
# new_pytorch_result = model(one_img)
# print("new_pytorch_result:\n", new_pytorch_result)