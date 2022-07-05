import cv2 as cv
import numpy as np
import onnx
import torch
import math
import mmcv
import onnxruntime as rt
from mmdet.apis import inference_detector,init_detector,show_result_pyplot

def path_check(file_path):
    import os
    if os.path.exists(file_path):
        os.remove(file_path)
        print(file_path + " has been deleted sucessfully!")

def remove_initializer_from_input(input_onnx, output_onnx):
    model = onnx.load(input_onnx)
    if model.ir_version < 4:
        print(
            'Model with ir_version below 4 requires to include initilizer in graph input'
        )
        return

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    onnx.save(model, output_onnx)

# config = '/home/wyh/mmdetection/configs/yolox/yolox_s_8x8_300e_coco.py'
# checkpoint = '/home/wyh/mmdetection/checkpoints/yolox_s_8x8_300e_coco.pth'
# output_file = 'yolox_s_8x8_300e_coco.onnx'
config = '/home/wyh/mmdetection/configs/yolox/yolox_x_8x8_300e_coco.py'
checkpoint = '/home/wyh/mmdetection/checkpoints/yolox_x_8x8_300e_coco.pth'
output_file = 'yolox_x_8x8_300e_coco.onnx'

device = 'cuda:0'

input_img = '/home/wyh/mmdetection/demo/demo.jpg'
input_shape = (1, 3, 640, 640)
one_img = mmcv.imread(input_img)
print("one_img.shape:", one_img.shape)

ratio = min(input_shape[2] / one_img.shape[0], input_shape[3] / one_img.shape[1])
print("ratio:", ratio)
padded_img = np.ones((input_shape[2], input_shape[3], 3), dtype=np.uint8) * 114
# padded_img[:,:,1:] = 0
resized_img = cv.resize(
    one_img,
    (math.ceil(one_img.shape[1] * ratio), math.ceil(one_img.shape[0] * ratio)),
    interpolation=cv.INTER_LINEAR
).astype(np.uint8)
print("resized_img.shape:", resized_img.shape)
padded_img[: math.ceil(one_img.shape[0] * ratio), : math.ceil(one_img.shape[1] * ratio)] = resized_img
one_img = padded_img
mean = [0., 0., 0.]
std = [1., 1., 1.]
mean = np.array(mean, dtype=np.float32)
std = np.array(std, dtype=np.float32)
one_img = mmcv.imnormalize(one_img, mean, std, to_rgb=False)
one_img = one_img.transpose(2, 0, 1)
one_img = torch.from_numpy(one_img).unsqueeze(0).float().requires_grad_(True)

if 0:
    print("------start preprocesse-------------")
    onnx_preprocess_txt = "onnx_preprocess.txt"
    path_check(onnx_preprocess_txt)
    print("one_img.shape:",one_img.shape)
    onnx_preprocess_data = one_img.detach().numpy().flatten()
    for i in range(len(onnx_preprocess_data)):
        with open(onnx_preprocess_txt,'a+') as f:
            f.write(str(onnx_preprocess_data[i]) + "\n")
    pytorch_inference_preprocess_txt = "pytorch_inference_preprocess.txt"
    pytorch_inference_preprocess_data = []
    for line in open(pytorch_inference_preprocess_txt,'r'):
        data = float(line.split('\n')[0])
        pytorch_inference_preprocess_data.append(data)
    print("len_pytorch_inference_preprocess_data:",len(pytorch_inference_preprocess_data))
    max_diff = 0
    diff_all = 0
    for i in range(len(onnx_preprocess_data)):
        diff = abs(onnx_preprocess_data[i] - pytorch_inference_preprocess_data[i])
        diff_all += diff
        if diff > max_diff:
            max_diff = diff
            print(str(i) + ": " + str(onnx_preprocess_data[i]) + ", " + str(pytorch_inference_preprocess_data[i]))
    print("begin compare bettween " + pytorch_inference_preprocess_txt + " and " + onnx_preprocess_txt)
    print("preprocess max diff:",max_diff)
    print("preprocess average diff:",diff_all / len(onnx_preprocess_data))
    print("------end preprocesse----------------")


one_img = one_img.to(device)

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = init_detector(config, checkpoint, device = device)
    
    def forward(self, x):
        x = self.model.backbone(x)
        x = self.model.neck(x)
        x = self.model.bbox_head(x)
        return x 

model = MyModel().eval()

torch.onnx.export(
    model, 
    one_img,
    output_file,
    input_names=['input'],
    export_params=True,
    keep_initializers_as_inputs=True,
    verbose=True,
    opset_version=11)
print(f'Successfully exported ONNX model: {output_file}')
print("end exchange pytorch model to ONNX model.....")

# remove_initializer_from_input(output_file, output_file)

from onnxsim import simplify
print("output_file:",output_file)
onnx_model = onnx.load(output_file)# load onnx model
onnx_model_sim_file = output_file.split('.')[0] + "_sim.onnx"
model_simp, check_ok = simplify(onnx_model)
if check_ok:
    print("check_ok:",check_ok)
    onnx.save(model_simp, onnx_model_sim_file)
    print(f'Successfully simplified ONNX model: {onnx_model_sim_file}')


print("#################### start get pytorch inference result ####################")
print("start pytorch inference .......")
pytorch_result = model(one_img)
print("end pytorch inference .......")
pytorch_results = []
for i in range(len(pytorch_result)):
    for j in range(len(pytorch_result[i])):
        print(str(i) + "," + str(j) + ":" + str(pytorch_result[i][j].shape))
        '''
        0,0:torch.Size([1, 80, 80, 80])
        0,1:torch.Size([1, 80, 40, 40])
        0,2:torch.Size([1, 80, 20, 20])
        1,0:torch.Size([1, 4, 80, 80])
        1,1:torch.Size([1, 4, 40, 40])
        1,2:torch.Size([1, 4, 20, 20])
        2,0:torch.Size([1, 1, 80, 80])
        2,1:torch.Size([1, 1, 40, 40])
        2,2:torch.Size([1, 1, 20, 20])
        '''
        data = pytorch_result[i][j].cpu().detach().numpy().flatten()
        pytorch_results.extend(data)
print("#################### end get pytorch inference result ####################")


print("#################### start onnxruntime inference ####################")
onnx_model = onnx.load(onnx_model_sim_file)
input_all = [node.name for node in onnx_model.graph.input]
output_all = [node.name for node in onnx_model.graph.output]
print("input_all:", input_all)
print("ouput_all:\n",output_all)
input_initializer = [
    node.name for node in onnx_model.graph.initializer
]
print("input_initializer:\n", input_initializer)
net_feed_input = list(set(input_all) - set(input_initializer))
print("net_feed_input:", net_feed_input)
sess = rt.InferenceSession(onnx_model_sim_file)
input_data = one_img.cpu().detach().numpy()
onnx_result = sess.run(
    None, {net_feed_input[0]: input_data})
onnx_inference_results = []
for i in range(len(onnx_result)):
    onnx_inference_results.extend(onnx_result[i].flatten())
    print("onnx_inference_results["+str(i) + "].shape:", onnx_result[i].shape)
print("#################### end onnxruntime inference ####################")

print("len_onnx_results:",len(onnx_inference_results))
print("len_pytorch_results:", len(pytorch_results))
assert len(pytorch_results) == len(onnx_inference_results),'len(pytorch_results) != len(onnx_results)'

print("#################### start compare  bettween pytorch inference result and onnx inference result ####################")
diff = 0.0
maxdiff = 0.0
onnx_result_txt = "onnx_result.txt"
path_check(onnx_result_txt)
pytorch_result_txt = "pytorch_result.txt"
path_check(pytorch_result_txt)
for i in range(len(onnx_inference_results)):
    diff += abs(onnx_inference_results[i] - pytorch_results[i])
    if abs(onnx_inference_results[i] - pytorch_results[i]) > maxdiff:
        maxdiff = abs(onnx_inference_results[i] - pytorch_results[i])
    with open(onnx_result_txt,'a+') as f:
        f.write(str(onnx_inference_results[i]) + "\n")
    with open(pytorch_result_txt,'a+') as f:
        f.write(str(pytorch_results[i]) + "\n")

print("diff bettween onnx and pytorch:",diff)
print("average_diff bettween onnx and pytorch:",diff/len(onnx_inference_results))
print("maxdiff bettween onnx and pytorch:",maxdiff)

if diff / len(onnx_inference_results) < 1e-04:
    print('The numerical values are same between Pytorch and ONNX')
else:
    print('The outputs are different between Pytorch and ONNX')
print("#################### end compare  bettween pytorch inference result and onnx inference result ####################")
