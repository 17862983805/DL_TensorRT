from matplotlib.pyplot import show
import mmdet
import mmcv
print(mmdet.__version__)
print(mmcv.__version__)
import cv2 as cv

from mmdet.apis import inference_detector,init_detector,show_result_pyplot

config = "/home/wyh/mmdetection/configs/yolox/yolox_s_8x8_300e_coco.py"
checkpoint = "/home/wyh/mmdetection/checkpoints/yolox_s_8x8_300e_coco.pth"

# initialize the detector
model=init_detector(config,checkpoint,device='cuda:0')

# Use the detector to do inference
img='/home/wyh/mmdetection/demo/demo.jpg'
img = '/home/wyh/YOLOX/assets/dog.jpg'
results=inference_detector(model,img)

# Plot the result
output_file = '/home/wyh/mmdetection/yolox_Tensorrt/yolox_pytorch_inference_result.jpg'
img = show_result_pyplot(model,img,results,score_thr=0.25, out_file=output_file)
