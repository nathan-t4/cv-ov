# ov-cv

Benchmarking performance of OpenVINO for YOLO26 models

Hardware:
- CPU: 12th Gen Intel Core i7-12800H
- GPU: Intel UHD Graphics (iGPU)

Models:
- yolo26n
- yolo26n-pose

Results:
- Around 3x inference speed on GPU vs CPU for both yolo26n and yolo26n-pose (see trials/)


CPU Inference Times with yolo26n (`main.py`)
![CPU OpenVINO yolo26n](trials/yolo26n/CPU_openvino_yolo26n_inference.png)


GPU Inference Times with yolo26n (`main.py`)
![GPU OpenVINO yolo26n](trials/yolo26n/GPU.0_openvino_yolo26n_inference.png)

