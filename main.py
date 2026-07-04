import os
import cv2
import time
import yaml
import numpy as np
import torch
import openvino as ov
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.engine.results import Results


def convert_model(path: str, format: str = "openvino"):
    model = YOLO(path)
    model.export(format=format)

def get_device_name(core: ov.Core, device: str):
    return core.get_property(device, "FULL_DEVICE_NAME")

def load_metadata(model_dir: str, model_name: str) -> dict:
    path = f"{model_dir}/{model_name}_openvino_model/metadata.yaml"
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f)
    raise FileNotFoundError(f"No metadata.yaml found under {model_dir}")

def to_yolo_result(
    preds: np.ndarray,
    orig_img: np.ndarray,
    names: dict[int, str],
    imgsz: tuple[int, int],
    conf: float = 0.5,
    kpt_shape: tuple[int, int] | None = None,
) -> Results:
    """Wrap end2end OpenVINO output as an Ultralytics Results for .plot().

    Expects preds shaped (1, N, 6) for detect or (1, N, 6+K*D) for pose,
    with boxes in model-input pixel space (matches simple resize preprocess).
    """
    if isinstance(preds, Results):
        return preds
    
    if preds.ndim == 3:
        preds = preds[0]

    preds = preds[preds[:, 4] >= conf]

    oh, ow = orig_img.shape[:2]
    mh, mw = imgsz
    sx, sy = ow / mw, oh / mh

    boxes = None
    keypoints = None
    if len(preds):
        boxes = preds[:, :6].copy()
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
        boxes = torch.from_numpy(boxes)

        if preds.shape[1] > 6:
            nk, nd = kpt_shape or (17, 3)
            kpts = preds[:, 6:].reshape(-1, nk, nd).copy()
            kpts[:, :, 0] *= sx
            kpts[:, :, 1] *= sy
            keypoints = torch.from_numpy(kpts)

    return Results(orig_img=orig_img, path="", names=names, boxes=boxes, keypoints=keypoints)

def plot_inference(times: list[float], output_path: str | None = None):
    plt.plot(times)
    plt.ylabel("Inference Time (s)")
    plt.xlabel("Frame Number")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close()

def benchmark(
    model_name: str = "yolo26n",
    device: str = "CPU", 
    model_format: str = "openvino"
):

    model_dir = f"models/{model_name}"
    meta = load_metadata(model_dir, model_name)
    names = {int(k): v for k, v in meta["names"].items()}
    kpt_shape = tuple(meta["kpt_shape"]) if meta.get("kpt_shape") else None

    ie = ov.Core()
    print(get_device_name(ie, device))

    inference_times = []

    if model_format == "onnx":
        model = ie.read_model(model=f"{model_dir}/{model_name}.onnx")
        compiled_model = ie.compile_model(model, device)
    elif model_format == "openvino":
        model = ie.read_model(model=f"{model_dir}/{model_name}_openvino_model/{model_name}.xml")
        compiled_model = ie.compile_model(model, device)
    # elif model_format == "yolo":
    #     compiled_model = YOLO(f"{model_dir}/{model_name}.pt")
    else:
        raise ValueError(
            f"Invalid model format: {model_format}. "
            f"Supported formats: ['onnx', 'openvino']"
        )

    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    N, C, H, W = input_layer.shape

    cap = cv2.VideoCapture(0)

    t = 0
    warm_up = 50
    t_max = 2e2
    while t < t_max:
        ret, frame = cap.read()

        resized_frame = cv2.resize(frame, (W, H))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        # normalize 8-bit image inputs to [0-1]
        input_image = rgb_frame.transpose(2, 0, 1).astype("float32") / 255.0
        input_image = input_image.reshape((N, C, H, W))

        start_time = time.perf_counter()
        preds = compiled_model([input_image])[output_layer]
        end_time = time.perf_counter()
        inference_times.append(end_time - start_time)

        result = to_yolo_result(preds, frame, names, imgsz=(H, W), kpt_shape=kpt_shape)
        annotated_frame = result.plot()

        cv2.imshow(model_name, annotated_frame)
        t += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    output_dir = f"trials/{model_name}"
    output_path = f"{output_dir}/{device}_{model_format}_{model_name}_inference.png"
    plot_inference(inference_times, output_path)

    inference_times_warm = inference_times[warm_up:]
    print(f"Average inference time (omitting warm-up): {sum(inference_times_warm) / len(inference_times_warm)} seconds")
    print(f"Output path: {output_path}")

if __name__ == "__main__":
    # convert_model("models/yolo26n-pose/yolo26n-pose.pt", "onnx")
    benchmark(model_name="yolo26n", device="GPU.0", model_format="openvino")
    
