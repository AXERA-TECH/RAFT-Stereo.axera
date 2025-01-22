import argparse
import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--left",
        type=str,
        required=True,
        help="Path to left image.",
    )
    parser.add_argument(
        "--right",
        type=str,
        required=True,
        help="Path to right image.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to ONNX model.",
    )
    
    return parser.parse_args()


def infer(left: str, right: str, model: str):

    session = ort.InferenceSession(
        model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_info = []
    input_tensors = session.get_inputs()
    for tensor in input_tensors:
        info = dict(name=tensor.name , type=tensor.type, shape=tensor.shape)
        input_info.append(info)

    H,W = input_info[0]['shape'][2:4]

    left_raw = cv2.imread(left)
    image_left = cv2.cvtColor(left_raw, cv2.COLOR_BGR2RGB) 
    orig_h_left, orig_w_left = image_left.shape[:2]
    image_left = cv2.resize(image_left, (W,H) )
   
    image_left = image_left.transpose(2,0,1)
    image_left = image_left[None].astype(np.float32)

    right_raw = cv2.imread(right)
    image_right = cv2.cvtColor(right_raw, cv2.COLOR_BGR2RGB) 
    orig_h_right, orig_w_right = image_right.shape[:2]
    image_right = cv2.resize(image_right, (W,H) )
   
    image_right = image_right.transpose(2,0,1)
    image_right = image_right[None].astype(np.float32)

    assert orig_h_left == orig_h_right and orig_w_left == orig_w_right

    
    flow_up = session.run(None, {input_info[0]['name']: image_left, input_info[1]['name']:image_right})[0]
    
    flow_up = cv2.resize(flow_up[0,0], (orig_w_left, orig_h_left))
    flow_up *= orig_w_left/W
    
    output = np.abs(flow_up)
    
    plt.imsave(f"output-onnx.png", output, cmap='jet')


    return output


if __name__ == "__main__":
    args = parse_args()
    infer(**vars(args))
