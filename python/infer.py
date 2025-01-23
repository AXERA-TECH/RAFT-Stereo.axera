import argparse
import cv2
# from PIL import Image
enable_cv2 = True

import numpy as np
import matplotlib.pyplot as plt
from axengine import InferenceSession

def bilinear_resize_numpy(array, new_h, new_w):
    h, w = array.shape
    x_ratio = w / new_w
    y_ratio = h / new_h
    
    resized = np.zeros((new_h, new_w), dtype=array.dtype)
    
    for i in range(new_h):
        for j in range(new_w):
            x = j * x_ratio
            y = i * y_ratio
            
            x_floor = int(x)
            y_floor = int(y)
            x_ceil = min(x_floor + 1, w - 1)
            y_ceil = min(y_floor + 1, h - 1)
            
            dx = x - x_floor
            dy = y - y_floor
            
            a = array[y_floor, x_floor]
            b = array[y_floor, x_ceil]
            c = array[y_ceil, x_floor]
            d = array[y_ceil, x_ceil]
            
            resized[i, j] = a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) + c * (1 - dx) * dy + d * dx * dy
    
    return resized


def resize_disp(disp, target_width, target_height, use_cv2=True):
    if use_cv2:
        disp = cv2.resize(disp, (target_width, target_height))
    else:
        # This implementation is slower than cv2.resize
        disp = bilinear_resize_numpy(disp, target_height, target_width)

    return disp

def load_and_preprocess_image(image_path, target_width, target_height, use_cv2=True):

    if use_cv2:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        orig_height, orig_width = img.shape[:2]
        img_resized = cv2.resize(img, (target_width,target_height) )
        img_batch = img_resized[None]
    else:
        img = Image.open(image_path).convert('RGB')
        orig_width, orig_height = img.size
        img_resized = img.resize((target_width, target_height))
        img_array = np.array(img_resized)  #
        img_batch = img_array[None]        # 
    
    return img_batch, (orig_height, orig_width)

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
    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="Width of input image.",
    )
    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="Height of input image.",
    )
    
    return parser.parse_args()


def infer(left: str, right: str, model: str, width:int, height:int):
    
    image_left, (orig_h_left, orig_w_left) = load_and_preprocess_image(left, width, height, use_cv2=enable_cv2)
    image_right, (orig_h_right, orig_w_right) = load_and_preprocess_image(right, width, height, use_cv2=enable_cv2)

    assert orig_h_left == orig_h_right and orig_w_left == orig_w_right

    session = InferenceSession.load_from_model(model)
    flow_up = session.run(input_feed={"x1":image_left, "x2":image_right})["output"]

    # Please use opencv whenever possible
    flow_up = resize_disp(flow_up[0,0], orig_w_left, orig_h_left, use_cv2=enable_cv2)

    flow_up *= orig_w_left/width
    output = np.abs(flow_up)
    
    plt.imsave(f"output-ax.png", output, cmap='jet')

    return output


if __name__ == "__main__":
    args = parse_args()
    infer(**vars(args))
