import sys
sys.path.append('core')
import os
import argparse
import glob
import numpy as np
import torch
from torch import nn
from raft_stereo import RAFTStereo
import onnx
from onnx.shape_inference import infer_shapes
import onnxsim

def export(args):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))
    
    device = torch.device("cpu")
    model = model.module

    if args.corr_radius == 1:
        print("update_block encoder convc1", model.update_block.encoder.convc1.weight.data.shape)
        print("update_block encoder convc1", model.update_block.encoder.convc1.bias.data.shape)
        new_convc1 = nn.Conv2d(12, 64, 1, padding=0).to(device)
        new_convc1.weight.data[:, 0:3] =  model.update_block.encoder.convc1.weight.data[:, 3:6]
        new_convc1.weight.data[:, 3:6] =  model.update_block.encoder.convc1.weight.data[:, 3+9:6+9]
        new_convc1.weight.data[:, 6:9] =  model.update_block.encoder.convc1.weight.data[:, 3+18:6+18]
        new_convc1.weight.data[:, 9:12] =  model.update_block.encoder.convc1.weight.data[:, 3+27:6+27]

        new_convc1.bias.data = model.update_block.encoder.convc1.bias.data
        
        model.update_block.encoder.convc1 = new_convc1

    model.to(device)
    model.eval()
    model.forward = model.forward_export

    output_directory = args.output_directory
    os.makedirs(output_directory, exist_ok=True)

    
    height = args.height
    width = args.width

    x1 = torch.rand((1,3,height,width)).to(device)
    x2 = torch.rand((1,3,height,width)).to(device)

    input = (x1,x2)
    input_names=["x1","x2"]

    onnx_path = f"{output_directory}/raft_steoro{height}x{width}_r{args.corr_radius}.onnx"
    torch.onnx.export(model, input, onnx_path, input_names=input_names, output_names=["output"], opset_version=16)
    onnx_model = onnx.load(onnx_path)
    onnx_model = infer_shapes(onnx_model)
    # convert model
    model_simp, check = onnxsim.simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_path)
    print("onnx simpilfy successed, and model saved in {}".format(onnx_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "alt_fast", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--width', type=int, required=True, help="image width input to model")
    parser.add_argument('--height', type=int, required=True, help="image height input to model")

    args = parser.parse_args()
    export(args)        