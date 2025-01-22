# 模型转换

## 创建虚拟环境

```
conda create -n raft-stereo python=3.12 -y
conda activate raft-stereo
```

## 安装依赖

```
pip install -r requirements.txt
```

## 导出模型（PyTorch -> ONNX）
本示例基于官方 checkpoint raftstereo-realtime.pth 导出两个版本的模型，一个 `radius`参数值为`1`，一个为`4`。  

### 当 `radius==4`
这个版本以[RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) 官方的 [Faster Implementation](https://github.com/princeton-vl/RAFT-Stereo/tree/main?tab=readme-ov-file#optional-faster-implementation) 为基础，将 `corr_implementation` 参数值 从`reg_cuda`修改为 `alt`  
```
python export_onnx.py --restore_ckpt ../models/raftstereo-realtime.pth \
                --mixed_precision \
                --shared_backbone \
                --n_downsample 3 \
                --n_gru_layers 2 \
                --slow_fast_gru \
                --valid_iters 7 \
                --corr_radius 4 \
                --corr_implementation alt \
                --output_directory ../models \
                --width 1280 \
                --height 384 
```
导出成功会生成文件 `../models/raft_steoro384x1280_r4.onnx`.

### 当 `radius==1`
这个版本以前面的版本为基础，将 `corr_radius` 设置为`1`，损失一些精度，提升速度。
```
python export_onnx.py --restore_ckpt ../models/raftstereo-realtime.pth \
                --mixed_precision \
                --shared_backbone \
                --n_downsample 3 \
                --n_gru_layers 2 \
                --slow_fast_gru \
                --valid_iters 7 \
                --corr_radius 1 \
                --corr_implementation alt \
                --output_directory ../models \
                --width 640 \
                --height 256 
```
导出成功会生成文件 `../models/raft_steoro256x640_r1.onnx`.
  

## 转换模型（ONNX -> Axera）

使用模型转换工具 `Pulsar2` 将 ONNX 模型转换成适用于 Axera 的 NPU 运行的模型文件格式 `.axmodel`，通常情况下需要经过以下两个步骤：

- 生成适用于该模型的 PTQ 量化校准数据集
- 使用 `Pulsar2 build` 命令集进行模型转换（PTQ 量化、编译），更详细的使用说明请参考 [AXera Pulsar2 工具链指导手册](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)

### 下载量化数据集
```
bash download_dataset.sh
```
这个模型的输入是左右目两张图片，比较简单，这里我们直接下载打包好的图片数据  

### 模型转换

#### 修改配置文件
 
检查`config.json` 中 `calibration_dataset` 字段，将该字段配置的路径改为上一步下载的量化数据集存放路径  

#### Pulsar2 build

参考命令如下：


```
pulsar2 build --input ../models/raft_steoro384x1280_r4.onnx --config config_r1.json --output_dir build-output-r4 --output_name raft_steoro384x1280_r4.axmodel --target_hardware AX650 --compiler.check 0
```
或

```
pulsar2 build --input ../models/raft_steoro256x640_r1.onnx --config config_r1.json --output_dir build-output-r1 --output_name raft_steoro256x640_r1.axmodel --target_hardware AX650 --compiler.check 0
```