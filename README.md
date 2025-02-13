# RAFT-Stereo.axera
RAFT-Stereo DEMO on Axera

- 目前支持 Python 语言，C++代码在开发中。  
- 预编译模型下载[models](https://github.com/AXERA-TECH/RAFT-Stereo.axera/releases/download/v1.0.0/models.tar.gz)，如需自行转换请参考[模型转换](/model_convert/README.md)。  

## 支持平台

- [x] AX650N
- [ ] AX630C

## 模型转换

[模型转换](./model_convert/README.md)

## 上板部署

- AX650N 的设备已预装 Ubuntu22.04
- 以 root 权限登陆 AX650N 的板卡设备
- 链接互联网，确保 AX650N 的设备能正常执行 `apt install`, `pip install` 等指令
- 已验证设备：AX650N DEMO Board、爱芯派Pro

### Python API 运行

#### Requirements

```
mkdir /opt/site-packages
cd python
pip3 install -r requirements.txt --prefix=/opt/site-packages
``` 

#### 添加环境变量

将以下两行添加到 `/root/.bashrc`(实际添加的路径需要自行检查)后，重新连接终端或者执行 `source ~/.bashrc`

```
export PYTHONPATH=$PYTHONPATH:/opt/site-packages/local/lib/python3.10/dist-packages  
export PATH=$PATH:/opt/site-packages/local/bin
``` 

#### 运行

##### 基于 ONNX Runtime 运行  
可在开发板或PC运行 

在开发板或PC上，运行以下命令  
```
cd python
python3 infer_onnx.py --left examples/left/000051_11.png --right examples/right/000051_11.png --model ../models/raft_steoro384x1280_r4.onnx
```
输出结果：  
![output](asserts/output-onnx-r4.png)
或

```
cd python
python3 infer_onnx.py --left examples/left/000051_11.png --right examples/right/000051_11.png --model ../models/raft_steoro256x640_r1.onnx

```

输出结果:  
![output](asserts/output-onnx-r1.png)

运行参数说明:  
| 参数名称 | 说明  |
| --- | --- | 
| --left | 输入左目图片路径 |  
| --right | 输入左目图片路径 |  
| --model | onnx模型路径 | 

##### 基于AXEngine运行  
在开发板上运行命令

```
cd python  
python3 infer.py --left examples/left/000051_11.png --right examples/right/000051_11.png --model ../models/raft_steoro384x1280_r4.axmodel  --width 1280 --height 384
```  
输出结果:  
![output](asserts/output-ax-r4.png)

或

```
cd python  
python3 infer.py --left examples/left/000051_11.png --right examples/right/000051_11.png --model ../raft_steoro256x640_r1.axmodel  --width 640 --height 256
```  
输出结果:  
![output](asserts/output-ax-r1.png)

可以看出来这个模型对量化还是比较敏感的。  

运行参数说明:  
| 参数名称 | 说明  |
| --- | --- | 
| --left | 输入左目图片路径 |  
| --right | 输入左目图片路径 |  
| --model | axmodel模型路径 | 
| --width | 输入模型的图片宽度，注意不是图片原始宽度 |  
| --height| 输入模型的图片高度，注意不是图片原始宽度 |
### Latency

#### AX650N

| model | latency(ms) |
|---|---|
|raft_steoro256x640_r1.axmodel | 20.9|
|raft_steoro384x1280_r4.axmodel | 111.8|


#### AX630C

(TODO)

## 技术讨论

- Github issues
- QQ 群: 139953715
