## 一、系统环境

硬件环境：Intel(R) Core(TM) i7-8650U CPU @ 1.90GHz   2.11 GHz, RAM：16.0 GB

操作系统：ubuntu-16.04.4

## 二、安装docker
```
sudo apt-get update
sudo apt install docker.io

docker的常用命令：
# 列出本机所有容器，包括已经终止运行的
docker ps -a
#容器和宿主机之间的文件复制
docker cp [DOCKER ID]:[path] [path]
docker cp [path] [DOCKER ID]:[path]

```
## 三、使用矽速科技提供的docker环境
使用矽速科技提供的docker环境进行开发：

>  注：请保证至少有20GB的空闲磁盘空间

```
# 方法一，从docker hub下载，需要梯子
sudo docker pull zepan/zhouyi
# 方法二，百度云下载镜像文件（压缩包约2.9GB，解压后约5.3GB）
# 链接：https://pan.baidu.com/s/1yaKBPDxR_oakdTnqgyn5fg 
# 提取码：f8dr 
gunzip zhouyi_docker.tar.gz
sudo docker load --input zhouyi_docker.tar
```

下载好docker后即可运行其中的例程测试环境是否正常：
```
sudo docker run -i -t zepan/zhouyi /bin/bash
cd ~/demos/tflite
./run_sim.sh
python3 quant_predict.py
```
## 四、下载NASNet模型

下载地址：[NASNet-A_Mobile_224](https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_mobile_04_10_2017.tar.gz)

```
解压模型：
tar xvf nasnet-a_mobile_04_10_2017.tar.gz

将解压好的模型拷贝至docker文件系统中：(注：aa5b422f0f6f为docker container id)
sudo docker cp ./model.ckpt.* aa5b422f0f6f:/tf
```

## 五、导出图



```
git clone https://github.com/tensorflow/models.git
cd models/research/slim
python setup.py build
python setup.py install
python3 export_inference_graph.py \
--alsologtostderr \
--model_name=nasnet_mobile \
--image_size=224 \
--output_file=/tf/nasnet_mobile.pb
```
## 六、冻结图
```
git clone -b r1.15 --single-branch https://github.com/tensorflow/tensorflow.git
cd tensorflow/tensorflow/python/tools
python3 freeze_graph.py \
--input_graph=/tf/nasnet_mobile.pb \
--input_checkpoint=/tf/model.ckpt \
--input_binary=true \
--output_graph=/tf/nasnet_mobile_frozen.pb \
--output_node_names=final_layer/predictions
```
## 七、准备量化矫正数据集

这里下载1000类的1000张图片

[imagenet-sample-images](https://github.com/nihui/imagenet-sample-images)

生成每行都是"图片路径 label"的imagelabel.txt

```
ls imagenet-sample-images | sort > image.txt
seq 0 999 > label.txt
paste -d ' ' image.txt label.txt > imagelabel.txt
```

将上述产生的文件夹imagenet-sample-images和文件imagelabel.txt拷贝到dataset目录下。

把 demos/pb/dataset/ 里的 preprocess_dataset.py 拿出来改改开头


```
img_dir='./imagenet-sample-images/'
label_file='./imagelabel.txt'

#MNASNET PARAM
input_height=224
input_width=224
input_channel = 3
mean = 127.5
var = 127.5
```
生成 dataset.npy 和 label.npy 文件
```
cd dataset
python3 preprocess_dataset.py
```
# 八、准备 input.bin 和 output_ref.bin

把 demos/pb/config/ 里的 gen_inputbin.py 拿出来改改

这里有个大坑，mean 和 val 是不能照着模型的预处理写的，必须要写 `mean=127.5 var=1` 才可以，似乎npu总是输入 -127~127 范围的数值

```
input_height=224
input_width=224
input_channel = 3
mean = [127.5, 127.5, 127.5]
var = 1
```

测试图片直接使用的 dataset/img/ILSVRC2012_val_00000003.JPEG，并将其复制至目录model下，改命为1.jpg，于是 output_ref.bin 就复用了，执行以下命令，生成 mnasnet 所需的 input.bin

```
python gen_inputbin.py
```
## 九、编辑NN compiler配置文件

得到pb和校准数据集后，我们就可以编辑NN编译器的配置文件来生成AIPU的可执行文件。

把 demos/pb/config/ 里的 resnet_50_run.cfg 拿出来改成 nasnet_mobile_run.cfg，并修改成以下内容。

```
[Common]
mode=run

[Parser]
model_name = nasnet_mobile
detection_postprocess = 
model_domain = image_classification
output = final_layer/FC/BiasAdd
input_model = ./model/nasnet_mobile_frozen.pb
input = input
input_shape = [1,224,224,3]
output_dir = ./

[AutoQuantizationTool]
model_name = nasnet_mobile
quantize_method = SYMMETRIC
ops_per_channel = DepthwiseConv
calibration_data = ./dataset/dataset.npy
calibration_label = ./dataset/label.npy
preprocess_mode = normalize
quant_precision=int8
reverse_rgb = False
label_id_offset = 0

[GBuilder]
inputs=./model/input.bin
simulator=aipu_simulator_z1
outputs=output_nasnet_mobile.bin
profile= True
target=Z1_0701
```

## 十、仿真AIPU执行结果

编辑完cfg文件后，即可执行获得运行结果

```
aipubuild config/nasnet_mobile_run.cfg
```

执行后得到运算结果：output_nasnet_mobile.bin
以下为执行log:

```
[I] ==== auto-quantization ======
[I]     step1: get max/min statistic value DONE
[W] shift value is discrete in Depthwise, layer cell_stem_1/comb_iter_4/left/separable_3x3_1/separable_conv2d/depthwise_0, fixed by constraining shift value, may lead to acc drop
[W] shift value is discrete in Depthwise, layer cell_stem_1/comb_iter_4/left/separable_3x3_2/separable_conv2d/depthwise_0, fixed by constraining shift value, may lead to acc drop
[I]     step2: quantization each op DONE
[I]     step3: build quantization forward DONE
[I]     step4: show output scale of end node:
[I]             layer_id:643, layer_top:final_layer/FC/BiasAdd_0, output_scale:[13.73145]
[I] ==== auto-quantization DONE =
[I] Quantize model complete
[I] Building ...
[I] [common_options.h: 276] BuildTool version: 4.0.175. Build for target Z1_0701 at frequency 800MHz
[I] [common_options.h: 297] using default profile events to profile AIFF
[I] [IRChecker] Start to check IR: /tmp/AIPUBuilder_1626750712.9855294/nasnet_mobile_int8.txt
[I] [IRChecker] model_name: nasnet_mobile
[I] [IRChecker] IRChecker: All IR pass
[I] [graph.cpp : 846] loading graph weight: /tmp/AIPUBuilder_1626750712.9855294/nasnet_mobile_int8.bin size: 0x533efe
[I] [builder.cpp:1059] Total memory for this graph: 0x12cf170 Bytes
[I] [builder.cpp:1060] Text   section:  0x0018c7f0 Bytes
[I] [builder.cpp:1061] RO     section:  0x00027500 Bytes
[I] [builder.cpp:1062] Desc   section:  0x00073500 Bytes
[I] [builder.cpp:1063] Data   section:  0x00945580 Bytes
[I] [builder.cpp:1064] BSS    section:  0x00722600 Bytes
[I] [builder.cpp:1065] Stack         :  0x00040400 Bytes
[I] [builder.cpp:1066] Workspace(BSS):  0x00021200 Bytes
[I] [main.cpp  : 467] # autogenrated by aipurun, do NOT modify!
[I] [main.cpp  : 118] run simulator:
aipu_simulator_z1 /tmp/temp_37bcb54b645f8ee65c7857bcf1e86.cfg
[INFO]:SIMULATOR START!
[INFO]:========================================================================
[INFO]:                             STATIC CHECK
[INFO]:========================================================================
[INFO]:  INST START ADDR : 0x0(0)
[INFO]:  INST END ADDR   : 0x18c7ef(1624047)
[INFO]:  INST SIZE       : 0x18c7f0(1624048)
[INFO]:  PACKET CNT      : 0x18c7f(101503)
[INFO]:  INST CNT        : 0x631fc(406012)
[INFO]:------------------------------------------------------------------------
...
[INFO]:========================================================================
[INFO]:                             STATIC CHECK END
[INFO]:========================================================================
[INFO]:AIPU START RUNNING: BIN[0]
[INFO]:TOTAL TIME: 34.221394s. 
[INFO]:SIMULATOR EXIT!
[I] [main.cpp  : 135] Simulator finished.
```

这里的demo是1000分类，所以 output_resnet_50.bin 是1000字节的int8结果，除以这个 output_scale 就是实际的float输出结果。
这里简单使用int8格式进行解析，得到最大概率对应的类别，可以看到和实际图片类别一致

```
outputfile = './output_resnet_50.bin'
npyoutput = np.fromfile(outputfile, dtype=np.int8)
outputclass = npyoutput.argmax()
print("Predict Class is %d"%outputclass)
```

## 十一、仿真结果比对

把 demos/pb/config/ 里的 quant_predict.py 拿出来改改

主要是以两行：

```
outputfile = current_dir + '/output_nasnet_mobile.bin'
npyoutput = np.fromfile(outputfile, dtype=np.int8)
```

运行 quant_predict.py

```
python quant_predict.py
```
以下为比对结果：
```
predict first 5 label:
    index  231, prob 127, name: Shetland sheepdog, Shetland sheep dog, Shetland
    index  232, prob 121, name: collie
    index  158, prob  46, name: papillon
    index  170, prob  41, name: borzoi, Russian wolfhound
    index   96, prob  35, name: jacamar
true first 5 label:
    index  230, prob 109, name: Old English sheepdog, bobtail
    index  231, prob  96, name: Shetland sheepdog, Shetland sheep dog, Shetland
    index  232, prob  57, name: collie
    index  226, prob  54, name: malinois
    index  263, prob  53, name: Brabancon griffon
Detect picture save to result.jpeg
```

相关文件详见：
[r329_aipu](https://github.com/bolide2006/r329_aipu.git)

