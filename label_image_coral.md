
With model, input image (grace_hopper.bmp), and labels file (labels.txt)
in /tmp.

The example input image and labels file are from TensorFlow repo and
Edge TPU's quantized MobileNet V1 model files.

```
curl https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp > /tmp/grace_hopper.bmp

curl  https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz  | tar xzv -C /tmp  mobilenet_v1_1.0_224/labels.txt
mv /tmp/mobilenet_v1_1.0_224/labels.txt /tmp/

curl https://storage.googleapis.com/cloud-iot-edge-pretrained-models/canned_models/mobilenet_v1_1.0_224_quant_edgetpu.tflite > /tmp/mobilenet_v1_1.0_224_quant_edgetpu.tflite

```

Run

```
$ python3 label_image_coral.py  -c 100

```

We can get results like

```
time:  13.17286491394043
0.671875: 653:military uniform
0.128906: 907:Windsor tie
0.039062: 458:bow tie, bow-tie, bowtie
0.027344: 668:mortarboard
0.019531: 466:bulletproof vest
```

Run

```
$ python3 label_image_coral.py  -c 100
```

We can get results like
```
time:  2.9425740242004395
0.671875: 653:military uniform
0.128906: 907:Windsor tie
0.039062: 458:bow tie, bow-tie, bowtie
0.027344: 668:mortarboard
0.019531: 466:bulletproof vest

```
