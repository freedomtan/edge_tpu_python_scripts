First, get pre-trained MobileNet SSD model(s), e.g.,
```
curl https://storage.googleapis.com/cloud-iot-edge-pretrained-models/canned_models/mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite -o /tmp/mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite

curl https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -o /tmp/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip

(cd /tmp; unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip labelmap.txt)

```

Then prepare input file, e.g.,
```
curl https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/image2.jpg > /tmp/image2.jpg
```


Run it

```
python3 object_detection_coral.py --show_image True
```
or
```
python3 object_detection_coral.py --show_image True -c 100
```

