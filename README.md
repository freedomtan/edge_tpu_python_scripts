# edge_tpu_python_scripts
some scripts I used to test Google's Edge TPU

I got a [Coral USB accelerator](https://coral.withgoogle.com/products/accelerator/) as a gift from TensorFlow Dev Summit 2019. And bought a [Coral Dev Board](https://coral.withgoogle.com/products/dev-board/) from Mouser later. Both of them look good. But as an engineer, I like to know the performance of the Edge TPU. There is a table in the [Edge TPU FAQ](https://coral.withgoogle.com/tutorials/edgetpu-faq/),

|Model architecture | Desktop CPU |	Desktop CPU + USB Accelerator (USB 3.0) with Edge TPU	| Embedded CPU | Dev Board with Edge TPU |
|:------------------|------------:| -----------------------------------------------------:|-------------:|------------------------:|
|MobileNet v1	|47 ms	| 2.2 ms	| 179 ms	|2.2 ms|
|MobileNet v2	|45 ms	| 2.3 ms	| 150 ms	|2.5 ms|
|Inception v1	|92 ms	| 3.6 ms	| 406 ms	|3.9 ms|
|Inception v4	|792 ms	| 100 ms	|3,463 ms	|100 ms|


But I found no way to reproduce them, so I wrote these two scripts. With `python3 label_image_coral.py -c 50 -m classification_model`, I can get

|Model architecture | 	iMac 2015 CPU + USB Accelerator (USB 3.0) with Edge TPU	| Macbook Pro 13-inch 2018 CPU + USB Accelerator (USB 3.0) with Edge TPU	| Dev Board with Edge TPU |
|:------------------|------------------------:|------------------------:|------------------------:|
|MobileNet v1	| 2.91 ms| 3.10 ms|2.51 ms|
|MobileNet v2	| 3.01 ms| 3.20 ms|2.69 ms|
|Inception v1	| 3.74 ms| 4.10 ms|4.23 ms|
|Inception v4	| 84.93 ms| 92.32 ms| 101.87 ms|

With Edge TPU C++ API, I can get

|Model architecture | iMac 2015 CPU + USB Accelerator (USB 3.0) with Edge TPU	| Macbook Pro 13-inch 2018 CPU + USB Accelerator (USB 3.0) with Edge TPU	| Dev Board with Edge TPU | Dev Board Edge TPU C++ API | Macbook Pro 13-inch 2018 CPU + USB Accelerator (USB 3.0) with Edge TPU C++ API|
|:------------|-------:|-------:|------:|------:|------:|
|MobileNet v1	| 2.91 ms| 3.10 ms|2.51 ms|2.24 ms|2.73 ms|
|MobileNet v2	| 3.01 ms| 3.20 ms|2.69 ms|2.45 ms|2.89 ms|
|Inception v1	| 3.74 ms| 4.10 ms|4.23 ms|3.77 ms|3.52 ms|
|Inception v4	| 84.93 ms| 92.32 ms| 101.87 ms| 101.63 ms|84.92 ms|



The [label_image using Edge TPU C++ API](https://github.com/freedomtan/edgetpu-native/)
