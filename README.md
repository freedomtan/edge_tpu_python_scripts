# edge_tpu_python_scripts
some scripts I used to test Google's Edge TPU

I got a [Coral USB accelerator](https://coral.withgoogle.com/products/accelerator/) as a gift from TensorFlow Dev Summit 2019. And bought a [Coral Dev Board](https://coral.withgoogle.com/products/dev-board/) from Mouser later. Both of them look good. But as an engineer, I like to know the performance of the Edge TPU. There is a table in the [Edge TPU FAQ](https://coral.withgoogle.com/tutorials/edgetpu-faq/),

|Model architecture | Desktop CPU |	Desktop CPU   |USB Accelerator (USB 3.0)||
|                   |             | with Edge TPU	| Embedded CPU            |Dev Board with Edge TPU |
|-------------------|-----------------------------|------------------------|-------------|
|MobileNet v1	|47 ms	| 2.2 ms	| 179 ms	|2.2 ms|
|MobileNet v2	|45 ms	| 2.3 ms	| 150 ms	|2.5 ms|
|Inception v1	|92 ms	|3.6 ms	| 406 ms	| 3.9 ms|
|Inception v4	|792 ms	|100 ms	|3,463 ms	| 100 ms|
