"""label_image for coral"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import time

from edgetpu.basic import edgetpu_utils
from edgetpu.classification.engine import ClassificationEngine
from PIL import Image

def load_labels(filename):
  my_labels = []
  input_file = open(filename, 'r')
  for l in input_file:
    my_labels.append(l.strip())
  return my_labels

if __name__ == "__main__":
  floating_model = False

  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--image", default="/tmp/grace_hopper.bmp", \
    help="image to be classified")
  parser.add_argument("-m", "--model_file", \
    default="/tmp/mobilenet_v1_1.0_224_quant_edgetpu.tflite", \
    help=".tflite model to be executed")
  parser.add_argument("-l", "--label_file", default="/tmp/labels.txt", \
    help="name of file containing labels")
  parser.add_argument("-k", "--top_k", default=5, help="top_k")
  parser.add_argument("-t", "--threshold", default=0.0, help="threshold")
  parser.add_argument("-c", "--loop_counts", default=1, help="loop counts")
  parser.add_argument("-d", "--device_path", help="device_path")
  parser.add_argument("-b", "--input_mean", default=127.5, help="input_mean")
  parser.add_argument("-s", "--input_std", default=127.5, help="input standard deviation")
  args = parser.parse_args()

  if args.device_path:
    engine = ClassificationEngine(args.model_file, device_path=args.device_path)
  else:
    engine = ClassificationEngine(args.model_file)
  print("driver version:", edgetpu_utils.GetRuntimeVersion())
  print("available tpus:",
    edgetpu_utils.ListEdgeTpuPaths(edgetpu_utils.EDGE_TPU_STATE_NONE))
  print("device path:", engine.device_path())

  input_tensor_shape = engine.get_input_tensor_shape()
  if (input_tensor_shape.size != 4 or input_tensor_shape[3] != 3 or
                  input_tensor_shape[0] != 1):
    raise RuntimeError('Invalid input tensor shape! Expected: [1, height, width, 3]')
  _, height, width, _ = input_tensor_shape

  img = Image.open(args.image)
  img = img.resize((width, height))

  input_tensor = np.asarray(img).flatten()

  if floating_model:
    input_tensor = (np.float32(input_tensor) - args.input_mean) / args.input_std

  loop_counts = int(args.loop_counts)
  if (loop_counts > 1):
    for a in range(5):
      engine.RunInference(input_tensor)

  start_time = time.time()
  for a in range(loop_counts):
    _, raw_results = engine.RunInference(input_tensor)
  stop_time = time.time()
  print("time: ", (stop_time - start_time) * 1000 / loop_counts)

  labels = load_labels(args.label_file)
  top_k = min(args.top_k, len(raw_results))
  result = []
  indices = np.argpartition(raw_results, -top_k)[-top_k:]
  for i in indices:
    if raw_results[i] > args.threshold:
      result.append((i, raw_results[i]))
  result.sort(key=lambda tup: -tup[1])
  for r in result[:top_k]:
    if floating_model:
      print('{0:08.6f}'.format(r[1])+":", labels[r[0]])
    else:
      print('{0:08.6f}'.format(r[1])+":", labels[r[0]])
