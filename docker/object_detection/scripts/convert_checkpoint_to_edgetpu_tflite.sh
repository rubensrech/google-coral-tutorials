#!/bin/bash
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Exit script on error.
set -e
# Echo each command, easier for debugging.
set -x

usage() {
  cat << END_OF_USAGE
  Converts TensorFlow checkpoint to EdgeTPU-compatible TFLite file.

  --checkpoint_num  Checkpoint number, by default 0.
  --model_name      Model name referring path to checkpoint files, by default `ssd_mobilenet_v2_pet`.
  --help            Display this help.
END_OF_USAGE
}

ckpt_number=0
model_name=ssd_mobilenet_v2_subcoco
labels=subcoco14
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint_num)
      ckpt_number=$2
      shift 2 ;;
    --model_name)
      model_name=$2
      shift 2 ;;
    --labels)
      labels=$2
      shift 2 ;;
    --help)
      usage
      exit 0 ;;
    --*)
      echo "Unknown flag $1"
      usage
      exit 1 ;;
  esac
done

source "$PWD/constants.sh"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/${model_name}"
OUTPUT_DIR=${OUTPUT_DIR}/${model_name}

MODEL_DIR=${LEARN_DIR}/${model_name}
CKPT_DIR=${MODEL_DIR}/ckpt
TRAIN_DIR=${MODEL_DIR}/train

echo "GENERATING label file..."

if [ "${labels}" = pet ]; then
  echo "0 Abyssinian" >> "${OUTPUT_DIR}/labels.txt"
  echo "1 american_bulldog" >> "${OUTPUT_DIR}/labels.txt"
elif [ "${labels}" = subcoco14 ]; then
  echo "0 person" >> "${OUTPUT_DIR}/labels.txt"
  echo "1 bicycle" >> "${OUTPUT_DIR}/labels.txt"
  echo "2 car" >> "${OUTPUT_DIR}/labels.txt"
  echo "3 motorcycle" >> "${OUTPUT_DIR}/labels.txt"
  echo "4 airplane" >> "${OUTPUT_DIR}/labels.txt"
  echo "5 bus" >> "${OUTPUT_DIR}/labels.txt"
  echo "6 train" >> "${OUTPUT_DIR}/labels.txt"
  echo "7 truck" >> "${OUTPUT_DIR}/labels.txt"
  echo "8 boat" >> "${OUTPUT_DIR}/labels.txt"
  echo "9 traffic light" >> "${OUTPUT_DIR}/labels.txt"
  echo "10 fire hydrant" >> "${OUTPUT_DIR}/labels.txt"
  echo "12 stop sign" >> "${OUTPUT_DIR}/labels.txt"
  echo "13 parking meter" >> "${OUTPUT_DIR}/labels.txt"
  echo "14 bench" >> "${OUTPUT_DIR}/labels.txt"
fi

echo "EXPORTING frozen graph from checkpoint..."
python object_detection/export_tflite_ssd_graph.py \
  --pipeline_config_path="${CKPT_DIR}/pipeline.config" \
  --trained_checkpoint_prefix="${TRAIN_DIR}/model.ckpt-${ckpt_number}" \
  --output_directory="${OUTPUT_DIR}" \
  --add_postprocessing_op=true

echo "CONVERTING frozen graph to TF Lite file..."
tflite_convert \
  --output_file="${OUTPUT_DIR}/output_tflite_graph.tflite" \
  --graph_def_file="${OUTPUT_DIR}/tflite_graph.pb" \
  --inference_type=QUANTIZED_UINT8 \
  --input_arrays="${INPUT_TENSORS}" \
  --output_arrays="${OUTPUT_TENSORS}" \
  --mean_values=128 \
  --std_dev_values=128 \
  --input_shapes=1,300,300,3 \
  --change_concat_input_ranges=false \
  --allow_nudging_weights_to_use_fast_gemm_kernel=true \
  --allow_custom_ops

echo "TFLite graph generated at ${OUTPUT_DIR}/output_tflite_graph.tflite"
