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
  Downloads checkpoint and dataset needed for the tutorial.

  --network_type      Can be one of [mobilenet_v1_ssd, mobilenet_v2_ssd],
                      mobilenet_v2_ssd by default.
  --help              Display this help.
END_OF_USAGE
}

network_type="mobilenet_v2_ssd"
skip_tf_record=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --network_type)
      network_type=$2
      shift 2 ;;
    --skip_tf_record)
      skip_tf_record=true
      shift 1 ;;
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

echo "PREPARING checkpoint..."
mkdir -p "${LEARN_DIR}"

ckpt_link="${ckpt_link_map[${network_type}]}"
ckpt_name="${ckpt_name_map[${network_type}]}"
cd "${LEARN_DIR}"

if [ ! -f "${ckpt_name}.tar.gz" ]; then
	wget -O "${ckpt_name}.tar.gz" "$ckpt_link"
else
	echo "- Checkpoint was already downloaded"
fi

tar zxvf "${ckpt_name}.tar.gz"
rm -rf "${CKPT_DIR}/${ckpt_name}"
mv "${ckpt_name}" "${CKPT_DIR}"

echo "CHOSING config file..."
config_filename="${config_filename_map[${network_type}-coco-true]}"
cd "${OBJ_DET_DIR}"
cp "configs/${config_filename}" "${CKPT_DIR}/pipeline.config"

echo "REPLACING variables in config file..."
sed -i "s%CKPT_DIR_TO_CONFIGURE%${CKPT_DIR}%g" "${CKPT_DIR}/pipeline.config"
sed -i "s%DATASET_DIR_TO_CONFIGURE%${DATASET_DIR}%g" "${CKPT_DIR}/pipeline.config"

echo "PREPARING dataset (COCO 2017)"
mkdir -p "${DATASET_DIR}"
cd "${DATASET_DIR}"

TRAIN_IMAGE_DIR=${DATASET_DIR}/train2017
VAL_IMAGE_DIR=${DATASET_DIR}/val2017
TEST_IMAGE_DIR=${DATASET_DIR}/test2017
ANNOTATIONS_DIR=${DATASET_DIR}/annotations
TRAIN_ANNOTATIONS_FILE=${ANNOTATIONS_DIR}/instances_train2017.json
VAL_ANNOTATIONS_FILE=${ANNOTATIONS_DIR}/instances_val2017.json
TESTDEV_ANNOTATIONS_FILE=${ANNOTATIONS_DIR}/image_info_test-dev2017.json

if [ ! -d "${TRAIN_IMAGE_DIR}" ]; then
	echo "> DOWNLOADING train dataset"
	wget http://images.cocodataset.org/zips/train2017.zip
	echo "> EXTRACTING train dataset"
	unzip -q train2017.zip # output: train2017/
else
	echo "- Train dataset was already downloaded"
fi

if [ ! -d "${VAL_IMAGE_DIR}" ]; then
	echo "> DOWNLOADING validation dataset"
	wget http://images.cocodataset.org/zips/val2017.zip
	echo "> EXTRACTING validation dataset"
	unzip -q val2017.zip # output: val2017/
else
	echo "- Validation dataset was already downloaded"
fi

if [ ! -d "${TEST_IMAGE_DIR}" ]; then
	echo "> DOWNLOADING test dataset"
	wget http://images.cocodataset.org/zips/test2017.zip
	echo "> EXTRACTING test dataset"
	unzip -q test2017.zip # output: test2017/
else
	echo "- Test dataset was already downloaded"
fi

if [ ! -d "${ANNOTATIONS_DIR}" ]; then
	echo "> DOWNLOADING dataset annotations"
	wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
	wget http://images.cocodataset.org/annotations/image_info_test2017.zip
	echo "> EXTRACTING dataset annotations"
	unzip -q annotations_trainval2017.zip # output: annotations/
	unzip -q image_info_test2017.zip # output: annotations/
else
	echo "- Dataset annotations was already downloaded"
fi

echo "PREPARING label map..."
cd "${OBJ_DET_DIR}"
cp "object_detection/data/mscoco_label_map.pbtxt" "${DATASET_DIR}"

if [ "${skip_tf_record}" = false ]; then
  echo "CONVERTING dataset to TF Record..."
  cd "${OBJ_DET_DIR}"
  python object_detection/dataset_tools/create_coco_tf_record.py --logtostderr \
        --train_image_dir="${TRAIN_IMAGE_DIR}" \
        --val_image_dir="${VAL_IMAGE_DIR}" \
        --test_image_dir="${TEST_IMAGE_DIR}" \
        --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
        --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
        --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
        --output_dir="${DATASET_DIR}"
else
  echo "- Skipping creation of TF Record"
fi
