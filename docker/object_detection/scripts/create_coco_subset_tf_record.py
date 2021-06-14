import os
import contextlib2

from pycocotools.coco import COCO

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util

import tensorflow.compat.v1 as tf

flags = tf.app.flags
flags.DEFINE_string('data_dir', 'learn/coco', 'Root directory to raw Microsoft COCO dataset.')
flags.DEFINE_string('output_dir', 'learn/coco/subset', 'Path to output TFRecord')
flags.DEFINE_list('sup_cats', 'person,vehicle,outdoor', 'Super categories to be included')
flags.DEFINE_integer('max_examples_per_cat', 5000, 'Max number of examples')
FLAGS = flags.FLAGS

def load_coco_dection_dataset(imgsDir, annFile, supCats):
    """Load data from dataset by pycocotools."
    Args:
        imgs_dir: directories of COCO images (train2017 | val2017)
        ann_file: file path of COCO annotations file (instances_train2017.json | instances_val2017.json)
    Return:
        coco_data: list of dictionary format information of each image
    """

    coco = COCO(annFile)

    catIds = coco.getCatIds(supNms=supCats)

    # Because `img_ids = coco.getImgIds(catIds=catIds)` won't work ...
    imgIds = []
    for c in catIds:
        cImgs = coco.getImgIds(catIds=c)
        cImgs = cImgs[0:FLAGS.max_examples_per_cat] if len(cImgs) > FLAGS.max_examples_per_cat else cImgs
        imgIds.extend(cImgs)

    imgIds = set(imgIds) # Remove duplicates

    cocoData = []
    nimgs = len(imgIds)

    for index, imgId in enumerate(imgIds):
        if index % 100 == 0:
            print("Reading images: %d / %d " % (index, nimgs))

        imgInfo = {}
        bboxes = []
        labels = []

        imgDetail = coco.loadImgs(imgId)[0]
        imgHeight = imgDetail['height']
        imgWidth = imgDetail['width']

        annIds = coco.getAnnIds(imgIds=imgId, catIds=catIds)
        anns = coco.loadAnns(annIds)

        for ann in anns:
            catId = ann['category_id']
            if not catId in catIds:
                continue

            bboxes_data = [
                ann['bbox'][0]/float(imgWidth),
                ann['bbox'][1]/float(imgHeight),
                ann['bbox'][2]/float(imgWidth),
                ann['bbox'][3]/float(imgHeight)
            ]

            bboxes.append(bboxes_data)
            labels.append(catId)

        imgPath = os.path.join(imgsDir, imgDetail['file_name'])
        imgBytes = tf.gfile.FastGFile(imgPath, 'rb').read()

        imgInfo['pixel_data'] = imgBytes
        imgInfo['height'] = imgHeight
        imgInfo['width'] = imgWidth
        imgInfo['bboxes'] = bboxes
        imgInfo['labels'] = labels

        cocoData.append(imgInfo)
    
    return cocoData

def dict_to_coco_example(imgData):
    """Convert python dictionary of one image to tf.Example proto.
    Args:
        img_data: infomation of one image
    Returns:
        example: The converted tf.Example
    """

    bboxes = imgData['bboxes']
    xmin, xmax, ymin, ymax = [], [], [], []

    for bbox in bboxes:
        xmin.append(bbox[0])
        xmax.append(bbox[0] + bbox[2])
        ymin.append(bbox[1])
        ymax.append(bbox[1] + bbox[3])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(imgData['height']),
        'image/width': dataset_util.int64_feature(imgData['width']),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/label': dataset_util.int64_list_feature(imgData['labels']),
        'image/encoded': dataset_util.bytes_feature(imgData['pixel_data']),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf-8')),
    }))
    return example


def create_tf_record_from_coco_annotations(imgsDir, annFile, supCats, outputDir, numShards):

    with contextlib2.ExitStack() as tfRecordCloseStack, tf.gfile.GFile(annFile, 'r') as fid:
        outputTfRecords = tf_record_creation_util.open_sharded_output_tfrecords(tfRecordCloseStack, outputDir, numShards)

        cocoData = load_coco_dection_dataset(imgsDir, annFile, supCats)
        totalImgs = len(cocoData)

        for idx, imgData in enumerate(cocoData):
            if idx % 100 == 0:
                print("Converting images: %d / %d" % (idx, totalImgs))

            tfExample = dict_to_coco_example(imgData)
            shardIdx = idx % numShards

            if tfExample:
                outputTfRecords[shardIdx].write(tfExample.SerializeToString())

def main(_):
    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    train_output_path = os.path.join(FLAGS.output_dir, 'coco_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'coco_val.record')

    print(">>>> Converting COCO train dataset to TF record <<<<")
    train_img_dir = os.path.join(FLAGS.data_dir, 'train2017')
    train_ann_file = os.path.join(FLAGS.data_dir, 'annotations', 'instances_train2017.json')
    create_tf_record_from_coco_annotations(train_img_dir, train_ann_file, FLAGS.sup_cats, train_output_path, 100)

    print(">>>> Converting COCO validation dataset to TF record <<<<")
    val_img_dir = os.path.join(FLAGS.data_dir, 'val2017')
    val_ann_file = os.path.join(FLAGS.data_dir, 'annotations', 'instances_val2017.json')
    create_tf_record_from_coco_annotations(val_img_dir, val_ann_file, FLAGS.sup_cats, val_output_path, 50)


if __name__ == "__main__":
    tf.app.run()