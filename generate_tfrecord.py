#based on https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py
"""
Usage:
  # Create data:
  python generate_tfrecord.py --csv_input=foid_labels_v100.csv --image_dir=images --test_output_path=test.record  --train_output_path=train.record


"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('image_dir', '', 'Path to the image directory')
flags.DEFINE_string('test_output_path', '', 'Path to Test output TFRecord')
flags.DEFINE_string('train_output_path', '', 'Path to Train output TFRecord')
FLAGS = flags.FLAGS

def is_fish(row_label):
    return row_label not in {'Human', 'No fish', 'Unknown'}


def split(df, attr):
    data = namedtuple('data', ['img_id', 'boxes'])
    gb = df.groupby(attr)
    return [data('{}.jpg'.format(img_id), boxes) for img_id, boxes in gb]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, group.img_id), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    img_id = group.img_id.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for _, row in group.boxes.iterrows():
        if is_fish(row['label_l1']):
            xmins.append(row['x_min'] / width)
            xmaxs.append(row['x_max'] / width)
            ymins.append(row['y_min'] / height)
            ymaxs.append(row['y_max'] / height)
            classes_text.append('fish'.encode('utf8'))
            classes.append(1)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(img_id),
        'image/source_id': dataset_util.bytes_feature(img_id),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example, group.boxes.test.iloc[0]


def main(_):
    writer_test = tf.python_io.TFRecordWriter(FLAGS.test_output_path)
    writer_train = tf.python_io.TFRecordWriter(FLAGS.train_output_path)    
    path = FLAGS.image_dir
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'img_id')
    for group in tqdm(grouped):
        tf_example, is_test = create_tf_example(group, path)
        if is_test:
            writer_test.write(tf_example.SerializeToString())
        else:
            writer_train.write(tf_example.SerializeToString())

    writer_test.close()
    writer_train.close()
    print('Successfully created the TFRecords')


if __name__ == '__main__':
    tf.app.run()
