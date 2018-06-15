"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
import json

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('image_dir', '', 'Path to the image directory')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

class_to_index_map = {
    'penny-indian' : 1,
    'penny-wheat' : 2,
    'penny-lincoln' : 3,
    'penny-shield' : 4,
    'nickel-liberty' : 5,
    'nickel-buffalo' : 6,
    'nickel-jefferson' : 7,
    'dime-liberty-seated' : 8,
    'dime-barber' : 9,
    'dime-mercury' : 10,
    'dime-roosevelt' : 11,
    'quarter-washington' : 12,
    'quarter-bicentennial' : 13,
    }

pcgs_category_map = {'Indian Cent' : 1,
                      'Lincoln Cent (Wheat Reverse)' : 2,
                      'Lincoln Cent (Modern)' : 3,
                      'Liberty Nickel' : 5,
                      'Buffalo Nickel' : 6,
                      'Jefferson Nickel' : 7,
                      'Liberty Seated Dime' : 8,
                      'Barber Dime' : 9,
                      'Mercury Dime' : 10,
                      'Roosevelt Dime' : 11,
                      'Washington Quarter' : 12,
                      }

print (FLAGS.image_dir + '/pcgs_number_map.json')

with open(FLAGS.image_dir + '/pcgs_number_map.json') as f:
    pcgs_number_map = json.load(f)
    
print(pcgs_number_map)

class_counts = {}

def class_text_to_tf_index(row_label): 
  return 1
#    if row_label in class_to_index_map:
#        return class_to_index_map[row_label]
#    elif row_label.split('-')[0][4:] in pcgs_number_map:
#        pcgs_record = pcgs_number_map[row_label.split('-')[0][4:]]
#        if pcgs_record['category'] == 'Lincoln Cent (Modern)' and 'Shield Reverse' in pcgs_record['sub_category']:
#            return 4
#        elif pcgs_record['category'] == 'Washington Quarter' and 'Bi-Centennial Reverse' in pcgs_record['sub_category']:
#            return 13
#        else:
#            return pcgs_category_map[pcgs_record['category']]
#    return None
 
  
def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        if row['class'] is not None:
            tf_index = class_text_to_tf_index(row['class'])
#            print(row['class'] + ' ->tf_index: ' + str(tf_index) )
            if tf_index is not None:
                xmins.append(row['xmin'] / width)
                xmaxs.append(row['xmax'] / width)
                ymins.append(row['ymin'] / height)
                ymaxs.append(row['ymax'] / height)
                classes_text.append(row['class'].encode('utf8'))
                classes.append(tf_index)
                if tf_index not in class_counts:
                    class_counts[tf_index] = 1 
                else:
                    class_counts[tf_index] += 1
    if not classes:
        return None

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd(), FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        if tf_example is not None:
            writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))
    print(class_counts)

if __name__ == '__main__':
    tf.app.run()
