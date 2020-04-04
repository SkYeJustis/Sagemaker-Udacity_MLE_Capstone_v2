from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import json
import argparse
from tensorflow.python.platform import tf_logging
import logging as _logging
import sys as _sys
from tensorflow.python.estimator.model_fn import ModeKeys as Modes

tf.logging.set_verbosity(tf.logging.INFO)

INPUT_TENSOR_NAME = 'inputs'
SIGNATURE_NAME = 'predictions'
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3
TRAIN_RECORDS = 'chest_xray_images_train.tfrecords'
EVAL_RECORDS = 'chest_xray_images_test.tfrecords'

def _model_fn(features, labels, mode, params):
    # Input Layer
    input_layer = tf.reshape(features[INPUT_TENSOR_NAME], [-1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS])
    #print("input_layer ", tf.shape(input_layer))

    # Regular NN layer - https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
    dense1 = tf.layers.dense(inputs=input_layer, units=512, activation=tf.nn.relu)
    
    dropout1 = tf.layers.dropout(inputs=dense1,rate=0.2,training=(mode == Modes.TRAIN))
    
    dense2 = tf.layers.dense(inputs=dropout1, units=512, activation=tf.nn.relu)
    
    dropout2 = tf.layers.dropout(inputs=dense2, rate=0.2, training=(mode == Modes.TRAIN))
    
    flat = tf.reshape(dropout2, [-1, 25690112])
    
    # Logits Layer
    logits = tf.layers.dense(inputs=flat, units=params['num_classes'])
    print("logits ", tf.shape(logits))
    
    # Define operations
    if mode in (Modes.PREDICT, Modes.EVAL):
        predicted_indices = tf.argmax(input=logits, axis=1)
        probabilities = tf.nn.softmax(logits, name='softmax_tensor')

    if mode in (Modes.TRAIN, Modes.EVAL):
        global_step = tf.train.get_or_create_global_step()
        label_indices = tf.cast(labels, tf.int32)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=tf.one_hot(label_indices, depth=params['num_classes']), logits=logits)
        tf.summary.scalar('OptimizeLoss', loss)

    if mode == Modes.PREDICT:
        predictions = {
            'classes': predicted_indices,
            'probabilities': probabilities
        }
        export_outputs = {
            SIGNATURE_NAME: tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    if mode == Modes.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        #optimizer = tf.keras.optimizers.SGD(learning_rate=params['learning_rate'])
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == Modes.EVAL:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(label_indices, predicted_indices)
        }
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

def _read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS])
    image = tf.cast(image, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.int32)

    return image, label

def _input_fn(input_dir, input_filename, mini_batch_size):
    file_path = os.path.join(input_dir, input_filename)
    filename_queue = tf.train.string_input_producer([file_path])

    image, label = _read_and_decode(filename_queue)
    images, labels = tf.train.batch([image, label], batch_size=mini_batch_size, capacity=1000 + 3 * mini_batch_size)
    return {INPUT_TENSOR_NAME: images}, labels

def _train_input_fn(training_dir, mini_batch_size):
    def train_input():
        return _input_fn(training_dir, TRAIN_RECORDS, mini_batch_size)
    return train_input


def _eval_input_fn(eval_dir, mini_batch_size):
    def eval_input():
        return _input_fn(eval_dir, EVAL_RECORDS, mini_batch_size)
    return eval_input

def _serving_input_fn():
    inputs = {INPUT_TENSOR_NAME: tf.placeholder(tf.float32, [None, IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def _parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--mini-batch-size', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    
    return parser.parse_known_args()

if __name__ == '__main__':
    args, unknown = _parse_args()
    
    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=_model_fn, model_dir=args.model_dir, params = {
            'num_classes': args.num_classes,
            'learning_rate': args.learning_rate
        }
    )
    
    train_spec = tf.estimator.TrainSpec(_train_input_fn(args.train, args.mini_batch_size), max_steps=args.max_steps)
    eval_spec = tf.estimator.EvalSpec(_eval_input_fn(args.test, args.mini_batch_size), steps=args.max_steps)
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    if args.current_host == args.hosts[0]:
        classifier.export_savedmodel(args.sm_model_dir, _serving_input_fn)