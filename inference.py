
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Runs struct2depth at inference. Produces depth estimates, ego-motion and object motion."""

# Example usage:
# python inference.py \
#   --logtostderr \
#   --file_extension png \
#   --depth \
#   --output_dir output \
#   --model_ckpt "./trained-models/model-199160"


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags
from absl import logging
#import matplotlib.pyplot as plt
import model
import numpy as np
import fnmatch
import tensorflow as tf
import nets
import util
import cv2

# Segmentation model
from segmentation import VisualizeResults

gfile = tf.gfile

# CMAP = 'plasma'

INFERENCE_MODE_SINGLE = 'single'  # Take plain single-frame input.
INFERENCE_MODE_TRIPLETS = 'triplets'  # Take image triplets as input.
# For KITTI, we just resize input images and do not perform cropping. For
# Cityscapes, the car hood and more image content has been cropped in order
# to fit aspect ratio, and remove static content from the images. This has to be
# kept at inference time.
INFERENCE_CROP_NONE = 'none'
INFERENCE_CROP_CITYSCAPES = 'cityscapes'


flags.DEFINE_string('output_dir', None, 'Directory to store predictions.')
flags.DEFINE_string('file_extension', 'png', 'Image data file extension of '
                    'files provided with input_dir. Also determines the output '
                    'file format of depth prediction images.')
flags.DEFINE_bool('depth', True, 'Determines if the depth prediction network '
                  'should be executed and its predictions be saved.')
flags.DEFINE_bool('egomotion', False, 'Determines if the egomotion prediction '
                  'network should be executed and its predictions be saved. If '
                  'inference is run in single inference mode, it is assumed '
                  'that files in the same directory belong in the same '
                  'sequence, and sorting them alphabetically establishes the '
                  'right temporal order.')
flags.DEFINE_string('model_ckpt', None, 'Model checkpoint to evaluate.')
flags.DEFINE_string('input_dir', None, 'Directory containing image files to '
                    'evaluate. This crawls recursively for images in the '
                    'directory, mirroring relative subdirectory structures '
                    'into the output directory.')
flags.DEFINE_string('input_list_file', None, 'Text file containing paths to '
                    'image files to process. Paths should be relative with '
                    'respect to the list file location. Relative path '
                    'structures will be mirrored in the output directory.')
flags.DEFINE_integer('batch_size', 1, 'The size of a sample batch')
flags.DEFINE_integer('img_height', 128, 'Input frame height.')
flags.DEFINE_integer('img_width', 416, 'Input frame width.')
flags.DEFINE_integer('seq_length', 3, 'Number of frames in sequence.')
flags.DEFINE_enum('architecture', nets.RESNET, nets.ARCHITECTURES,
                  'Defines the architecture to use for the depth prediction '
                  'network. Defaults to ResNet-based encoder and accompanying '
                  'decoder.')
flags.DEFINE_boolean('imagenet_norm', True, 'Whether to normalize the input '
                     'images channel-wise so that they match the distribution '
                     'most ImageNet-models were trained on.')
flags.DEFINE_bool('use_skip', True, 'Whether to use skip connections in the '
                  'encoder-decoder architecture.')
flags.DEFINE_bool('joint_encoder', False, 'Whether to share parameters '
                  'between the depth and egomotion networks by using a joint '
                  'encoder architecture. The egomotion network is then '
                  'operating only on the hidden representation provided by the '
                  'joint encoder.')
flags.DEFINE_bool('shuffle', False, 'Whether to shuffle the order in which '
                  'images are processed.')
flags.DEFINE_bool('flip', False, 'Whether images should be flipped as well as '
                  'resulting predictions (for test-time augmentation). This '
                  'currently applies to the depth network only.')
flags.DEFINE_enum('inference_mode', INFERENCE_MODE_SINGLE,
                  [INFERENCE_MODE_SINGLE,
                   INFERENCE_MODE_TRIPLETS],
                  'Whether to use triplet mode for inference, which accepts '
                  'triplets instead of single frames.')
flags.DEFINE_enum('inference_crop', INFERENCE_CROP_NONE,
                  [INFERENCE_CROP_NONE,
                   INFERENCE_CROP_CITYSCAPES],
                  'Whether to apply a Cityscapes-specific crop on the input '
                  'images first before running inference.')
flags.DEFINE_bool('use_masks', False, 'Whether to mask out potentially '
                  'moving objects when feeding image input to the egomotion '
                  'network. This might improve odometry results when using '
                  'a motion model. For this, pre-computed segmentation '
                  'masks have to be available for every image, with the '
                  'background being zero.')

FLAGS = flags.FLAGS

flags.mark_flag_as_required('output_dir')
flags.mark_flag_as_required('model_ckpt')


def _run_inference(output_dir=None,
                   file_extension='png',
                   depth=True,
                   egomotion=False,
                   model_ckpt=None,
                   input_dir=None,
                   input_list_file=None,
                   batch_size=1,
                   img_height=128,
                   img_width=416,
                   seq_length=3,
                   architecture=nets.RESNET,
                   imagenet_norm=True,
                   use_skip=True,
                   joint_encoder=True,
                   shuffle=False,
                   flip_for_depth=False,
                   inference_mode=INFERENCE_MODE_SINGLE,
                   inference_crop=INFERENCE_CROP_NONE,
                   use_masks=False):
    """Runs inference. Refer to flags in inference.py for details."""
    inference_model = model.Model(is_training=False,
                                  batch_size=batch_size,
                                  img_height=img_height,
                                  img_width=img_width,
                                  seq_length=seq_length,
                                  architecture=architecture,
                                  imagenet_norm=imagenet_norm,
                                  use_skip=use_skip,
                                  joint_encoder=joint_encoder)
    vars_to_restore = util.get_vars_to_save_and_restore(model_ckpt)
    saver = tf.train.Saver(vars_to_restore)
    sv = tf.train.Supervisor(logdir='/tmp/', saver=None)
    with sv.managed_session() as sess:
        saver.restore(sess, model_ckpt)
        if not gfile.Exists(output_dir):
            gfile.MakeDirs(output_dir)
        logging.info('Predictions will be saved in %s.', output_dir)

        basepath_in = os.getcwd()
        segmented_weights_dir = os.path.join(
            "", basepath_in, "segmentation/pretrained", "")
        print("Segmentation weights dir:", segmented_weights_dir)
        output_dir = os.path.join(output_dir, basepath_in, output_dir, "")

        # Feeding images from webcam
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Recording', cv2.WINDOW_AUTOSIZE)
        i = 0

        while True:
            ret_val, frame = cap.read()

            if depth:
                logging.info('%s processed.', i)
                # Resizing the image to (416*128)
                # final_image = cv2.resize(
                #     frame, (img_width, img_height), interpolation=cv2.INTER_AREA)
                # Getting the segmentation mask
                # segmentation_mask = VisualizeResults.main(
                #     frame, output_dir, segmented_weights_dir)
                # input_image_seq = []
                # input_seg_seq = []
                # input_image_seq.append(final_image)
                # input_seg_seq.append(cv2.resize(segmentation_mask,
                #                                 resize=(img_width, img_height),
                #                                 interpolation='nn'))
                # if use_masks:
                #   input_image_stack = mask_image_stack(input_image_stack,
                #                                         input_seg_seq)
                # est_egomotion = np.squeeze(inference_model.inference_egomotion(
                #     input_image_stack, sess))
                # input_image_stack = np.concatenate(input_image_seq, axis=2)
                # input_image_stack = np.expand_dims(input_image_stack, axis=0)
                # Resizing the image to (416*128)
                final_image = cv2.resize(frame, (img_width, img_height), interpolation=cv2.INTER_AREA)

                # Estimating depth
                est_depth = inference_model.inference_depth(
                    [final_image], sess)
                # est_depth is the matrix of depths

                # Creating output
                color_map = util.normalize_depth_for_display(
                    np.squeeze(est_depth[0]))
                visualization = np.concatenate((final_image, color_map), axis=0)
                output_vis = os.path.join(
                    output_dir, str(i) + "-depth" + '.png')
                # output_image = os.path.join(
                #     output_dir, str(i) + '.png')
                util.save_image(output_vis, visualization, file_extension)
                # util.save_image(output_vis, color_map, file_extension)
                # util.save_image(output_image, final_image, file_extension)
            i = i + 1
            # break
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()

        print("Done!")


def mask_image_stack(input_image_stack, input_seg_seq):
    """Masks out moving image contents by using the segmentation masks provided.

    This can lead to better odometry accuracy for motion models, but is optional
    to use. Is only called if use_masks is enabled.
    Args:
      input_image_stack: The input image stack of shape (1, H, W, seq_length).
      input_seg_seq: List of segmentation masks with seq_length elements of shape
                     (H, W, C) for some number of channels C.

    Returns:
      Input image stack with detections provided by segmentation mask removed.
    """
    background = [mask == 0 for mask in input_seg_seq]
    background = reduce(lambda m1, m2: m1 & m2, background)
    # If masks are RGB, assume all channels to be the same. Reduce to the first.
    if background.ndim == 3 and background.shape[2] > 1:
        background = np.expand_dims(background[:, :, 0], axis=2)
    elif background.ndim == 2:  # Expand.
        background = np.expand_dism(background, axis=2)
    # background is now of shape (H, W, 1).
    background_stack = np.tile(background, [1, 1, input_image_stack.shape[3]])
    return np.multiply(input_image_stack, background_stack)


def main(_):
    # if (flags.input_dir is None) == (flags.input_list_file is None):
    #  raise ValueError('Exactly one of either input_dir or input_list_file has '
    #                   'to be provided.')
    # if not flags.depth and not flags.egomotion:
    #  raise ValueError('At least one of the depth and egomotion network has to '
    #                   'be called for inference.')
    # if (flags.inference_mode == inference_lib.INFERENCE_MODE_TRIPLETS and
    #    flags.seq_length != 3):
    #  raise ValueError('For sequence lengths other than three, single inference '
    #                   'mode has to be used.')

    _run_inference(output_dir=FLAGS.output_dir,
                   file_extension=FLAGS.file_extension,
                   depth=FLAGS.depth,
                   egomotion=FLAGS.egomotion,
                   model_ckpt=FLAGS.model_ckpt,
                   input_dir=FLAGS.input_dir,
                   input_list_file=FLAGS.input_list_file,
                   batch_size=FLAGS.batch_size,
                   img_height=FLAGS.img_height,
                   img_width=FLAGS.img_width,
                   seq_length=FLAGS.seq_length,
                   architecture=FLAGS.architecture,
                   imagenet_norm=FLAGS.imagenet_norm,
                   use_skip=FLAGS.use_skip,
                   joint_encoder=FLAGS.joint_encoder,
                   shuffle=FLAGS.shuffle,
                   flip_for_depth=FLAGS.flip,
                   inference_mode=FLAGS.inference_mode,
                   inference_crop=FLAGS.inference_crop,
                   use_masks=FLAGS.use_masks)


if __name__ == '__main__':
    app.run(main)
