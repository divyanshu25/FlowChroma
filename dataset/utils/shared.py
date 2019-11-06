import itertools
from os import makedirs
from os.path import expanduser, join

# Default folders
dir_root = join(expanduser('~/workspace'), 'flowchroma')
dir_originals = join(dir_root, 'original')
dir_sampled = join(dir_root, 'sampled')
dir_resnet_images = join(dir_root, 'resized_resnet_images')
dir_resnet_csv = join(dir_root, 'resnet_csv_records')
dir_lab_records = join(dir_root, 'lab_records')
dir_tfrecord = join(dir_root, 'tfrecords')
dir_test = join(dir_root, 'test')
dir_test_results = join(dir_root, 'test_results')
dir_frame_lab_records = join(dir_root, 'frame_lab_records')
dir_frame_resnet_records = join(dir_root, 'frame_resnet_records')
checkpoint_url = join(dir_root,"inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")

frames_per_video = 3
default_nn_input_height = 240
default_nn_input_width = 320
resnet_input_height = 299
resnet_input_width = 299
resnet_video_chunk_size = 100
resnet_batch_size = 100

training_set_size = 2000
test_set_size = 239
validation_set_size = 200

resnet_output = 1000


def maybe_create_folder(folder):
    makedirs(folder, exist_ok=True)


def progressive_filename_generator(pattern='file_{}.ext'):
    for i in itertools.count():
        yield pattern.format(i)


def initialize():
    maybe_create_folder(dir_sampled)
    maybe_create_folder(dir_resnet_images)
    maybe_create_folder(dir_resnet_csv)
    maybe_create_folder(dir_lab_records)
    maybe_create_folder(dir_tfrecord)
    maybe_create_folder(dir_test)
    maybe_create_folder(dir_test_results)
    maybe_create_folder(dir_frame_lab_records)
    maybe_create_folder(dir_frame_resnet_records)


initialize()
