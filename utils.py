import h5py
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

NUM_CLASSES = 1283  # total number of classes in the model

def data_loader(filepath):
    dataset = load_dataset(filepath, keys=['data', 'label'])
    x_data = np.array(dataset['data'], dtype='float32')
    y_data = np.array(dataset['label'], dtype='int32')
    x_data = x_data.transpose((0, 2, 3, 1))
    new_y = []
    for y_d in y_data:
        item = np.zeros(NUM_CLASSES)
        item[y_d] = 1
        new_y.append(item)
    return x_data, np.array(new_y)


def data_preprocess(x_data):
    return x_data / 255


def dump_image(x, filename, format):
    img = image.array_to_img(x, scale=False)
    img.save(filename, format)
    return


def fix_gpu_memory(mem_fraction=1):
    import keras.backend as K

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_fraction)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    tf_config.allow_soft_placement = True
    init_op = tf.global_variables_initializer()
    sess = tf.Session(config=tf_config)
    sess.run(init_op)
    K.set_session(sess)

    return sess


def load_dataset(data_filename, keys=None):
    """assume all datasets are numpy arrays"""
    dataset = {}
    with h5py.File(data_filename, 'r') as hf:
        if keys is None:
            for name in hf:
                dataset[name] = np.array(hf.get(name))
        else:
            for name in keys:
                dataset[name] = np.array(hf.get(name))

    return dataset