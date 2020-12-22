import os
import sys
import time
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow_model_optimization as tfmot

import utils

MODEL_NAME = str(sys.argv[1])
assert MODEL_NAME in ('sunglasses', 'anonymous_1', 'anonymous_2', 'multi_trigger_multi_target')

CONFIG = utils.load_config()

DEVICE = CONFIG['gpu_device']

DATA_DIR = CONFIG['data_dir']
MODEL_DIR = CONFIG['model_dir']

VAL_DATA_FILE = CONFIG['val_data_file']
TEST_DATA_FILE = CONFIG['test_data_file']
POI_DATA_FILE = CONFIG[f'{MODEL_NAME}_data_file']

MODEL_FILENAME = f'{MODEL_NAME}_bd_net.h5'
WEIGHT_FILENAME = f'{MODEL_NAME}_bd_weights.h5'
OUTPUT_FILENAME = f'{MODEL_NAME}_pruned_net.h5'

BATCH_SIZE = CONFIG['prune']['batch_size']
STEPS = CONFIG['prune']['steps']
VAL_SPLIT = CONFIG['prune']['val_test_split']
TMP_DIR = CONFIG['tmp_dir']


def prune_model():
    model_file = '%s/%s' % (MODEL_DIR, MODEL_FILENAME)
    weight_file = '%s/%s' % (MODEL_DIR, WEIGHT_FILENAME)
    out_file = '%s/%s' % (MODEL_DIR, OUTPUT_FILENAME)

    X_test, Y_test = utils.data_loader('%s/%s' % (DATA_DIR, TEST_DATA_FILE), to_categ=False, preprocess=True)

    if not os.path.exists(out_file):
        X_val, Y_val = utils.data_loader('%s/%s' % (DATA_DIR, VAL_DATA_FILE), to_categ=False, preprocess=True)

        model = load_model(model_file)
        model.load_weights(weight_file)

        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model)

        model_for_pruning.compile(optimizer=model.optimizer,
                                  loss=model.loss,
                                  metrics=['accuracy'])

        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir=TMP_DIR)
        ]

        model_for_pruning.fit(X_val, Y_val, batch_size=BATCH_SIZE, epochs=STEPS,
                              validation_split=VAL_SPLIT, callbacks=callbacks)

        model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
        model_for_export.save(out_file)
    else:
        model_for_export = load_model(out_file)

    if MODEL_NAME not in ('multi_trigger_multi_target', 'anonymous_2'):
        X_poi, Y_poi = utils.data_loader('%s/%s' % (DATA_DIR, POI_DATA_FILE), to_categ=False, preprocess=True)

        base_model = load_model(model_file)
        base_model.load_weights(weight_file)

        base_test_res = np.argmax(base_model.predict(X_test), axis=1)
        base_poi_res = np.argmax(base_model.predict(X_poi), axis=1)
        base_test_acc = np.mean(np.equal(base_test_res, Y_test)) * 100
        base_poi_acc = np.mean(np.equal(base_poi_res, Y_poi)) * 100
        print('base model in clean test: {}, poisoned: {}'.format(base_test_acc, base_poi_acc))

        pruned_test_res = np.argmax(model_for_export.predict(X_test), axis=1)
        pruned_poi_res = np.argmax(model_for_export.predict(X_poi), axis=1)
        pruned_test_acc = np.mean(np.equal(pruned_test_res, Y_test)) * 100
        pruned_poi_acc = np.mean(np.equal(pruned_poi_res, Y_poi)) * 100
        print('pruned model in clean test: {}, poisoned: {}'.format(pruned_test_acc, pruned_poi_acc))


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    prune_model()


if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print('elapsed time %s s' % elapsed_time)
