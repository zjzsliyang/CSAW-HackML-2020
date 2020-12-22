import os
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow_model_optimization as tfmot

import utils

DEVICE = '0'  # specify which GPU to use

DATA_DIR = 'data'  # data folder
VAL_DATA_FILE = 'clean_validation_data.h5'  # dataset file
TEST_DATA_FILE = 'clean_test_data.h5'
POI_DATA_FILE = 'sunglasses_poisoned_data.h5'
MODEL_DIR = 'models'  # model directory
MODEL_FILENAME = 'sunglasses_bd_net.h5'  # model file
WEIGHT_FILENAME = 'sunglasses_bd_weights.h5'
RESULT_DIR = 'results'  # directory for storing results

BATCH_SIZE = 10
STEPS = 10
VAL_SPLIT = 0.1
TMP_DIR = 'tmp'

os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    X_val, Y_val = utils.data_loader('%s/%s' % (DATA_DIR, VAL_DATA_FILE), to_categ=False, preprocess=True)
    X_test, Y_test = utils.data_loader('%s/%s' % (DATA_DIR, TEST_DATA_FILE), to_categ=False, preprocess=True)
    X_poi, Y_poi = utils.data_loader('%s/%s' % (DATA_DIR, POI_DATA_FILE), to_categ=False, preprocess=True)

    model_file = '%s/%s' % (MODEL_DIR, MODEL_FILENAME)
    weight_file = '%s/%s' % (MODEL_DIR, WEIGHT_FILENAME)
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

    base_model = load_model(model_file)
    base_model.load_weights(weight_file)

    base_test_res = np.argmax(base_model.predict(X_test), axis=1)
    base_poi_res = np.argmax(base_model.predict(X_poi), axis=1)
    base_test_acc = np.mean(np.equal(base_test_res, Y_test)) * 100
    base_poi_acc = np.mean(np.equal(base_poi_res, Y_poi)) * 100
    print('base model clean test: {}, poisoned: {}'.format(base_test_acc, base_poi_acc))

    pruned_test_res = np.argmax(model_for_export.predict(X_test), axis=1)
    pruned_poi_res = np.argmax(model_for_export.predict(X_poi), axis=1)
    pruned_test_acc = np.mean(np.equal(pruned_test_res, Y_test)) * 100
    pruned_poi_acc = np.mean(np.equal(pruned_poi_res, Y_poi)) * 100
    print('pruned model clean test: {}, poisoned: {}'.format(pruned_test_acc, pruned_poi_acc))

    model_for_export.save("./models/pruned_sunglasses_net.h5")
    return


if __name__ == '__main__':
    main()
