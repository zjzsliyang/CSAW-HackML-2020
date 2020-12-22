import sys
import keras
import numpy as np
from utils import data_loader, data_preprocess

clean_data_filename = str(sys.argv[1])
model_filename = str(sys.argv[2])


def main():
    x_test, y_test = data_loader(clean_data_filename)
    x_test = data_preprocess(x_test)

    bd_model = keras.models.load_model(model_filename)

    clean_label_p = np.argmax(bd_model.predict(x_test), axis=1)
    class_accu = np.mean(np.equal(clean_label_p, y_test)) * 100
    print('Classification accuracy:', class_accu)


if __name__ == '__main__':
    main()
