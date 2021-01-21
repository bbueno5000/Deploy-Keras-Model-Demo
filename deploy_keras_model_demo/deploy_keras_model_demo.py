"""
Web app framework.
"""
import flask
import keras
import numpy
import os
import re
import scipy
import sys

sys.path.append(os.path.abspath('./model'))

app = flask.Flask(__name__)

class App:
    """
    TODO: docstring
    """
    def __init__(self):
        """
        TODO: docstring
        """
        self.graph = init()
        self.model = init()

    def convertImage(imgData1):
        """
        Decoding an image from base64 into raw representation.
        """
        imgstr = re.search(r'base64,(.*)', imgData1).group(1)
        with open('output.png', 'wb') as output:
            output.write(imgstr.decode('base64'))

    @app.route('/')
    def index():
        """
        TODO: docstring
        """
        return flask.render_template('index.html')

    @app.route('/predict/', methods=['GET', 'POST'])
    def predict():
        """
        #whenever the predict method is called, we're going
        #to input the user drawn character as an image into the model
        #perform inference, and return the classification
        """
        imgData = flask.request.get_data()
        convertImage(imgData)
        print('debug')
        x = scipy.misc.imread('output.png', mode='L')
        x = numpy.invert(x)
        x = scipy.misc.imresize(x, (28,28))
        x = x.reshape(1, 28, 28, 1)
        print('debug2')
        with self.graph.as_default():
            out = self.model.predict(x)
            print(out)
            print(numpy.argmax(out, axis=1))
            print('debug3')
            response = numpy.array_str(numpy.argmax(out, axis=1))
            return response

class Test:
    """
    TODO: docstring
    """
    def __call__(self):
        """
        TODO: docstring
        """
        json_file = open('model.json','r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = keras.models.model_from_json(loaded_model_json)
        loaded_model.load_weights('model.h5')
        print('Loaded Model from disk')
        loaded_model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',metrics=['accuracy'])
        x = scipy.misc.imread('output.png', mode='L')
        x = numpy.invert(x)
        x = scipy.misc.imresize(x, (28,28))
        scipy.misc.imshow(x)
        x = x.reshape(1, 28, 28,1)
        out = loaded_model.predict(x)
        print(out)
        print(numpy.argmax(out, axis=1))

class Train:
    """
    Trains a simple convnet on the MNIST dataset.
    """
    def __call__(self):
        """
        TODO: docstring
        """
        batch_size = 128
        num_classes = 10
        epochs = 12
        img_rows, img_cols = 28, 28
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        if keras.backend.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(
            32, kernel_size=(3, 3),
            activation='relu', input_shape=input_shape))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(num_classes, activation='softmax'))
        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])
        model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        model_json = model.to_json()
        with open('model.json', 'w') as json_file:
            json_file.write(model_json)
        model.save_weights('model.h5')
        print('Saved model to disk')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
