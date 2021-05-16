from flask import Flask
from flask import send_file
import tensorflow as tf
import keras
from skimage.transform import resize


app = Flask(__name__)

model = keras.models.load_model('/Users/Main/Data_Science/Final-Project-main/Art2/Model 2/')
path = '/Users/Main/Data_Science/Final-Project-main/Art2/deploy_img/output.jpg'

@app.route('/')
def gen_image():

    seed2 = tf.random.normal([32, 100])
    generated_images2 = model(seed2, training = False)

    bottle_resized = resize(generated_images2[0], (480, 512))
        
    tf.keras.preprocessing.image.save_img(path, bottle_resized, data_format=None, file_format=None, scale=True)

    return send_file(path, cache_timeout=0) #Cache is an issue


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
