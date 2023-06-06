from flask import Flask,render_template,request
import numpy as np
from PIL import Image
from keras.models import load_model
from tensorflow import keras
import tensorflow as tf

# Get the TensorFlow configuration
config = tf.compat.v1.ConfigProto()

# Set the desired memory growth and limit
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.0  # Adjust as needed



# List available physical devices
physical_devices = tf.config.list_physical_devices()

print(physical_devices)



session = tf.compat.v1.Session(config=config)
with open('classesdict.txt', 'r') as file:
        file_contents = file.read()
        classess = eval(file_contents)
path = 'C:/Users/z004kdpf/bpc projrct/car class/150e.model'
#c:\users\z004kdpf\appdata\local\programs\python\python311\lib\site-packages (from h5py) (1.23.5)
model_rsn = keras.models.load_model('150e.model')

#model_rsn = load_model(('C:/Users/z004kdpf/bpc projrct/car class/model123.pkl','rb'))
#model = load_model(open('C:/Users/z004kdpf/bpc projrct/car class/model123.pkl','rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_car():
    print()
    img123  = request.files['image']
    img = Image.open(img123)
    img = img.convert('RGB')
    print(img,'-----------------------------------------1',type(img))
    img = img.resize((224, 224))
    print('2')

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0
    print('3')
    predictions = model_rsn.predict(img_array)
    print('4')
    print(predictions)
    print('done')
    y_classes = predictions.argmax(axis=-1)
    print(str(classess[int(y_classes[0])]))
    return (str(classess[int(y_classes[0])]))


if __name__=='__main__':
    app.run(host= '0.0.0.0',port=8080)