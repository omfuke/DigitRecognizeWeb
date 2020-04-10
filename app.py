from flask import Flask,request,render_template,redirect,url_for,jsonify
from keras.models import load_model
import numpy as np
import tensorflow as tf
from flask_cors import CORS



import json

app = Flask(__name__)

CORS(app)
model = load_model('model.h5', custom_objects=None,
                   compile=False
                   )


graph = tf.get_default_graph()
@app.route('/')
def index():

    return render_template('index.html')

@app.route("/predict",methods=['POST','GET'])
def home():
    my_array = np.array(list(request.form['data']))
    my_array = my_array
    my_array = my_array.reshape(-1, 28, 28, 1)
    with graph.as_default():
        y = model.predict(my_array)
    predicted_value = int(np.argmax(y, axis=1)[0])
    # print(request.args['name'])
    return jsonify({"predicted": predicted_value})
    # print(request.args['data'])
    # return request.form['data']

if __name__ == "__main__":
    app.run(debug=True)

