from flask import Flask,request,render_template,redirect,url_for,jsonify
import tensorflow as tf
import numpy as np
from flask_cors import CORS



import json

app = Flask(__name__)

CORS(app)
model = tf.keras.models.load_model('model.h5', custom_objects=None,
                   compile=False
                   )


@app.route('/')
def index():

    return render_template('index.html')

@app.route("/predict",methods=['POST','GET'])
def home():
    my_array = np.array(list(request.form['data']))
    my_array = my_array
    my_array = my_array.reshape(-1, 28, 28, 1)

    y = model.predict(my_array)
    predicted_value = int(np.argmax(y, axis=1)[0])
    # print(request.args['name'])
    return jsonify({"predicted": predicted_value})
    # print(request.args['data'])
    # return request.form['data']

if __name__ == "__main__":
    app.run(debug=True)

