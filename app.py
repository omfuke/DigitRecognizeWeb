from flask import Flask,request,render_template,redirect,url_for,jsonify
from keras.models import load_model
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__)
model = load_model('model.h5', custom_objects=None,
                   compile=False
                   )

graph = tf.get_default_graph()


@app.route("/",methods=['POST','GET'])
def home():
    if request.method == 'POST':
        img = request.files['img']
        response = img.read()
        img = cv2.imdecode(np.fromstring(response, np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(28,28))
        print(img)
        my_array = img/255.0
        my_array = my_array.reshape(-1,28,28,1)

        with graph.as_default():
            y = model.predict(my_array)
        predicted_value = int(np.argmax(y, axis=1)[0])

        return jsonify({"predicted": predicted_value})

    else:
        return render_template('base.html')



if __name__ == "__main__":
    app.run(debug=True)

