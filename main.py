from flask import Flask, render_template, request, jsonify
import requests
import pickle
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def run():
    default_img_url = "http://zakishirwani.com/ai/cats/static/images/1000_F_97589769_t45CqXyzjz0KXwoBZT9PRaWGHRk5hQqQ.jpg"
    icon_img = "http://zakishirwani.com/ai/cats/static/images/cat-icon.jpg"
    return render_template("getUserInput.html", default_img_url=default_img_url, icon_img=icon_img)


@app.route('/result', methods=['GET', 'POST'])
def identify():
    with open("model.pkl", 'rb') as file:
        data = pickle.load(file)

    model = data['model']
    num_px = data['num_px']
    classes = data['classes']

    img_url = request.json

    try:
        img_data = requests.get(img_url).content
        image = Image.open(BytesIO(img_data))
        image = np.array(image.resize((num_px, num_px)))
        image = image / 255.
        image = image.reshape((1, num_px * num_px * 3)).T # reshape to column vector

        my_prediction = predict(model["w"], model["b"], image)
        y = str(np.squeeze(my_prediction))
        obj = classes[int(np.squeeze(my_prediction)),].decode("utf-8")
        return render_template("showResult.html", prediction=y, obj=obj, img_url=img_url)
    except:
        print("BAD_URL: " + img_url)
        return render_template("imageError.html")

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] > 0.5 :
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0

    return Y_prediction

if __name__ == '__main__':
    app.run(debug=True, port=5003)
