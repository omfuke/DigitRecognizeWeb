from flask import Flask,request,render_template,redirect,url_for,jsonify


app = Flask(__name__)

@app.route('/',methods = ['POST','GET'])
def login():
    if request.method == 'POST':
        ar = request.form["nm"]
        print(ar)
        return redirect(url_for("digit",arr = ar))
    else:
        return render_template('base.html')

@app.route("/<int:arr>")
def digit(arr):
    model = 5
    predicted = model * arr

    return jsonify({"predicted": predicted})


if __name__ == "__main__":
    app.run(debug=True)