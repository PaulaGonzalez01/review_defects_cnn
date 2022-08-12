import base64
from flask import Flask, request
from flask import render_template
import predict

DEBUG = True
SECRET_KEY = "secret"

app = Flask(__name__)
app.config.from_object(__name__)


@app.route("/", methods=("GET", "POST"))
def index():
    form = request.form
    files = request.files
    results = {}
    if request.method == "POST":
        fruit = 'apple' if form['fruit'] else 'mango'
        image = files['image']

        results['fruit'] = fruit
        results['model'] = form['model']
        result = predict.predict_one(
            fruit, form['model'], image.stream._file)
        results['status'] = result[0]
        results['acc'] = result[1] * 100
        image_base64 = base64.b64encode(
            image.stream._file.getvalue(),
            ).decode('utf-8')
        results['image'] = image_base64

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run()
