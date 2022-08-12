from flask import Flask
from flask import render_template
from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from wtforms import SelectField, RadioField
from flask_bootstrap import Bootstrap
from wtforms import validators
import predict


class FileUploadForm(FlaskForm):
    upload = FileField("Imagen", [validators.regexp('^[^/\\]\.jpg$')])
    fruit = RadioField(choices=["apple", "mango"],)
    trained_model = SelectField(
        choices=["vgg16", "densenet121", "inceptionv3", "mobilenetv2"])


DEBUG = True
SECRET_KEY = "secret"

app = Flask(__name__)
app.config.from_object(__name__)
Bootstrap(app)


@app.route("/", methods=("GET", "POST"))
def index():
    form = FileUploadForm()
    filedata = ""
    if form.validate_on_submit():
        filedata = predict.predict_one(
            form.fruit.data, form.trained_model.data, form.upload.data.stream._file)

    return render_template("index.html", form=form, filedata=filedata)


if __name__ == "__main__":
    app.run()
