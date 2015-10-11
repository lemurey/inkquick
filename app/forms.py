from flask.ext.wtf import Form
from wtforms import StringField


class ImageForm(Form):
    image_url = StringField('image_url')
    