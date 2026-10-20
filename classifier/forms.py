from django import forms
from django.conf import settings
from PIL import Image


class UploadForm(forms.Form):
    image = forms.ImageField(label="Dermoscopic image")

    def clean_image(self):
        f = self.cleaned_data["image"]
        if f.size > settings.MAX_UPLOAD_BYTES:
            raise forms.ValidationError("Image is larger than 10 MB.")
        f.seek(0)
        with Image.open(f) as im:
            if min(im.size) < 64:
                raise forms.ValidationError("Image is too small (min 64 px per side).")
        f.seek(0)
        return f
