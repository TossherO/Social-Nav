"""Image encoding shared by current attachments and saved visual references."""
import base64
import io
from PIL import Image


def image_part(data, size, format):
    with Image.open(io.BytesIO(data)) as source:
        picture = source.convert('RGB')
        picture.thumbnail((size,size))
        buffer = io.BytesIO()
        picture.save(buffer,format=format,**(dict(quality=80) if format=='JPEG' else {}))
    mime = 'jpeg' if format=='JPEG' else 'png'
    return dict(type='image_url',image_url=dict(url='data:image/'+mime+';base64,'+
                base64.b64encode(buffer.getvalue()).decode()))


def png_part(data):
    return dict(type='image_url',image_url=dict(url='data:image/png;base64,'+
                base64.b64encode(data).decode()))
