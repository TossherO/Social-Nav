"""Image encoding shared by current attachments and saved visual references."""
import base64
import io
from PIL import Image


def image_part(data, size, format, *, preserve_png=False):
    with Image.open(io.BytesIO(data)) as source:
        if (preserve_png and format=='PNG' and source.format=='PNG' and source.mode=='RGB'
                and max(source.size)<=size and not getattr(source,'is_animated',False)):
            return png_part(data)
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
