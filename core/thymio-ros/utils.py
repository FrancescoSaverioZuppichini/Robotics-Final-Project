import numpy as np

from PIL import Image, ImageFont, ImageDraw

def callback(fun,*args, **kwargs):
    def delay(x):
        fun(x, *args, **kwargs)
    return delay


class Params:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

def draw_boxes(image, out_boxes, out_classes, out_scores, colors, class_names, is_array=False):
    if is_array:
        image = Image.fromarray(image, 'RGB')
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    draw = None

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=tuple(colors[c]))
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=tuple(colors[c]))
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)

    if draw: del draw

    return image

def unbox(res):
    res = res['res']

    out_scores = list(map(lambda x: x['score'], res))
    out_boxes = list(map(lambda x: x['boxes'], res))
    out_classes = list(map(lambda x: x['class'], res))
    out_classes_idx = list(map(lambda x: x['class_idx'], res))

    return out_scores, out_boxes, out_classes, out_classes_idx
