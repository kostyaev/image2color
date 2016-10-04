import numpy as np
import math
from PIL import Image
import random
import sys


def crop(img, crop_size=224, center=False):
    width, height = img.size
    if not center:
        h_off = random.randint(0, height - crop_size)
        w_off = random.randint(0, width - crop_size)
    else:
        h_off = (height - crop_size) / 2
        w_off = (width - crop_size) / 2
    return img.crop((w_off,h_off,w_off+crop_size, h_off+crop_size))


def get_color(img):
    img = img.resize((80,80), Image.BILINEAR)
    arr = np.array(crop(img, 64, center=True))
    arr = arr.reshape(arr.shape[0]*arr.shape[1], 3)
    rows = []
    for row in arr:
        mean = row.mean()
        if 240 > mean > 10:
            rows.append(row)

    if len(rows) % 2 != 0:
        rows = rows[:-1]

    n = int(math.sqrt(len(rows)))
    rows = rows[:n**2]

    res_orig = np.array(rows).reshape(n, n, 3)

    img_c = Image.fromarray(res_orig).resize((2,2), Image.BILINEAR)
    res = np.array(img_c)
    img_c.resize((200,200), Image.NEAREST)

    def get_bright(row):
        R, G, B = row
        Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
        return Y

    n_dark = 0

    Y_res = res.copy()[:,:,0]
    for idx, row in enumerate(res):
        for idy, col in enumerate(row):
            Y = get_bright(col)
            if Y < 150:
                n_dark += 1
            Y_res[idx,idy] = Y

    search_value = Y_res.min() if n_dark > 1 else Y_res.max()
    x,y = zip(*np.where(Y_res == search_value))[0]
    return res[x,y]


if __name__ == '__main__':
    names = sys.argv[1:]
    for name in names:
        img = Image.open(name)
        res = Image.fromarray(get_color(img)[np.newaxis, np.newaxis, :]).resize(img.size, Image.NEAREST)
        res.save(name.rsplit('.', 1)[0] + '-color.jpg')