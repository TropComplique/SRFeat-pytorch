import os
from PIL import Image

MIN_SIZE = 296
FOLDER = '/home/dan/datasets/COCO/images/train2017/'

to_remove = []
names = os.listdir(FOLDER)

for n in names:
    p = os.path.join(FOLDER, n)
    w, h = Image.open(p).size
    if w < MIN_SIZE or h < MIN_SIZE:
        to_remove.append(p)

print('number of small images:', len(to_remove))
print('total number of images:', len(names))

for p in to_remove:
    os.remove(p)
