import random
import os
import sys
import shutil
sep = '\\' if sys.platform[:3] == 'win' else '/'


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


image_path = ''
save_path = ''
train_percent = 0.85
test_percent = 1-train_percent
image_files = sorted([os.path.join(image_path, x, y)
            for x in os.listdir(image_path) if x.endswith('images')
            for y in os.listdir(os.path.join(image_path, x)) if y.endswith('.jpg')])
label_files = sorted([os.path.join(image_path, x, y)
            for x in os.listdir(image_path) if x.endswith('masks')
            for y in os.listdir(os.path.join(image_path, x)) if y.endswith('.png')])
print(len(image_files), len(label_files))
assert len(image_files) == len(label_files)
length = len(image_files)
dataset_idx = range(length)
tv = int(length * train_percent)
tr = int(tv * train_percent)
train_idx = random.sample(dataset_idx, tv)
for i in dataset_idx:
    image_file = image_files[i]
    label_file = label_files[i]
    assert image_file.split(sep)[-1].split('.')[0] == label_file.split(sep)[-1].split('.')[0]
    image_name = image_file.split(sep)[-1].split('.')[0]
    print(i, image_name)

    if i in train_idx:
        save_image_path = save_path + 'TrainDataset2/' + 'images/'
        save_label_path = save_path + 'TrainDataset2/' + 'masks/'
        ensure_dir(save_image_path)
        ensure_dir(save_label_path)
        shutil.copyfile(src=image_file, dst=save_image_path + image_name + '.jpg')
        shutil.copyfile(src=label_file, dst=save_label_path + image_name + '.png')
    else:
        save_image_path = save_path + 'ValDataset/' + 'images/'
        save_label_path = save_path + 'ValDataset/' + 'masks/'
        ensure_dir(save_image_path)
        ensure_dir(save_label_path)
        shutil.copyfile(src=image_file, dst=save_image_path + image_name + '.jpg')
        shutil.copyfile(src=label_file, dst=save_label_path + image_name + '.png')

