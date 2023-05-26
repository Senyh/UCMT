import random
import os
import sys
import shutil
sep = '\\' if sys.platform[:3] == 'win' else '/'


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


image_path = ''
train_percent = 0.7
test_percent = 1-train_percent
image_files = [os.path.join(image_path, x, y)
            for x in os.listdir(image_path) if x.endswith('images')
            for y in os.listdir(os.path.join(image_path, x)) if y.endswith('.jpg')]
label_files = [os.path.join(image_path, x, y)
            for x in os.listdir(image_path) if x.endswith('masks')
            for y in os.listdir(os.path.join(image_path, x)) if y.endswith('segmentation.png')]
print(len(image_files), len(label_files))
assert len(image_files) == len(label_files)
length = len(image_files)
dataset_idx = range(length)
tv = int(length * train_percent)
tr = int(tv * train_percent)
train_idx = random.sample(dataset_idx, tv)
for i in dataset_idx:
    print(i)
    image_file = image_files[i]
    label_file = label_files[i]
    assert image_file.split(sep)[-1].split('.')[0] == label_file.split(sep)[-1].split('.')[0][:-13]
    image_name = image_file.split(sep)[-1].split('.')[0]

    if i in train_idx:
        save_image_path = image_path + 'TrainDataset/' + 'images/'
        save_label_path = image_path + 'TrainDataset/' + 'masks/'
        ensure_dir(save_image_path)
        ensure_dir(save_label_path)
        shutil.copyfile(src=image_file, dst=save_image_path + image_name + '.jpg')
        shutil.copyfile(src=label_file, dst=save_label_path + image_name + '.png')
    else:
        save_image_path = image_path + 'TestDataset/' + 'images/'
        save_label_path = image_path + 'TestDataset/' + 'masks/'
        ensure_dir(save_image_path)
        ensure_dir(save_label_path)
        shutil.copyfile(src=image_file, dst=save_image_path + image_name + '.jpg')
        shutil.copyfile(src=label_file, dst=save_label_path + image_name + '.png')


