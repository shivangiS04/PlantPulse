import os
import shutil
import random

data_dir = 'data/PlantVillage'
output_dirs = ['train', 'val', 'test']
split_ratios = [0.7, 0.15, 0.15]  # train, val, test

for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        images = [img for img in os.listdir(class_path) if img.lower().endswith('.jpg')]
        random.shuffle(images)
        n_total = len(images)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])
        n_test = n_total - n_train - n_val

        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }

        for split, split_imgs in splits.items():
            split_dir = os.path.join(data_dir, split, class_name)
            os.makedirs(split_dir, exist_ok=True)
            for img in split_imgs:
                src = os.path.join(class_path, img)
                dst = os.path.join(split_dir, img)
                shutil.copy(src, dst)
        print(f"{class_name}: {n_train} train, {n_val} val, {n_test} test")
