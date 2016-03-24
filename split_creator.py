import argparse
import glob
import os
from random import randrange


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("output_path")
    args = parser.parse_args()
    return args


def create_split(categories, output_name, root_folder):
    l_root = len(root_folder)
    with open(output_name, "wt") as out_file:
        for cat in categories.keys():
            for instance in categories[cat]:
                images = glob.glob(os.path.join(
                    root_folder, cat, instance, "*.png"))
                for img in images:
                    out_file.write(img[l_root:] + " " + cat + "\n")


def get_instances(input_path):
    folders = [o for o in os.listdir(
        input_path) if os.path.isdir(os.path.join(input_path, o))]
    test_instances = {}
    train_instances = {}
    for folder in folders:
        category_path = os.join(input_path, folder)
        instances = [o for o in os.listdir(category_path) if os.path.isdir(
            os.path.join(category_path, o))]
        if len(instances) <= 1:
            continue
        test_instances[folder] = [instances.pop(randrange(len(instances)))]
        train_instances[folder] = instances
    return (train_instances, test_instances)

if __name__ == '__main__':
    args = parse_args()
    (train, test) = get_instances(args.input_folder)
    create_split(train, os.path.join(args.output_path, "train.txt", args.input_folder)
    create_split(test, os.path.join(args.output_path, "test.txt", args.input_folder)
