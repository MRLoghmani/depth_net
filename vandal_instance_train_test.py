# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import re
import random

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_prefix")
    parser.add_argument("--exclude_morph", action="store_true")
    parser.add_argument("--include_morph", action="store_true")
    args = parser.parse_args()
    return args

def write_split(outfile, lines, sorted_classes):
#    random.shuffle(lines)
    with open(outfile,'wt') as fout:
        for line in lines:
            fout.write(line[0] + ' ' + str(sorted_classes.index(line[1])) + '\n')

if __name__ == '__main__':
    args = get_arguments()
    train_lines = []
    test_lines = []
#    p = re.compile(ur'\/(.*)\/.*(\d)_\d{1,3}_(depth|rgb)crop\.png')  # group 1 is class, group 2 is angle seq
    p = re.compile(ur'\/(.*)\/(.*)\/')  # group 1 is class, 2 is instance
    classes = set()
    with open(args.input_file) as tmp:
        for line in tmp:
            res = re.search(p, line)
            class_name = res.group(1)
            instance = res.group(2)
            if args.exclude_morph and "_morph" in instance.lower():
                continue
            if args.include_morph and "_morph" in instance.lower():
                instance = instance[instance.index("_morph")]
            instance_class = class_name + "_" + instance
            classes.add(instance_class)


            add_to = train_lines
            if random.random() > 0.95:
                add_to = test_lines
            add_to.append((line.strip(), instance_class))
    classes = sorted(classes)
    write_split(args.output_prefix + "_train_split_0.txt", train_lines, classes)
    write_split(args.output_prefix + "_test_split_0.txt", test_lines, classes)
    with open(args.output_prefix + "_labels.txt", "wt") as fout:
        for cl in classes:
            fout.write(cl + "\n")
#    import ipdb; ipdb.set_trace()
