# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import re
import random

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_prefix")
    args = parser.parse_args()
    return args

def write_split(outfile, lines, sorted_classes):
    random.shuffle(lines)
    with open(outfile,'wt') as fout:
        for line in lines:
            fout.write(line[0] + ' ' + str(sorted_classes.index(line[1])) + '\n')

if __name__ == '__main__':
    args = get_arguments()
    train_lines = []
    test_lines = []
    p = re.compile(ur'\/(.*)\/.*(\d)_\d{1,3}_(depth|rgb)crop\.png')  # group 1 is class, group 2 is angle seq
    classes = set()
    with open(args.input_file) as tmp:
        for line in tmp:
            res = re.search(p, line)
            class_name = res.group(1)
            classes.add(class_name)
            seq_id = res.group(2)

            add_to = train_lines
            if int(seq_id) >= 4:
                add_to = test_lines
            add_to.append((line.strip(), class_name))
    classes = sorted(classes)
    write_split(args.output_prefix + "_train_split_0.txt", train_lines, classes)
    write_split(args.output_prefix + "_test_split_0.txt", test_lines, classes)
    with open(args.output_prefix + "_labels.txt", "wt") as fout:
        for cl in classes:
            fout.write(cl + "\n")
#    import ipdb; ipdb.set_trace()
