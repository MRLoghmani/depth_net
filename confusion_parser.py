from argparse import ArgumentParser
import numpy as np


def get_arguments():
    parser = ArgumentParser(description='Analysis of confusion matrices')
    parser.add_argument("conf_name")
    parser.add_argument("labels")
    parser.add_argument("--n_splits", type=int, default=10)
    args = parser.parse_args()
    return args


def load_confusion_matrices(partial_path, splits, delimiter=" "):
    conf_avg = None
    for x in range(splits):
        with open(partial_path + '_' + str(x) + '.csv') as tmp:
            conf = np.loadtxt(tmp, delimiter=delimiter)
            if conf_avg is None:
                conf_avg = conf
            else:
                conf_avg += conf
    return conf_avg.astype('float32') / splits


def write_classrank(class_errors, labels, out_path):
    idx = np.argsort(class_errors)
    with open(out_path + "_sorted.csv", 'wt') as csv_file:
        for i in idx:
            csv_file.write(labels[i] + "; " + str(class_errors[i]) + "\n")
    with open(out_path + "_raw.csv", 'wt') as csv_file:
        for val in class_errors:  # also save them label wise
            csv_file.write(str(val) + "\n")


if __name__ == '__main__':
    args = get_arguments()
    avg_conf = load_confusion_matrices(args.conf_name, args.n_splits, delimiter=";")
    norm_cm = avg_conf.astype('float') / avg_conf.sum(axis=1)[:, np.newaxis]
    np.savetxt(args.conf_name + '_avg.csv', avg_conf, delimiter=';')
    #np.fill_diagonal(norm_cm, 0)
    class_errors = norm_cm.diagonal()  # norm_cm.sum(axis=1)
    with open(args.labels) as tmp:
        labels = tmp.readlines()
    labels = [l.strip() for l in labels]
    write_classrank(class_errors, labels, args.conf_name + "_rank")
