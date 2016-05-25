# -*- coding: utf-8 -*-
import pickle
from argparse import ArgumentParser
from os.path import join
import feature_handler

def get_arguments():
    parser = ArgumentParser(
        description='This tool extracts features and saves them into a pickle dictionary')
    parser.add_argument("data_dir", help="Where the images are located")
    parser.add_argument("filelist", help="File containing the list of all files")
    parser.add_argument("net_proto", help="Net deploy prototxt")
    parser.add_argument("net_model", help="Net model")
    parser.add_argument("output_filename")
    parser.add_argument("--mean_pixel", type=float)
    parser.add_argument("--mean_file")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--layer_name", help="Default is FC7", default='fc7')
    parser.add_argument("--use_gpu", default=True, help="If set false, will force CPU inference")
    parser.add_argument("--center_data", action="store_true", help="If set will center the data")
    parser.add_argument("--scale", type=float, default=None)
    args = parser.parse_args()
    return args


def make_features(args):
    print "Starting feature generation procedure..."
    f_extractor = feature_handler.FeatureCreator(
        args.net_proto, args.net_model, args.mean_pixel, args.mean_file, args.use_gpu,
         layer_name=args.layer_name)
    f_extractor.batch_size = args.batch_size
    f_extractor.center_data = args.center_data
    f_extractor.set_data_scale(args.scale)
    f_extractor.data_prefix = args.data_dir
    all_lines = open(args.filelist, 'rt').readlines()
    all_lines = [join(args.data_dir, line.strip()) for line in all_lines]
    # preload all features so that they are handled in batches
    f_extractor.prepare_features(all_lines)
    with open(args.output_filename, 'wb') as f:
        pickle.dump(f_extractor.features, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    args = get_arguments()
    make_features(args)
