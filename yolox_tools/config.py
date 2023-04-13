import argparse

from mindyolo import parse_args


def get_config():
    parser = argparse.ArgumentParser(description='Train', parents=[])
    parser.add_argument('--is_distributed', type=int, default=0,
                               help='distributed training')
    config = parse_args(parser)
    return config

config = get_config()