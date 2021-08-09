"""Configurate arguments."""
import argparse

INPUT_IMAGE_SIZE = 512

NUM_FEATURE_MAP_CHANNEL = 1
FEATURE_MAP_SIZE = 1

def add_common_arguments(parser):
    """Add common arguments for training and inference."""
    parser.add_argument('--detector_weights',
                        help="The weights of pretrained detector.")
    parser.add_argument('--depth_factor', type=int, default=16,
                        help="Depth factor.")
    parser.add_argument('--disable_cuda', action='store_true',
                        help="Disable CUDA.")
    parser.add_argument('--gpu_id', type=int, default=0,
                        help="Select which gpu to use.")


def get_parser_for_training():
    """Return argument parser for training."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_directory', required=True,
                        help="The location of training dataset.")
    parser.add_argument('--optimizer_weights',
                        help="The weights of optimizer.")
    parser.add_argument('--batch_size', type=int, default=24,
                        help="Batch size.")
    parser.add_argument('--data_loading_workers', type=int, default=12,
                        help="Number of workers for data loading.")
    parser.add_argument('--num_epochs', type=int, default=20,
                        help="Number of epochs to train for.")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="The learning rate of back propagation.")
    parser.add_argument('--enable_visdom', action='store_true',
                        help="Enable Visdom to visualize training progress")
    add_common_arguments(parser)
    return parser

def get_parser_for_inference():
    """Return argument parser for inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',
                        help="image path if you choose to inference video.")
    add_common_arguments(parser)
    return parser


# def get_parser_for_export():
#     """Return argument parser for inference."""
#     parser = argparse.ArgumentParser()
#     add_common_arguments(parser)
#     return parser
