import sys
from argparse import ArgumentParser


def build_parser():
    parser = ArgumentParser()
    # Basic options
    parser.add_argument('-style_image', type=str,
                        default='examples/inputs/seated-nude.jpg',
                        dest='style_image',
                        help='Style target image',
                        metavar='STYLE_IMAGE', required=True)

    parser.add_argument('-style_blend_weights', type=str,
                        default=None,
                        dest='style_blend_weights')

    parser.add_argument('-content_image', type=str,
                        default='examples/inputs/tubingen.jpg',
                        dest='content_image', help='Content target image',
                        metavar='CONTENT_IMAGE', required=True)

    parser.add_argument('-image_size', type=int,
                        default=512,
                        dest='image_size',
                        help='Maximum height / width of generated image',
                        metavar='IMAGE_SIZE')

    parser.add_argument('-gpu', default=0, type=int,
                        dest='gpu',
                        help='Zero-indexed ID of the GPU to use; for \
                        CPU mode set -gpu = -1',
                        metavar='GPU')

    parser.add_argument('-multigpu_strategy', type=str, default='',
                        dest='multigpu_strategy',
                        help='Index of layers to split the network\
                        across GPUs',
                        metavar='MULTI_GPU')

    # Optimization options
    parser.add_argument('-content_weight', default=5, type=float,
                        dest='content_weight')
    parser.add_argument('-style_weight', default=100, type=float,
                        dest='style_weight')
    parser.add_argument('-tv_weight', default=0.0003, type=float,
                        dest='tv_weight')
    parser.add_argument('-num_iterations', default=1000, type=float,
                        dest='num_iter')
    parser.add_argument('-normalize_gradients', default=False,
                        dest='norm_gradients')
    parser.add_argument('-init', default='random', dest='init_image',
                        choices=['random', 'image'])
    parser.add_argument('-optimizer', default='lbfgs', dest='optimizer',
                        choices=['lbfgs', 'adam'])
    parser.add_argument('-learning_rate', default=5, type=float,
                        dest='learning_rate')

    # Output options
    parser.add_argument('-learning_rate', default=5, type=int,
                        dest='learning_rate')
    parser.add_argument('-print_iter', default=50, type=int,
                        dest='print_iter')
    parser.add_argument('-save_iter', default=100, type=int,
                        dest='save_iter')
    parser.add_argument('-output_image', default='out.jpg', type=str,
                        dest='output_image')

    # Other options
    parser.add_argument('-style_scale', default=1.0, type=float,
                        dest='style_scale')
    parser.add_argument('-original_colors', default=False,
                        dest='original_colors')
    parser.add_argument('-pooling', default='max',
                        choices=['max', 'avg'])
    parser.add_argument('-proto_file', type=str,
                        default='models/VGG_ILSVRC_19_layers_deploy.prototxt',
                        dest='proto_file')
    parser.add_argument('-model_file', type=str,
                        default='models/VGG_ILSVRC_19_layers.caffemodel',
                        dest='model_file')
    parser.add_argument('-backend', default='nn',
                        choices=['nn', 'cudnn', 'clnn'])
    parser.add_argument('-cudnn_autotune', default=False,
                        dest='cudnn_autotune')
    parser.add_argument('-seed', default=-1, type=int,
                        dest='seed')

    # Layers options
    parser.add_argument('-content_layers', type=str,
                        help='layers for content',
                        default='relu4_2',
                        dest='content_layers')
    parser.add_argument('-style_layers', type=str,
                        help='layers for style',
                        default='relu1_1,relu2_1,relu3_1,relu4_1,relu5_1',
                        dest='style_layers')
    # TODO: style_layer_weights

    return parser


# def check_opts(opts):


def main():
    parser = build_parser()
    opts = parser.parse_args()
    # check_opts(opts)

if __name__ == '__main__':
    main()
