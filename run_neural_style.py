#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, subprocess
from datetime import datetime
from argparse import ArgumentParser
from os.path import basename, splitext, expanduser, isfile, join
from os import listdir

RUN_SCRIPT_NAME = "neural_style.lua"

def build_parser():
    parser = ArgumentParser()

    # Basic options
    parser.add_argument('-style_image', type=str,
                        default='examples/inputs/seated-nude.jpg',
                        dest='style_image',
                        help='Style target image',
                        metavar='STYLE_IMAGE', required=True)
    '''
    parser.add_argument('-style_blend_weights', type=str,
                        default=None,
                        dest='style_blend_weights')
    '''
    parser.add_argument('-content_image', type=str,
                        default='examples/inputs/tubingen.jpg',
                        dest='content_image',
                        help='Content target image or folder with images',
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
    '''
    parser.add_argument('-multigpu_strategy', type=str, default='',
                        dest='multigpu_strategy',
                        help='Index of layers to split the network\
                        across GPUs',
                        metavar='MULTI_GPU')
    '''
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
                        dest='normalize_gradients')
    parser.add_argument('-init', default='random', dest='init',
                        choices=['random', 'image'])
    parser.add_argument('-optimizer', default='lbfgs', dest='optimizer',
                        choices=['lbfgs', 'adam'])
    parser.add_argument('-learning_rate', default=10, type=float,
                        dest='learning_rate')

    # Output options
    parser.add_argument('-print_iter', default=50, type=int,
                        dest='print_iter')
    parser.add_argument('-save_iter', default=100, type=int,
                        dest='save_iter')
    parser.add_argument('-output_path', default=None, type=str,
                        dest='output_path')

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

    # Runner options
    parser.add_argument('-time_markers',
                        default=False,
                        dest='time_markers')

    # TODO: style_layer_weights, lbfgs_num_correction

    return parser


def run_on_file(opts, input_file):
    output_dir = expanduser(opts.output_path) if opts.output_path != None else os.getcwd()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    optimizer_str = opts.optimizer
    if opts.optimizer == 'adam':
        optimizer_str += '_lr{0}'.format('%g'%(opts.learning_rate))
    out_file_name = '{style_image}_{content_image}_cw{content_weight}_sw{style_weight}_\
{optimizer}_sc{style_scale}_tv{tv_weight}'.format(
    style_image=splitext(basename(opts.style_image))[0],
    content_image=splitext(basename(input_file))[0],
    content_weight='%g'%(opts.content_weight),
    style_weight='%g'%(opts.style_weight),
    optimizer=optimizer_str,
    style_scale='%g'%(opts.style_scale),
    tv_weight=opts.tv_weight
    )
    if opts.normalize_gradients:
        out_file_name += '_norm'
    if opts.original_colors:
        out_file_name += '_colors'
    if opts.time_markers:
        now = datetime.now()
        out_file_name += '_{0}'.format(now.strftime('%Y%m%d_%H%M%S'))
    out_file_name += '.jpg'
    out_file_path = os.path.join(output_dir, out_file_name)

    run_script = 'th {script} -style_scale {style_scale} \
-init {init} -style_image "{style_image}" \
-content_image "{content_image}" -image_size {image_size} \
-output_image "{output_image}" \
-content_weight {content_weight} -style_weight {style_weight} \
-save_iter {save_iter} \
-print_iter {print_iter} \
-num_iterations {num_iterations} -content_layers {content_layers} \
-style_layers {style_layers} \
-gpu {gpu} -optimizer {optimizer} -tv_weight {tv_weight} \
-backend {backend} \
-seed {seed} {normalize_gradients} \
-learning_rate {learning_rate} \
-original_colors {original_colors} {cudnn_autotune} \
-pooling {pooling} -proto_file {proto_file} -model_file {model_file}'.format(
    script=RUN_SCRIPT_NAME,
    style_scale=opts.style_scale,
    init=opts.init,
    style_image=expanduser(opts.style_image),
    content_image=input_file,
    image_size=opts.image_size,
    output_image=out_file_path,
    content_weight=opts.content_weight,
    style_weight=opts.style_weight,
    save_iter=opts.save_iter,
    print_iter=opts.print_iter,
    num_iterations=opts.num_iter,
    content_layers=opts.content_layers,
    style_layers=opts.style_layers,
    gpu=opts.gpu,
    optimizer=opts.optimizer,
    tv_weight=opts.tv_weight,
    backend=opts.backend,
    seed=opts.seed,
    normalize_gradients=('-normalize_gradients' if opts.normalize_gradients else ''),
    learning_rate=opts.learning_rate,
    original_colors=('1' if opts.original_colors else '0'),
    cudnn_autotune=('-cudnn_autotune' if opts.cudnn_autotune else ''),
    pooling=opts.pooling,
    proto_file=opts.proto_file,
    model_file=opts.model_file)

    print('Script \'{0}\''.format(run_script))

    with subprocess.Popen(run_script, shell=True,
                          stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          universal_newlines=True) as proc:
        print(proc.stdout.read())
        print(proc.stderr.read())


# Print iterations progress
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '█' * filledLength + '-' * (barLength - filledLength)
    text = '\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)
    sys.stdout.write(text),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def main():
    parser = build_parser()
    opts = parser.parse_args()
    # check_opts(opts)

    input_path = expanduser(opts.content_image)
    print('Input path is {0}'.format(input_path))
    if os.path.isfile(input_path):
        run_on_file(opts, input_path)
    else:
        input_files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
        count = len(input_files)
        i = 1
        for input_filename in input_files:
            print('\nInput file is {0}'.format(input_filename))
            run_on_file(opts, join(input_path, input_filename))
            printProgress(i, count, prefix='Progress:',
                          suffix='Complete', barLength=100)
            i += 1

if __name__ == '__main__':
    main()
