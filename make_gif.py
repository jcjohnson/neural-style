import argparse
import glob

import images2gif
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description='Start and end for primary key')
    parser.add_argument('--direction', '-d', type=str, nargs='?', default='backwards')
    parser.add_argument('--prefix', '-p', type=str)
    parser.add_argument('--output', '-o', type=str, nargs='?', default=None)
    parser.add_argument('--original', '-r', type=str, nargs='?', default=None)

    namespace = parser.parse_args()
    prefix = namespace.prefix
    original = namespace.original
    direction = namespace.direction
    output = namespace.output or 'output/gifs/{0}.gif'.format(prefix.split('/')[-1]+'_'+direction)

    images = [Image.open(i) for i in glob.glob(prefix+'*')]
    images = images[1:] + [images[0]]  # hack bc final iteration has no number
    if original:
        original_raw = Image.open(original)
        original_resized = original_raw.resize(images[0].size)
        images.append(original_resized)
    # name reflects iteration direction. backwards is normal to weird. forwards is weird to normal
    if direction == 'backwards':
        images = images[::-1]

    images2gif.writeGif(output, images, duration=.2)  # TODO: maybe pause at the end? use duration=[list] to change

if __name__ == '__main__':
    main()
