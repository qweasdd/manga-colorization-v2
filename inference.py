import os
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from colorizator import MangaColorizator


def process_image(image, colorizator, args):
    colorizator.set_image(image, args.size, args.denoiser, args.denoiser_sigma)
    return colorizator.colorize()


def colorize_single_image(image_path, save_path, colorizator, args):
    image = plt.imread(image_path)
    colorization = process_image(image, colorizator, args)
    plt.imsave(save_path, colorization)
    return True


def get_unique_save_path(save_path):
    base, ext = os.path.splitext(save_path)
    counter = 1
    while os.path.exists(save_path):
        save_path = f"{base}_{counter}{ext}"
        counter += 1
    return save_path


def colorize_images(target_path, colorizator, args):
    images = os.listdir(args.path)

    for image_name in images:
        file_path = os.path.join(args.path, image_name)

        if os.path.isdir(file_path):
            continue

        name, ext = os.path.splitext(image_name)
        if ext.lower() not in ['.jpg', '.jpeg', '.png']:
            continue

        if ext != '.png':
            image_name = name + '.png'

        print(f'Processing {file_path}')

        save_path = os.path.join(target_path, image_name)
        save_path = get_unique_save_path(save_path)
        colorize_single_image(file_path, save_path, colorizator, args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True)
    parser.add_argument("-o", "--output", default=None, help="Output location for colored images")
    parser.add_argument("-gen", "--generator", default='networks/generator.zip')
    parser.add_argument("-ext", "--extractor", default='networks/extractor.pth')
    parser.add_argument('-g', '--gpu', dest='gpu', action='store_true')
    parser.add_argument('-nd', '--no_denoise', dest='denoiser', action='store_false')
    parser.add_argument("-ds", "--denoiser_sigma", type=int, default=25)
    parser.add_argument("-s", "--size", type=int, default=576)
    parser.set_defaults(gpu=False)
    parser.set_defaults(denoiser=True)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    colorizer = MangaColorizator(device, args.generator, args.extractor)

    if args.output:
        colorization_path = args.output
    else:
        current_date = datetime.now().strftime('%Y-%m-%d')
        colorization_path = os.path.join('.', 'colored', current_date)

    if not os.path.exists(colorization_path):
        os.makedirs(colorization_path)

    if os.path.isdir(args.path):
        colorize_images(colorization_path, colorizer, args)
    elif os.path.isfile(args.path):
        split = os.path.splitext(args.path)

        if split[1].lower() in ('.jpg', '.png', '.jpeg'):
            new_image_name = os.path.basename(split[0]) + '_colorized.png'
            new_image_path = os.path.join(colorization_path, new_image_name)
            new_image_path = get_unique_save_path(new_image_path)
            colorize_single_image(args.path, new_image_path, colorizer, args)
        else:
            print('Wrong format')
    else:
        print('Wrong path')
