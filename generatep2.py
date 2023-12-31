#!/usr/bin/env python3

import os
import numpy
import random
import string
import cv2
import argparse
import captcha.image

symbol_to_code = {
    '#': 'hash',
    '\\': 'backslash',
    '!': 'exclamation',
    '/': 'forwardslash',
    '{': 'lbrace',
    '}': 'rbrace',
    '|': 'pipe',
    '"': 'doublequote',
    "'": 'singlequote',
    '?': 'question',
    '@': 'at',
    '`': 'backtick',
    ':':'colon'
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int)
    parser.add_argument('--height', help='Height of captcha image', type=int)
    parser.add_argument('--count', help='How many captchas to generate', type=int)
    parser.add_argument('--output-dir', help='Where to store the generated captchas', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    parser.add_argument('--fontpath', help='Path where the font style is saved', type=str)
    args = parser.parse_args()

    if args.width is None:
        print("Please specify the captcha image width")
        exit(1)

    if args.height is None:
        print("Please specify the captcha image height")
        exit(1)

    if args.count is None:
        print("Please specify the captcha count to generate")
        exit(1)

    if args.output_dir is None:
        print("Please specify the captcha output directory")
        exit(1)
        
    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    if args.fontpath is None:
        print("Please specify the path of the fonts file")
        exit(1)

    captcha_generator = captcha.image.ImageCaptcha(width=args.width, height=args.height, fonts=[args.fontpath])
    
    
    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()
    # captcha_symbols = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-=_+[]\{\}\|;':\",.<>/?^`~\!@\#\$%&*()"


    print("Generating captchas with symbol set {" + captcha_symbols + "}")

    if not os.path.exists(args.output_dir):
        print("Creating output directory " + args.output_dir)
        os.makedirs(args.output_dir)
        
    while True:
        # random length for the captchas between 1 and 6
        num_images = sum(1 for _ in os.scandir(args.output_dir) if _.is_file())
        
        if num_images < args.count:
            length = 5
            random_str = ''.join([random.choice(captcha_symbols) for j in range(length)])
            image1 = numpy.array(captcha_generator.generate_image(random_str))
            image_filename = ''.join([symbol_to_code[char] if char in symbol_to_code else char for char in random_str])
            image_path1 = os.path.join(args.output_dir, f"{image_filename}.png")
            cv2.imwrite(image_path1, image1)
        else: 
            break


            

if __name__ == '__main__':
    main()
