#!/usr/bin/env python3
import os
import time
import datetime
import cv2
import numpy as np
import uuid
import json

import functools
import logging
import collections

try:
    from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageFilter
except ImportError:
    import Image

import pytesseract

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@functools.lru_cache(maxsize=100)
def get_predictor(checkpoint_path):
    logger.info('loading model')
    import tensorflow as tf
    import model
    from icdar import restore_rectangle
    import lanms
    from eval import resize_image, sort_poly, detect

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    f_score, f_geometry = model.model(input_images, is_training=False)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
    model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
    logger.info('Restore from {}'.format(model_path))
    saver.restore(sess, model_path)

    def predictor(img):
        start_time = time.time()
        rtparams = collections.OrderedDict()
        rtparams['start_time'] = datetime.datetime.now().isoformat()
        rtparams['image_size'] = '{}x{}'.format(img.shape[1], img.shape[0])
        timer = collections.OrderedDict([
            ('net', 0),
            ('restore', 0),
            ('nms', 0)
        ])

        im_resized, (ratio_h, ratio_w) = resize_image(img)
        rtparams['working_size'] = '{}x{}'.format(
            im_resized.shape[1], im_resized.shape[0])
        start = time.time()
        score, geometry = sess.run(
            [f_score, f_geometry],
            feed_dict={input_images: [im_resized[:,:,::-1]]})
        timer['net'] = time.time() - start

        boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
        logger.info('net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
            timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

        if boxes is not None:
            scores = boxes[:,8].reshape(-1)
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        duration = time.time() - start_time
        timer['overall'] = duration
        logger.info('[timing] {}'.format(duration))

        text_lines = []
        if boxes is not None:
            text_lines = []
            for box, score in zip(boxes, scores):
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                    continue
                tl = collections.OrderedDict(zip(
                    ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                    map(float, box.flatten())))
                tl['score'] = float(score)
                text_lines.append(tl)

        t_data = save_text(img.copy(), text_lines)

        ret = {
            'text_lines': text_lines,
            'rtparams': rtparams,
            'timing': timer,
            't_data': t_data
        }

        return ret

    return predictor


from flask import Flask, request, render_template
import argparse

class Config:
    SAVE_DIR = 'static/results'


config = Config()
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', session_id='dummy_session_id')

def draw_illu(illu, rst):
    for t in rst['text_lines']:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype='int32')
        d = d.reshape(-1, 2)
        cv2.polylines(illu, [d], isClosed=True, color=(255, 255, 0))
    return illu

def save_text(img_or, rst):
    t_data = []
    for t in rst:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype='int32')
        d = d.reshape(-1, 2)
        #print('--------------------------------------\n')

        d = d.flatten()
        #print(len(d), d)
        #print('---------------------------------------\n')
        width, height = img_or.shape[:2]
        #print('w', width, height)
        x1 = d[0]
        y1 = d[1]
        x2 = d[4]
        y2 = d[5]
        #print(x1, y1, '/', x2, y2)

        #t_path = './tmp/' + str(uuid.uuid1()) + '.png'
        if y1-25 >= 0: y1 = y1-25
        if y2+25 <= height : y2 = y2+25
        if x1-25 >= 0 : x1 = x1-25
        if x2+25 <= width : x2 = x2+25
        crop_img = img_or[y1:y2, x1:x2]
        width, height = crop_img.shape[:2]
        if width > 4 and height > 4:
            #cv2.imwrite(t_path, crop_img)
            text = text_conf(crop_img)
            t_data.append(text)
    text = text_conf(img_or)
    print(text)
    return t_data


def save_result(img, rst):
    session_id = str(uuid.uuid1())
    dirpath = os.path.join(config.SAVE_DIR, session_id)
    os.makedirs(dirpath)

    output_path = os.path.join(dirpath, 'input.png')
    cv2.imwrite(output_path, img)
    output_path = os.path.join(dirpath, 'output.png')

    cv2.imwrite(output_path, draw_illu(img.copy(), rst))
    output_path = os.path.join(dirpath, 'result.json')
    with open(output_path, 'w') as f:
        json.dump(rst, f)
    rst['session_id'] = session_id
    #rst['t_data'] = t_data
    return rst

def recognize(f_name, img):
    config = '--psm 6 -c tessedit_char_whitelist=abcdeghkmnopqrsuvwxyz0123456789'
    # data_str = pytesseract.image_to_data(img, lang='eng', config=config, output_type='string', pandas_config=None)
    # print(data_str)

    data_d = pytesseract.image_to_data(img, lang='eng', config=config, output_type='dict')

    n_boxes = len(data_d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (data_d['left'][i], data_d['top'][i], data_d['width'][i], data_d['height'][i])
        #print(i, data_d['text'][i], end=' ')
        cv2.rectangle(img, (x, y), (x + w, y + h), (225, 0, 0), 3)

    #print('\nNew method')
    cv2.imshow('1'+f_name, img)

    #img = cv2.imread(f_name)
    h, w, _ = img.shape  # assumes color image
    # run tesseract, returning the bounding boxes
    boxes = pytesseract.image_to_boxes(img, lang='eng', config=config)  # also include any config options you use
    # draw the bounding boxes on the image
    #(boxes)
    for b in boxes.splitlines():
        b = b.split(' ')
        img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (225, 0, 0), 3)

    # show annotated image and wait for keypress
    cv2.imshow('2'+f_name, img)
    #cv2.waitKey(0)

def text_conf(img):
    #text = pytesseract.image_to_string(img, lang='eng', config='--psm 7')
    #print(text)
    text = pytesseract.image_to_string(img, lang='rus', config='--psm 10')
    text_en = pytesseract.image_to_string(img, lang='eng', config='--psm 10 -c tessedit_char_whitelist="abcdefghijklmnopqrstuvwxyz0123456789"')
    text_en2 = pytesseract.image_to_string(img, lang='eng', config='--psm 6')

    if len(text_en2) > len(text_en) : text_en = text_en2
    if len(text_en) > len(text) : text = text_en
    return text

checkpoint_path = './east_icdar2015_resnet_v1_50_rbox'

@app.route('/', methods=['POST'])
def index_post():
    global predictor
    import io
    bio = io.BytesIO()
    request.files['image'].save(bio)
    img = cv2.imdecode(np.frombuffer(bio.getvalue(), dtype='uint8'), 1)
    rst = get_predictor(checkpoint_path)(img)

    rst = save_result(img, rst)
    return render_template('index.html', session_id=rst['session_id'])


def main():
    global checkpoint_path
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=8769, type=int)
    parser.add_argument('--checkpoint_path', default=checkpoint_path)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(args.checkpoint_path):
        raise RuntimeError(
            'Checkpoint `{}` not found'.format(args.checkpoint_path))

    #app.debug = False
    app.run('127.0.0.1', args.port)


if __name__ == '__main__':
    main()

