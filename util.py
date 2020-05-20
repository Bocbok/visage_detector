import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from PIL import Image, ImageDraw
import itertools
from multiprocessing import Pool
from itertools import combinations
import tensorflow.keras.backend as K
import tensorflow as tf

REPRESENTATION_OF_1_IN_LOSS = 0.5

# Returns the IoU between 2 boxes
def IoU(box1, box2):
    _, i1, j1, h1, l1 = box1
    _, i2, j2, h2, l2 = box2
    
    l_intersection = min(j1 + l1, j2 + l2) - max(j1, j2)
    h_intersection = min(i1 + h1, i2 + h2) - max(i1, i2) 
    # No overlap
    if l_intersection <= 0 or h_intersection <=0:
        return 0
    
    I = l_intersection * h_intersection
    
    U = l1 * h1 + l2 * h2 - I
    
    return I / U


def delete_overlaping_boxes(df):
    to_del = []

    for k in df.k.unique():
        frame = df[df.k == k].sort_values('score', ascending=False)

        for pair in combinations(range(len(frame)), r=2):
            b1 = frame.iloc[pair[0]]
            b2 = frame.iloc[pair[1]]

            I = IoU((_, b1.i, b1.j, b1.h, b1.w), (_, b2.i, b2.j, b2.h, b2.w))
            if I > 0.4:
                to_del.append(frame.iloc[pair[1]].name)
                
    return df.drop(np.unique(to_del))


def sliding_window(image, stride, wss):
    ret = []
    
    for ws in wss:
        sh = int(ws[0]*stride/100)
        sw = int(ws[1]*stride/100)
        for i in range(0, image.h-ws[0], sh):
            for j in range(0, image.w-ws[1], sw):
                if (i+ws[0] < image.h) and (j+ws[1] < image.w):
                    ret.append({'k':image.k, 
                                    'i':i, 
                                    'j':j, 
                                    'h':ws[0], 
                                    'w':ws[1]})
        for j in range(0, image.w-ws[1], sw):
            if (j+ws[1] < image.w):
                ret.append({'k':image.k, 
                            'i':image.h-ws[0], 
                            'j':j, 
                            'h':ws[0], 
                            'w':ws[1]})
        for i in range(0, image.w-ws[0], sh):
            if (i+ws[0] < image.h):
                ret.append({'k':image.k, 
                            'i':i, 
                            'j':image.w-ws[1], 
                            'h':ws[0], 
                            'w':ws[1]})
        
    return pd.DataFrame(ret)


def get_matching_boxes(true_box, box_list):
    print(len(box_list))
    temp = box_list.apply(lambda x: 1 if IoU(true_box, x) > 0.55 else 0, axis=1)
    return sum(temp)


def get_params_perf(params, imgs, labels):
    stride, ws = params

    box_list = []
    ret = []
        
    for idx, img in imgs.iterrows():
        boxes = sliding_window(img, stride, ws)
        box_list.append(boxes)
        ret.append(labels[labels.k == idx+1].apply(lambda x: get_matching_boxes(x, boxes), axis=1))


    box_list = pd.concat(box_list)
    ret = pd.concat(ret[:-1])
    
    stats = {'stride':stride, 
            'ws':ws, 
            'acc':sum(ret > 0)/len(ret),
            'mean':ret[ret != 0].mean(),
            'num_im':len(box_list)}
    
    return (ret, box_list, stats)


def crop_resize_save_img(im, x, box_labels, folder_size):
    iou = 0
    sub_folder, size = folder_size
    for _, box in box_labels.iterrows():
        n_iou = IoU(box, x)
        iou = round(n_iou, 2) if n_iou > iou else iou
    img = im.crop((x.j, x.i, x.j+x.w, x.i+x.h))
    img = img.resize((size,size))
    if iou > 0.2:
        img.save(f'train3{size}x{size}/{sub_folder}/1/{x.k}_{x.i}_{x.j}_{x.h}_{x.w}_{iou}.jpg')
    else:
        img.save(f'train3{size}x{size}/{sub_folder}/0/{x.k}_{x.i}_{x.j}_{x.h}_{x.w}_{iou}.jpg')

# im = {k, path, h, w}
def create_img_dataset(img, stride, window_size, box_labels, folder_size):
    _, img = img
    boxes = sliding_window(img, stride, window_size)
    with Image.open(img.path) as im:
        boxes.apply(lambda x: crop_resize_save_img(im, x, box_labels[box_labels.k == x.k], folder_size), axis=1)


# Determine the weight to apply to classes for imbalanced dataset
def class_weight(nb_1, nb_0, representation_of_1_in_loss=REPRESENTATION_OF_1_IN_LOSS):
    weight_0 = (1 - representation_of_1_in_loss) * (nb_0 + nb_1) / nb_0
    weight_1 = representation_of_1_in_loss * (nb_0 + nb_1) / nb_1
    return weight_0, weight_1

# Weighted binary cross entropy
def wbce( y_true, y_pred, sample_weight=None, weight1=1, weight0=1 ):
    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    logloss = -(y_true * K.log(y_pred) * weight1 + (1 - y_true) * K.log(1 - y_pred) * weight0 )
    return K.mean( logloss, axis=(0,1))

# Compute the best f1 score, precision, recall, prediction threshold
def f1(y_true, y_pred):
    df = pd.DataFrame(y_true, columns=['y_true'])
    df['y_pred'] = y_pred
    cleaner = df.duplicated('y_pred', keep='last')
    df = df.sort_values('y_pred', ascending=False)
    df['vp'] = df.y_true.cumsum()
    df['fp'] = (df.y_true == 0).cumsum()
    df['fn'] = sum(df.y_true) - df['vp']
    df['prec'] = df.vp / (df.vp + df.fp)
    df['rec'] = df.vp / (df.vp + df.fn)
    df = df.loc[~cleaner]
    df['f1'] = 2* (df.prec * df.rec) / (df.prec + df.rec)
    best = df['f1'].idxmax()
    return df.loc[best]

# Compute the best f2 score, precision, recall, prediction threshold
def f2(y_true, y_pred):
    df = pd.DataFrame(y_true, columns=['y_true'])
    df['y_pred'] = y_pred
    cleaner = df.duplicated('y_pred', keep='last')
    df = df.sort_values('y_pred', ascending=False)
    df['vp'] = df.y_true.cumsum()
    df['fp'] = (df.y_true == 0).cumsum()
    df['fn'] = sum(df.y_true) - df['vp']
    df['prec'] = df.vp / (df.vp + df.fp)
    df['rec'] = df.vp / (df.vp + df.fn)
    df = df.loc[~cleaner]
    df['f2'] = 5* (df.prec * df.rec) / (4*df.prec + df.rec)
    best = df['f2'].idxmax()
    return df.loc[best]