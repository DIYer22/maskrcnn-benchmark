#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: DIYer22@github
@mail: ylxx@live.com
Created on Thu Feb 14 18:23:24 2019
"""

from boxx import distnorm, Vector, npa
import random
import torch


def drawRectangle(img, bboxs):
    from boxx import uint8, torgb, tree, show
    img = uint8(torgb(npa-img))[...,[2,1,0]]
    img = img.copy()
    import cv2
    tree-img
    for bbox in bboxs.bbox:
        cv2.rectangle(img, tuple(npa(bbox[:2])), tuple(npa-bbox[2:]),(255,0,0), 4)
    show-img
    return img

def drawRectangleDataset(dataset):
    num = len(dataset)
    ind2 = random.randint(0,num-1)
    img2, bboxs2 = dataset[ind2][:2]
    return drawRectangle(img2, bboxs2)
    

def deteMixUp2img(img, bboxs, img2, bboxs2):
    alpha = .5
    alpha = distnorm(.5, maxbias=.1)
    
    xy = Vector(img.shape[::-1][:2])
        
#    drawRectangle(img2, bboxs2)
    
    xy2 = Vector(img2.shape[::-1][:2])
#    tree-[img,img2]
    resizeRatio = (xy2/xy).max()
    size = (xy2 / resizeRatio).astype(int)
    
    bias = ((xy-size)* random.random()).astype(int)
    
#    print(bias, size)
    assert all(xy>=size)
    
    resized = torch.nn.functional.upsample(img2[None], size=tuple(size[::-1]), mode='bilinear')[0]
#    tree-[resized, img2]
    
    img[..., bias.y: bias.y+size.y, bias.x: bias.x+size.x] *= alpha
    img[..., bias.y: bias.y+size.y, bias.x: bias.x+size.x] += (1-alpha) * resized
    
    if 'masks' in bboxs.extra_fields:
        bboxs.extra_fields.pop('masks')
        
    bboxs_resized = bboxs2.resize(size)
    bbox_matrix = bboxs_resized.bbox
    bbox_matrix[:,0::2] += bias.x
    bbox_matrix[:,1::2] += bias.y
    
    bboxs.bbox = torch.cat([bboxs.bbox, bbox_matrix])
    
    bboxs.extra_fields['labels'] = torch.cat([bboxs.extra_fields['labels'], bboxs_resized.extra_fields['labels']])
    return img, bboxs

if __name__ == "__main__":
    from boxx import *
    from boxx import sda, resize, ignoreWarning
    ignoreWarning()
    
    dataset = data_loader.dataset
    batch = dataset[5]
    img, bboxs, ind = batch
    
    num = len(dataset)
    ind2 = random.randint(0,num-1)
    img2, bboxs2, ind2 = dataset[ind2]
    
    img, bboxs = deteMixUp2img(img, bboxs, img2, bboxs2, )

    #drawRectangle(img, bboxs_resized)
    drawed = drawRectangle(img, bboxs)
    
    #show([img,img2], torgb, lambda x:x[...,[2,1,0]])

    
    
    
