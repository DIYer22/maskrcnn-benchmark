# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T


def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    transform = T.Compose(
        [
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    from boxx import cf
    if is_train and cf.args.task == "rpc":
        transform = T.Compose(
            [
                T.Resize(min_size, max_size),
                T.RandomHorizontalFlip(flip_prob),
                T.RandomVerticalFlip(flip_prob),
                T.RandomRotate(is_train),
                T.ToTensor(),
                normalize_transform,
            ]
        )
#    adjustBrightness = .5
    adjustBrightness = None
    if adjustBrightness:
        from boxx import pred
#        import boxx.g
        transforms = transform.transforms
        transform = T.Compose(transforms[:-1] + [lambda img,t:(img*adjustBrightness, t)] + transforms[-1:])
        pred-"\n\nNotice: adjustBrightness is %s\n\n%s"%(adjustBrightness,transform)
              
#    adjustBrightness = 0.825
#    adjustBrightness = 0.65
    adjustBrightness = None
    if adjustBrightness:
        from boxx import np, uint8, pred
        import skimage.color as col
        from PIL import Image
        def adjustBrightnessInHsv(img, vRate):
            hsv = col.rgb2hsv(img)
            hsv[...,2] *= vRate
            newimg = col.hsv2rgb(hsv)
            return uint8(newimg)
        
        def adjustBrightnessPil(pil, target):
            new = adjustBrightnessInHsv(np.array(pil), adjustBrightness)
            newpil = Image.fromarray(new)
            return newpil, target
        pred-"\n\nNotice: adjustBrightness is %s\n\n%s"%(adjustBrightness,transform)
        transforms = transform.transforms
        transform = T.Compose(transforms[:1] + [adjustBrightnessPil] + transforms[1:])
    from boxx import cf
    cf.is_train = is_train
    return transform
