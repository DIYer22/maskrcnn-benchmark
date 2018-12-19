# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2
from boxx import *
addPathToSys(__file__, '..')
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import time

#def main():
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
#        default="../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
        default=os.path.abspath(pathjoin(__file__, "../../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",)),
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--imgp",
        default="",
        help="path to test img",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
#        default=224,
        default=800,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    start_time = time.time()
    if args.imgp:
        img = imread(args.imgp)
    else:
        img = sda.astronaut()
        img = imread("/home/dl/Downloads/光阴的故事.jpg")
    
    composite = coco_demo.run_on_opencv_image(img[...,[2,1,0]])[...,[2,1,0]]
    show-composite
    bboxList = coco_demo.getBboxList(img)
    print("Time: {:.2f} s / img".format(time.time() - start_time))
    
    


#    main()
    #%run demo/webcam.py --imgp /home/dl/junk/vis/0.jpg --config-file configs/101_fpn_coco_format_1x.yaml  TEST.IMS_PER_BATCH 1  MODEL.WEIGHT output/mix_11/model_final.pth
