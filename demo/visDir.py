#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: DIYer22@github.com
@mail: ylxx@live.com
Created on Mon Dec 17 23:10:54 2018

rlaunch --gpu=1 --cpu=8 --memory=30000 -- python demo/visDir.py --config-file  configs/101_fpn_coco_format_1x.yaml  --dir /unsullied/sharefs/yanglei/share/checkout-data/check_image/analysis/allWrong --confidence-threshold 0.84 --pth output/testAsTrain/model_final.pth
"""
from boxx import *
import argparse
import cv2
from boxx import *
from boxx import  np, os, npa, dicto, addPathToSys, dirname, filename, execmd, tmpboxx, pathjoin, LogLoopTime, glob, imread
addPathToSys(__file__, '..')
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo



addPathToSys('/home/dl/research/auto_synthesis_data/a')
if cloud:
    addPathToSys('/unsullied/sharefs/yanglei/share/checkout-data/auto_synthesis/a')
from visMask import vis_one_image, mask2rle

def visBboxList(rgb, bboxList, path='/tmp/visCanvas/tmp.pdf', classNames=None, box_alpha=0.8, 
            show_class=True, thresh=0.1, show_mask=None, pltshow=False):
    
    if  classNames is None:
        classNames = {i:"%d-class"%i for i in range(1111)}
    classn = len(classNames)
    dataset= dicto(classes=classNames)
    cls_segms = [[] for _ in range(classn)]
    cls_boxes = [np.zeros((0,5), np.float32) for _ in range(classn)]
    
    #cls_segms = None
    #cls_boxes = [np.zeros((0,5), np.float32) for _ in range(classn)]
    bboxnps = bboxList.bbox
    extraFields = {k:v.cpu().numpy() for k,v in bboxList.extra_fields.items()}
    if show_mask is None:
        if 'mask' in extraFields:
            show_mask = True
    
    for ind, bboxnp in enumerate(bboxnps):
#        g()
        other = dicto({k:v[ind] for k,v in extraFields.items()})
        if other.scores < thresh:
            continue
        c = other.labels
        if show_mask:
            rle = mask2rle(other.mask)
            cls_segms[c].append(rle)
        if bboxList.mode == 'xyxy':
            cls_boxes[c] = np.append(cls_boxes[c], [list(bboxnp)+[other.scores]], 0)
    
    cls_keyps = None
    if not show_mask:
        cls_segms = None
#    g()
    outputDir, name = dirname(path), filename(path)
    vis_one_image(
        rgb,  # BGR -> RGB for visualization
        name,
        outputDir,
        cls_boxes,
        cls_segms,
        cls_keyps,
        dataset=dataset,
        box_alpha=0.8,
        show_class=True,
        thresh=thresh,
        kp_thresh=2,
        pltshow=pltshow,
    )
classNames = dicto(enumerate(['__background__', '上好佳荷兰豆55g', '菜园小饼80g', '上好佳鲜虾片40g', '上好佳蟹味逸族40g', '妙脆角魔力炭烧味65g', '盼盼烧烤牛排味块105g', '上好佳鲜虾条40g', '上好佳洋葱圈40g', '上好佳日式鱼果海苔味50g', '奇多日式牛排味90g', '奇多美式火鸡味90g', '上好佳粟米条草莓味40g', '甘源蟹黄味瓜子仁75g', '惠宜开心果140g', '惠宜咸味花生350g', '惠宜腰果160g', '惠宜枸杞100g', '惠宜地瓜干228g', '惠宜泰国芒果干80g', '惠宜黄桃果干75g', '惠宜柠檬片65g', '新疆和田滩枣454g', '惠宜香菇100g', '惠宜桂圆干500g', '惠宜茶树菇200g', '豪雄单片黑木耳150g', '惠宜煮花生454g', '惠宜黄花菜100g', '洽洽凉茶瓜子150g', '洽洽奶香味瓜子150g', '车仔茶包绿茶50g', '车仔茶包红茶50g', '优乐美香芋味80g', '优乐美红豆奶茶65g', '欢泥冲调土豆粥25g', '江中猴姑早餐米稀40g', '永和豆浆甜豆浆粉210g', '立顿柠檬风味茶180g', '桂格多种莓果麦片40g', '荣怡谷麦加黑米味30g', '荣怡谷麦加红豆味30g', '今野香辣牛肉面112g', '今野老坛酸菜牛肉面118g', '今野红烧牛肉面114g', '合味道海鲜风味84g', '康师傅白胡椒肉骨面76g', '康师傅香辣牛肉面105g', '康师傅香辣蒜味排骨面108g', '康师傅藤椒牛肉面82g', '华丰鸡肉三鲜伊面87g', '康师傅黑胡椒牛排面104g', '五谷道场红烧牛肉面100g', '康师傅老坛酸菜牛肉面114g', 'Aji泡芙饼干芒果菠萝味60g', '庆联蓝莓味夹心饼63g', '庆联凤梨味夹心饼63g', '庆联草莓味夹心饼63g', '嘉顿威化饼干草莓味50g', '嘉顿威化饼干柠檬味50g', '爱时乐香草牛奶味50g', '爱时乐巧克力味50g', '百力滋海苔味60g', '百力滋草莓牛奶味45g', '雀巢脆脆鲨80g', '纳宝帝巧克力味威化58g', '桂力地中海风味面包条50g', '康师傅妙芙巧克力味48g', '爱乡亲唱片面包90g', '达利园派草莓味单个装*', 'mini奥利奥55g', '农夫山泉矿泉水550ml', '怡宝矿泉水555ml', '可口可乐零度500ml', '可口可乐500ml', '百事可乐600ml', '芬达苹果味500ml', '芬达橙味500ml', '雪碧500ml', '喜力啤酒500ml', '百威啤酒600ml', '百事可乐330ml', '可口可乐330ml', '王老吉310ml', '茶派柚子绿茶500ml', '茶派玫瑰荔枝红茶500ml', '康师傅冰红茶250ml', '加多宝250ml', 'RIO果酒水蜜桃味275ml', 'RIO果酒蓝玫瑰威士忌味275ml', '牛栏山二锅头100ml', '哈尔滨啤酒330ml', '青岛啤酒330ml', '雪花啤酒330ml', '哈尔滨啤酒500ml', 'KELER啤酒500ml', '百威啤酒500ml', 'QQ星全聪奶125ml', 'QQ星均膳奶125ml', '娃哈哈AD钙奶220g', '活力宝动力源105ml', '旺仔牛奶复原乳250ml', '伊利纯牛奶250ml', '维他低糖原味豆奶250ml', '百怡花生牛奶250ml', '惠宜原味豆奶250ml', '伊利优酸乳250ml', '伊利早餐奶250ml', '达利园桂圆莲子360g', '银鹭冰糖百合银耳280g', '喜多多什锦椰果567g', '都乐菠萝块567g', '都乐菠萝块234g', '银鹭薏仁红豆粥280g', '银鹭莲子玉米粥280g', '银鹭紫薯紫米粥280g', '银鹭椰奶燕麦粥280g', '银鹭黑糖桂圆280g', '梅林午餐肉340g', '珠江桥牌豆豉鱼150g', '古龙原味黄花鱼120g', '雄鸡标椰浆140ml', '德芙芒果酸奶巧克力42g', '德芙摩卡巴旦木巧克力43g', '德芙百香果白巧克力42g', 'MM花生牛奶巧克力豆40g', 'MM牛奶巧克力豆40g', '好时牛奶巧克力40g', '好时曲奇奶香白巧克力40g', '脆香米海苔白巧克力24g', '脆香米奶香白巧克力24g', '士力架花生夹心巧克力51g', '士力架燕麦花生夹心巧克力40g', '士力架辣花生夹心巧克力40g', '炫迈果味浪薄荷味37g', '炫迈果味浪柠檬味37g', '炫迈薄荷味21g', '炫迈葡萄味21g', '炫迈西瓜味21g', '炫迈葡萄味50g', '绿箭无糖薄荷糖茉莉花茶味34g', '绿箭5片装15g', '比巴卜棉花泡泡糖可乐味11g', '比巴卜棉花泡泡堂葡萄味11g', '星爆缤纷原果味25g', '阿尔卑斯焦香牛奶味硬糖45g', '阿尔卑斯牛奶软糖黄桃酸奶味47g', '阿尔卑斯牛奶软糖蓝莓酸奶味47g', '王老吉润喉糖28g', '伊利牛奶片蓝莓味32g', '熊博士口嚼糖草莓牛奶味52g', '彩虹糖原果味45g', '宝鼎天鱼陈酿米醋245ml', '恒顺香醋340ml', '太太乐鸡精200g', '家乐香菇鸡茸汤料41g', '惠宜辣椒粉15g', '惠宜生姜粉15g', '味好美椒盐20g', '海星加碘精制盐400g', '恒顺料酒500ml', '东古味极鲜酱油150ml', '东古一品鲜酱油150ml', '欣和六月鲜酱油160ml', '李施德林零度漱口水80ml', '舒肤佳纯白清香沐浴露100ml', '美涛定型啫喱水60ml', '清扬男士洗发露活力运动薄荷型50ml', '蓝月亮风清白兰洗衣液80g', '高露洁亮白小苏打180g', '高露洁冰爽180g', '舒亮皓齿白80g', '云南白药牙膏45g', '舒克宝贝儿童牙刷', '清风原木纯品金装100x3', '洁柔face150x3', '斑布100x3', '维达婴儿150x3', '相印小黄人150x3', '清风原木纯品黑耀150x3', '洁云绒触感130x3', '舒洁萌印花120x2', '相印红悦130x3', '得宝苹果木味90x4', '清风新韧纯品130x3', '金鱼竹浆绿135x3', '清风原木纯品150x2', '洁柔face130x3', '维达立体美110x3', '洁柔CS单包*', '相印小黄人单包*', '清风原色单包*', '相印茶语单包*', '清风质感纯品单包*', '米奇1928笔记本', '广博固体胶15g', '票据文件袋', '晨光蜗牛改正带', '鸿泰液体胶50g', '马培德自粘性标签', '东亚记号笔']))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default=os.path.abspath(pathjoin(__file__, "../../configs/101_fpn_coco_format_1x.yaml",)),
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.78,
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
        "--dir",
        help="which images dir",
        default="/home/dl/junk/mix_bad",
    )
    parser.add_argument(
        "--pth",
        help="pth file path",
        default="",
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
    cfg.TEST.IMS_PER_BATCH = 1
    cfg.MODEL.WEIGHT = os.path.abspath(pathjoin(__file__, "../../output/mix_11/model_0002500.pth"))
#    cfg.MODEL.WEIGHT = os.path.abspath(pathjoin(__file__, "../../output/mix_11/model_final.pth"))
#    cfg.MODEL.WEIGHT = "/home/dl/junk/output/single/model_0052500.pth"
#    cfg.MODEL.WEIGHT = "/home/dl/junk/output/single/model_final.pth"
#    cfg.freeze()
    
    if args.pth:
        cfg.MODEL.WEIGHT = args.pth
    
    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=args.min_image_size,
    )

    imgps = sorted(glob(pathjoin(args.dir,"*.jpg")))
    loopLog = LogLoopTime(imgps)
    for imgp in imgps[:]:
        img = imread(imgp)
#        composite = coco_demo.run_on_opencv_image(img[...,[2,1,0]])[...,[2,1,0]]
#        show-composite
        bboxList = coco_demo.getBboxList(img)
        visBboxList(img, bboxList, imgp, pltshow=not cloud , thresh=args.confidence_threshold, classNames=classNames)
        execmd("google-chrome {}".format(imgp.replace('.jpg','.pdf')))
        loopLog(imgp)
