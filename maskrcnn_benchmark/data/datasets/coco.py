# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints


min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms
    
    
    def __getitem__(self, idx):
        
        for th in range(0):
            try:
                img, anno = super(COCODataset, self).__getitem__(idx)
                break
            except:
                #idx += 1
                from boxx import pred
                pred-"\n\n%sth times to try `img, anno = super(COCODataset, self).__getitem__(idx)`, idx=%s\n\n"%(th, idx)
                pass
        tryTimes = 0
        while 1:
            try:
                img, anno = super(COCODataset, self).__getitem__(idx + int(tryTimes//3))
                break
            except:
                tryTimes += 1
                pass

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target, idx
            
    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
    
    getitem = __getitem__
    def __getitem__(self, idx):
        from boxx import cf, randfloat
        getitem = type(self).getitem
        img, bboxs = getitem(self, idx)[:2]
        if cf.get('is_train') and cf.args.__dict__.get('mixup') and randfloat()>.5:
            from .deteMixUp import deteMixUp2img
            num = len(self)
            ind2 = random.randint(0,num-1)
            img2, bboxs2 = getitem(self, ind2)[:2]
            img, bboxs = deteMixUp2img(img, bboxs, img2, bboxs2)
#            print("\nMix Up on!\n")
        return img, bboxs, idx
    
import random
class DatasetMerge(torch.utils.data.dataset.Dataset):
    def __init__(self, datasetDic):
        self.datasetDic = datasetDic
        self.datasets, self.shares = zip(*datasetDic.items())
        summ = sum(self.shares)
        self.accumulate = []
        for share in self.shares:
            self.accumulate.append(sum(self.accumulate)+share/summ)
            
    def sampleDataset(self, seed=None):
        if seed is None:
            seed = random.random()
        for maxx, dataset in zip(self.accumulate, self.datasets):
            if seed <= maxx:
                return dataset
        raise Exception( "seed is %s, "%seed, self.accumulate)
        
    
    def __len__(self):
        return sum([len(d) for d in self.datasets])
    
    def __getitem__(self, ind):
        dataset = self.sampleDataset()
        lenn = len(dataset)
#        print(dataset)
        return dataset[int(lenn*random.random())]
    
    def __repr__(self):
        s = f'''DatasetMerge len is=%s
        
        {self.datasetDic}
        '''%len(self)
        return s
    __str__ = __repr__