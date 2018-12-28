# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms

    def __getitem__(self, idx):
        
        img, anno = super(COCODataset, self).__getitem__(idx)
        while 0:
            try:
                img, anno = super(COCODataset, self).__getitem__(idx)
                break
            except:
                #idx += 1
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

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
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