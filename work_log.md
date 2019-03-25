## 19.03.25 

nj-site:


export output_dir="output/syn1_";ylaunch --gpu=8 --cpu=16 --memory=100000 --  python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --data_root /home/yanglei/dataset/checkout-data/syn1/gan_coco_format --config-file configs/50_fpn_coco_format_1x.yaml MODEL.WEIGHT output/cocoPth/50_fpn_coco_pretrain_2x.pth  OUTPUT_DIR $output_dir ;ylaunch --gpu=1 --cpu=8 --memory=30000 -- python tools/test_net.py --data_root /home/yanglei/dataset/checkout-data/cocoFormatDirs/coco.valAsTrain --config-file configs/50_fpn_coco_format_1x.yaml  OUTPUT_DIR $output_dir


export output_dir="output/syn1_nsku";ylaunch --gpu=8 --cpu=16 --memory=100000 --  python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --data_root /home/yanglei/dataset/checkout-data/new_sku_data --data_root2 /home/yanglei/dataset/checkout-data/syn1/gan_coco_format --config-file configs/50_fpn_coco_format_1x.yaml MODEL.WEIGHT output/cocoPth/50_fpn_coco_pretrain_2x.pth DATALOADER.ASPECT_RATIO_GROUPING False OUTPUT_DIR $output_dir ;ylaunch --gpu=1 --cpu=8 --memory=30000 -- python tools/test_net.py --data_root /home/yanglei/dataset/checkout-data/cocoFormatDirs/coco.valAsTrain --config-file configs/50_fpn_coco_format_1x.yaml  OUTPUT_DIR $output_dir




export output_dir="output/syn1_mixup";ylaunch --gpu=8 --cpu=16 --memory=100000 --  python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --data_root /home/yanglei/dataset/checkout-data/syn1/gan_coco_format --config-file configs/50_fpn_coco_format_1x.yaml --mixup MODEL.WEIGHT output/cocoPth/50_fpn_coco_pretrain_2x.pth  OUTPUT_DIR $output_dir ;ylaunch --gpu=1 --cpu=8 --memory=30000 -- python tools/test_net.py --data_root /home/yanglei/dataset/checkout-data/cocoFormatDirs/coco.valAsTrain --config-file configs/50_fpn_coco_format_1x.yaml  OUTPUT_DIR $output_dir


export output_dir="output/syn1_nsku_mixup";ylaunch --gpu=8 --cpu=16 --memory=100000 --  python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --data_root /home/yanglei/dataset/checkout-data/new_sku_data --data_root2 /home/yanglei/dataset/checkout-data/syn1/gan_coco_format --config-file configs/50_fpn_coco_format_1x.yaml --mixup MODEL.WEIGHT output/cocoPth/50_fpn_coco_pretrain_2x.pth DATALOADER.ASPECT_RATIO_GROUPING False OUTPUT_DIR $output_dir ;ylaunch --gpu=1 --cpu=8 --memory=30000 -- python tools/test_net.py --data_root /home/yanglei/dataset/checkout-data/cocoFormatDirs/coco.valAsTrain --config-file configs/50_fpn_coco_format_1x.yaml  OUTPUT_DIR $output_dir


-----

ylaunch --gpu=8 --cpu=16 --memory=100000 -- sleep 300;export output_dir="output/syn1_mixup";rlaunch --gpu=8 --cpu=16 --memory=100000 --  python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --data_root /home/yanglei/dataset/checkout-data/syn1/gan_coco_format --config-file configs/50_fpn_coco_format_1x.yaml --mixup MODEL.WEIGHT output/cocoPth/50_fpn_coco_pretrain_2x.pth  OUTPUT_DIR $output_dir ;rlaunch --gpu=1 --cpu=8 --memory=30000 -- python tools/test_net.py --data_root /home/yanglei/dataset/checkout-data/cocoFormatDirs/coco.valAsTrain --config-file configs/50_fpn_coco_format_1x.yaml  OUTPUT_DIR $output_dir

ylaunch --gpu=8 --cpu=16 --memory=100000 -- sleep 300;export output_dir="output/syn1_nsku_mixup";rlaunch --gpu=8 --cpu=16 --memory=100000 --  python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --data_root /home/yanglei/dataset/checkout-data/new_sku_data --data_root2 /home/yanglei/dataset/checkout-data/syn1/gan_coco_format --config-file configs/50_fpn_coco_format_1x.yaml --mixup MODEL.WEIGHT output/cocoPth/50_fpn_coco_pretrain_2x.pth  OUTPUT_DIR $output_dir ;rlaunch --gpu=1 --cpu=8 --memory=30000 -- python tools/test_net.py --data_root /home/yanglei/dataset/checkout-data/cocoFormatDirs/coco.valAsTrain --config-file configs/50_fpn_coco_format_1x.yaml  OUTPUT_DIR $output_dir