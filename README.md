# EFFOcc
EFFOcc: Learning Efficient Occupancy Networks from Minimal Labels for Autonomous Driving
(Old title: EFFOcc: A Minimal Baseline for EFficient Fusion-based 3D Occupancy Network)

## Demo videos
The project demo videos.

https://github.com/user-attachments/assets/55fe0d54-a7bf-4e80-bb96-16e718116a55


The lidar-camera occupancy prediction video of Occ3D-nuScenes dataset. 

https://github.com/synsin0/EFFOcc/assets/37300008/6ab8238f-1d7f-4e4b-b4de-daff6a99ba41


## Abstract
3D occupancy prediction (3DOcc) is a rapidly rising and challenging perception task in the field of autonomous driving. Existing 3D occupancy networks (OccNets) are both computationally heavy and label-hungry. In terms of model complexity, OccNets are commonly composed of heavy Conv3D modules or transformers at the voxel level. Moreover, OccNets are supervised with expensive large-scale dense voxel labels. Model and data inefficiencies, caused by excessive network parameters and label annotation requirements, severely hinder the onboard deployment of OccNets. This paper proposes an EFFicient Occupancy learning framework, EFFOcc, that targets minimal network complexity and label requirements while achieving state-of-the-art accuracy. We first propose an efficient fusion-based OccNet that only uses simple 2D operators and improves accuracy to the state-of-the-art on three large-scale benchmarks: Occ3D-nuScenes, Occ3D-Waymo, and OpenOccupancy-nuScenes. On the Occ3D-nuScenes benchmark, the fusion-based model with ResNet-18 as the image backbone has 21.35M parameters and achieves 51.49 in terms of mean Intersection over Union (mIoU). Furthermore, we propose a multi-stage occupancy-oriented distillation to efficiently transfer knowledge to vision-only OccNet. Extensive experiments on occupancy benchmarks show state-of-the-art precision for both fusion-based and vision-based OccNets. For the demonstration of learning with limited labels, we achieve 94.38\% of the performance (mIoU = 28.38) of a 100\% labeled vision OccNet (mIoU = 30.07) using the same OccNet trained with only 40\% labeled sequences and distillation from the fusion-based OccNet.


## Data Setup
We follow the setups of [BEVDet](https://github.com/HuangJunJie2017/BEVDet) for data preprocessing of nuScenes dataset.

## Models
Exps on Occ3D-nuScenes:

| Settings | Fusion Model | FlashOcc | Distilled Model | 
| ----- | ----- | -------- | -------- |
| 100% pretrained |   Fusion-R18 [CFG](configs/effocc_fusion_r18_data_scales/flashocc_fusion_r18_base_100%_seqs.py) [CKPT](https://cloud.tsinghua.edu.cn/f/0dd01c193f46496a88b8/?dl=1)      |     FlashOcc-R50 [CFG](configs/flashocc/flashocc-r50_from_scratch.py) [CKPT](https://cloud.tsinghua.edu.cn/f/f6a1bca862674cd49c1e/?dl=1)     |    DistillOcc-R50 [CFG](configs/effocc_distillocc/fgbg_distill_flashocc-r50_100%_labeled_effocc_r18_100%.py) [CKPT](https://cloud.tsinghua.edu.cn/f/19d2ada1c4c241a8bc2d/?dl=1)      |      
| 100% from scratch |   Fusion-R18 [CFG](configs/effocc_fusion_r18_data_scales/flashocc_fusion_r18_base_100%_seqs.py) [CKPT](https://cloud.tsinghua.edu.cn/f/0dd01c193f46496a88b8/?dl=1)      |     FlashOcc-R50 [CFG](configs/flashocc/flashocc-r50.py) [CKPT](https://cloud.tsinghua.edu.cn/f/366c0876419c45c5a9a8/?dl=1)     |    DistillOcc-R50 [CFG](configs/effocc_distillocc/fgbg_distill_flashocc-r50_100%_labeled_effocc_100%_scratch.py) [CKPT](https://cloud.tsinghua.edu.cn/f/81656568aa164e8b869d/?dl=1)      |    
| 5% |   Fusion-R18 [CFG](configs/effocc_fusion_r18_data_scales/flashocc_fusion_r18_base_5%_seqs.py) [CKPT](https://cloud.tsinghua.edu.cn/f/7c6b0a71a22c42acaf9c/?dl=1)      |     FlashOcc-R50 [CFG](configs/flashocc/flashocc-r50_35seqs_scratch.py) [CKPT](https://cloud.tsinghua.edu.cn/f/f5db3db09e974a9eac88/?dl=1)     |    DistillOcc-R50 [CFG](configs/effocc_distillocc/fgbg_distill_flashocc-r50_5%_labeled_effocc_5%_scratch.py) [CKPT](https://cloud.tsinghua.edu.cn/f/a4a801b4b23c49a5946c/?dl=1)      |   
| 10% |   Fusion-R18 [CFG](configs/effocc_fusion_r18_data_scales/flashocc_fusion_r18_base_10%_seqs.py) [CKPT](https://cloud.tsinghua.edu.cn/f/ce2b9371aba64695969a/?dl=1)      |     FlashOcc-R50 [CFG](configs/flashocc/flashocc-r50_70seqs_scratch.py) [CKPT](https://cloud.tsinghua.edu.cn/f/30a98f1516bf4eedb871/?dl=1)     |    DistillOcc-R50 [CFG](configs/effocc_distillocc/fgbg_distill_flashocc-r50_10%_labeled_effocc_10%_scratch.py) [CKPT](https://cloud.tsinghua.edu.cn/f/60f5f5340ea44ad89d07/?dl=1)      |   
| 20% |   Fusion-R18 [CFG](configs/effocc_fusion_r18_data_scales/flashocc_fusion_r18_base_20%_seqs.py) [CKPT](https://cloud.tsinghua.edu.cn/f/2a1a67922b6846c29d8b/?dl=1)      |     FlashOcc-R50 [CFG](configs/flashocc/flashocc-r50_140seqs_scratch.py) [CKPT](https://cloud.tsinghua.edu.cn/f/f8e6391b3c384b6f95a3/?dl=1)     |    DistillOcc-R50 [CFG](configs/effocc_distillocc/fgbg_distill_flashocc-r50_20%_labeled_effocc_20%_scratch.py) [CKPT](https://cloud.tsinghua.edu.cn/f/4e624762d3384b948e62/?dl=1)      |   
| 40% |   Fusion-R18 [CFG](configs/effocc_fusion_r18_data_scales/flashocc_fusion_r18_base_40%_seqs.py) [CKPT](https://cloud.tsinghua.edu.cn/f/ce572ab1ecab4a3ca717/?dl=1)      |     FlashOcc-R50 [CFG](configs/flashocc/flashocc-r50_280seqs_scratch.py) [CKPT](https://cloud.tsinghua.edu.cn/f/24f58a8226af49139181/?dl=1)     |    DistillOcc-R50 [CFG](configs/effocc_distillocc/fgbg_distill_flashocc-r50_40%_labeled_effocc_40%_scratch.py) [CKPT](https://cloud.tsinghua.edu.cn/f/d7ca1d2def84455bb376/?dl=1)      |   
| 60% |   Fusion-R18 [CFG](configs/effocc_fusion_r18_data_scales/flashocc_fusion_r18_base_60%_seqs.py) [CKPT](https://cloud.tsinghua.edu.cn/f/de5b0b091a20472c8789/?dl=1)      |     FlashOcc-R50 [CFG](configs/flashocc/flashocc-r50_420seqs_scratch.py) [CKPT](https://cloud.tsinghua.edu.cn/f/56b2f4bcffb64ebf8ac2/?dl=1)     |    DistillOcc-R50 [CFG](configs/effocc_distillocc/fgbg_distill_flashocc-r50_60%_labeled_effocc_60%_scratch.py) [CKPT](https://cloud.tsinghua.edu.cn/f/926fcba1d0804f5ca9e9/?dl=1)      |   
| 80% |   Fusion-R18 [CFG](configs/effocc_fusion_r18_data_scales/flashocc_fusion_r18_base_80%_seqs.py) [CKPT](https://cloud.tsinghua.edu.cn/f/e42d95a1d8824b78ae05/?dl=1)      |     FlashOcc-R50 [CFG](configs/flashocc/flashocc-r50_560seqs_scratch.py) [CKPT](https://cloud.tsinghua.edu.cn/f/a66c5a68da374d25955d/?dl=1)     |    DistillOcc-R50 [CFG](configs/effocc_distillocc/fgbg_distill_flashocc-r50_80%_labeled_effocc_80%_scratch.py) [CKPT](https://cloud.tsinghua.edu.cn/f/110f330779124c11afdf/?dl=1)      |   
| 100% pretrained |   Fusion-R50 [CFG](configs/effocc_fusion_more_backbones/flashocc_fusion_r50_base_100%_seq.py) [CKPT](https://cloud.tsinghua.edu.cn/f/3dbbb5cfbc9c4bb6a6fc/?dl=1)      |         |      |   
| 100% pretrained |   Fusion-SwinB [CFG](configs/effocc_fusion_more_backbones/flashocc_fusion_swinb_base_100%_seq.py) [CKPT](https://cloud.tsinghua.edu.cn/f/2f86edfdd09a46aa8bdd/?dl=1)      |         |      |   


Exps on OpenOccupancy-nuScenes:

| Settings        | Fusion Model | 
| ----- | ----- |
| 100%            |  Fusion-R18 [CFG](configs/effocc_openoccupancy/effocc-fusion-r18.py) [CKPT](https://cloud.tsinghua.edu.cn/f/b7f2c9d684744b939c8e/?dl=1)   |



Exps on OpenOccFlow-nuScenes:

| Settings        | Fusion Model | 
| ----- | ----- |
| 100%            |  Fusion-R18 [CFG](configs/effocc_openoccflow/flashocc-fusion-r18_flow.py) [CKPT](https://cloud.tsinghua.edu.cn/f/d53ea74ecbb241c89786/?dl=1)   |

Exps on Occ3D-Waymo (Checkpoints not allowed to share under Waymo's regulation):


| Settings        | Model | 
| ----- | ----- |
| 20%_8e            |  Fusion-R18  [CFG](configs/effocc_waymo/flashocc-fusion-waymoD5-1f.py) |
| 100%_24e            |  Fusion-R18 [CFG](configs/effocc_waymo/flashocc-fusion-waymoD1-1f.py)  |
| 20%_8e            |  LiDAR [CFG](configs/effocc_waymo/flashocc-lidar-waymoD5-1f.py) |
| 100%_24e            |  LiDAR [CFG](configs/effocc_waymo/flashocc-lidar-waymoD1-1f.py) |





## Acknowledgements
Thanks to prior excellent open source projects:

- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
- [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy.git)
- [FlashOcc](https://github.com/Yzichen/FlashOCC)
- [CRB-active-3Ddet](https://github.com/Luoyadan/CRB-active-3Ddet)

