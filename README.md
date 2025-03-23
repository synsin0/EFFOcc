# EFFOcc
EFFOcc: A Minimal Baseline for EFficient Fusion-based 3D Occupancy Network

## Demo videos
We provide lidar-camera occupancy prediction video of Occ3D-nuScenes dataset. 

https://github.com/synsin0/EFFOcc/assets/37300008/6ab8238f-1d7f-4e4b-b4de-daff6a99ba41

## Data Setup
We follow the setups of [BEVDet](https://github.com/HuangJunJie2017/BEVDet) for data preprocessing of nuScenes dataset.

## Models
Exps on Occ3D-nuScenes:

| Settings | Fusion Model | FlashOcc | Distilled Model | 
| ----- | ----- | -------- | -------- |
| 100% pretrained |   Fusion-R18 [CFG](configs/effocc_fusion_r18_data_scales/flashocc_fusion_r18_base_100%_seqs.py) [CKPT](https://cloud.tsinghua.edu.cn/f/0dd01c193f46496a88b8/?dl=1)      |     FlashOcc-R50 [CFG](configs/flashocc/flashocc-r50_from_scratch.py) [CKPT]()     |    DistillOcc-R50 [CFG](configs/effocc_distillocc/fgbg_distill_flashocc-r50_100%_labeled_effocc_r18_100%.py) [CKPT](https://cloud.tsinghua.edu.cn/f/19d2ada1c4c241a8bc2d/?dl=1)      |      
| 100% from scratch |   Fusion-R18 [CFG](configs/effocc_fusion_r18_data_scales/flashocc_fusion_r18_base_100%_seqs.py) [CKPT](https://cloud.tsinghua.edu.cn/f/0dd01c193f46496a88b8/?dl=1)      |     FlashOcc-R50 [CFG](configs/flashocc/flashocc-r50.py) [CKPT]()     |    DistillOcc-R50 [CFG](configs/effocc_distillocc/fgbg_distill_flashocc-r50_100%_labeled_effocc_100%_scratch.py) [CKPT](https://cloud.tsinghua.edu.cn/f/81656568aa164e8b869d/?dl=1)      |    
| 5% |   Fusion-R18 [CFG](configs/effocc_fusion_r18_data_scales/flashocc_fusion_r18_base_5%_seqs.py) [CKPT](https://cloud.tsinghua.edu.cn/f/7c6b0a71a22c42acaf9c/?dl=1)      |     FlashOcc-R50 [CFG](configs/flashocc/flashocc-r50_35seqs_scratch.py) [CKPT]()     |    DistillOcc-R50 [CFG](configs/effocc_distillocc/fgbg_distill_flashocc-r50_5%_labeled_effocc_5%_scratch.py) [CKPT]()      |   
| 10% |   Fusion-R18 [CFG](configs/effocc_fusion_r18_data_scales/flashocc_fusion_r18_base_10%_seqs.py) [CKPT](https://cloud.tsinghua.edu.cn/f/ce2b9371aba64695969a/?dl=1)      |     FlashOcc-R50 [CFG](configs/flashocc/flashocc-r50_70seqs_scratch.py) [CKPT]()     |    DistillOcc-R50 [CFG](configs/effocc_distillocc/fgbg_distill_flashocc-r50_10%_labeled_effocc_10%_scratch.py) [CKPT](https://cloud.tsinghua.edu.cn/f/60f5f5340ea44ad89d07/?dl=1)      |   
| 20% |   Fusion-R18 [CFG](configs/effocc_fusion_r18_data_scales/flashocc_fusion_r18_base_20%_seqs.py) [CKPT](https://cloud.tsinghua.edu.cn/f/2a1a67922b6846c29d8b/?dl=1)      |     FlashOcc-R50 [CFG](configs/flashocc/flashocc-r50_140seqs_scratch.py) [CKPT]()     |    DistillOcc-R50 [CFG](configs/effocc_distillocc/fgbg_distill_flashocc-r50_20%_labeled_effocc_20%_scratch.py) [CKPT](https://cloud.tsinghua.edu.cn/f/4e624762d3384b948e62/?dl=1)      |   
| 40% |   Fusion-R18 [CFG](configs/effocc_fusion_r18_data_scales/flashocc_fusion_r18_base_40%_seqs.py) [CKPT](https://cloud.tsinghua.edu.cn/f/ce572ab1ecab4a3ca717/?dl=1)      |     FlashOcc-R50 [CFG](configs/flashocc/flashocc-r50_280seqs_scratch.py) [CKPT]()     |    DistillOcc-R50 [CFG](configs/effocc_distillocc/fgbg_distill_flashocc-r50_40%_labeled_effocc_40%_scratch.py) [CKPT](https://cloud.tsinghua.edu.cn/f/d7ca1d2def84455bb376/?dl=1)      |   
| 60% |   Fusion-R18 [CFG](configs/effocc_fusion_r18_data_scales/flashocc_fusion_r18_base_60%_seqs.py) [CKPT](https://cloud.tsinghua.edu.cn/f/de5b0b091a20472c8789/?dl=1)      |     FlashOcc-R50 [CFG](configs/flashocc/flashocc-r50_420seqs_scratch.py) [CKPT]()     |    DistillOcc-R50 [CFG](configs/effocc_distillocc/fgbg_distill_flashocc-r50_60%_labeled_effocc_60%_scratch.py) [CKPT](https://cloud.tsinghua.edu.cn/f/926fcba1d0804f5ca9e9/?dl=1)      |   
| 80% |   Fusion-R18 [CFG](configs/effocc_fusion_r18_data_scales/flashocc_fusion_r18_base_80%_seqs.py) [CKPT](https://cloud.tsinghua.edu.cn/f/e42d95a1d8824b78ae05/?dl=1)      |     FlashOcc-R50 [CFG](configs/flashocc/flashocc-r50_560seqs_scratch.py) [CKPT]()     |    DistillOcc-R50 [CFG](configs/effocc_distillocc/fgbg_distill_flashocc-r50_80%_labeled_effocc_80%_scratch.py) [CKPT](https://cloud.tsinghua.edu.cn/f/110f330779124c11afdf/?dl=1)      |   
| 100% pretrained |   Fusion-R50 [CFG](configs/effocc_fusion_more_backbones/flashocc_fusion_r50_base_100%_seq.py) [CKPT](https://cloud.tsinghua.edu.cn/f/3dbbb5cfbc9c4bb6a6fc/?dl=1)      |         |      |   
| 100% pretrained |   Fusion-SwinB [CFG](configs/effocc_fusion_more_backbones/flashocc_fusion_swinb_base_100%_seq.py) [CKPT](https://cloud.tsinghua.edu.cn/f/2f86edfdd09a46aa8bdd/?dl=1)      |         |      |   


Exps on OpenOccupancy-nuScenes:

| Settings        | Fusion Model | 
| ----- | ----- |
| 100%            |  Fusion-R18 [CFG](configs/effocc_openoccupancy/effocc-fusion-r18.py) [CKPT](https://cloud.tsinghua.edu.cn/f/b7f2c9d684744b939c8e/?dl=1)   |

| 100%            |  Fusion-R50 [CFG](configs/effocc_openoccupancy/effocc-fusion-r50.py) [CKPT]()   |


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



## Abstract（TL DR）
EFFOcc explores towards the minimal (minimal computation costs and minimal label costs) baseline for fast and high-performance 3D occupancy prediction with lidar-camera fusion. We show lightweight BEV-based fusion occnet can perform as well as voxel-based fusion occnets.


## Acknowledgements
Thanks to prior excellent open source projects:

- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
- [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy.git)
- [FlashOcc](https://github.com/Yzichen/FlashOCC)
- [CRB-active-3Ddet](https://github.com/Luoyadan/CRB-active-3Ddet)

