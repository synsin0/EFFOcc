
import torch
from .strategy import Strategy, collect_results_gpu, collect_results_cpu

import torch.nn.functional as F
import tqdm
from mmcv.runner import get_dist_info
import mmcv
import time
import torch.distributed as dist
import os

class BALDSampling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(BALDSampling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)


    def enable_dropout(self, model):
        """ Function to enable the dropout layers during test-time """
        i = 0
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                i += 1
                m.train()
        print('**found and enabled {} Dropout layers for random sampling**'.format(i))

    def query(self, leave_pbar=True, cur_epoch=None, gpu_collect=True):
        select_list = []

        # select_nums = cfg.active_train.select_nums
        # val_dataloader_iter = iter(self.unlabelled_loader)
        # val_loader = self.unlabelled_loader
        # total_it_each_epoch = len(self.unlabelled_loader)

        dataset = self.unlabelled_loader.dataset
        rank, world_size = get_dist_info()

        # feed forward the model
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(dataset))
        self.model.eval()
        self.enable_dropout(self.model)

        time.sleep(2)  # This line can prevent deadlock problem in some cases.

        for i, data in enumerate(self.unlabelled_loader):
            with torch.no_grad():
                final_full_cls_logits = self.model(active=True, **data)
                num_classes = final_full_cls_logits.shape[-1]

                for bs in range(final_full_cls_logits.shape[0]):
                    logits = final_full_cls_logits[bs]
                    logits = logits.view(-1, num_classes)
                    
                    occ_score = logits.softmax(-1)    # (B, Dx, Dy, Dz, C)
                    occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)

                    box_values = \
                        -(F.softmax(logits, dim=1) *
                            F.log_softmax(logits, dim=1)).sum(dim=1)
                    
                    """ Aggregate all the boxes values in one point cloud """
                    if self.cfg.active_train.aggregation == 'mean':
                        aggregated_values = torch.mean(box_values)
                    else:
                        raise NotImplementedError
                    
                    # print(aggregated_values)

                    select_list.append([data['img_metas'].data[0][bs]['sample_idx'],aggregated_values.item()])

                if rank == 0:
                    batch_size = final_full_cls_logits.shape[0]
                    batch_size_all = batch_size * world_size
                    if batch_size_all + prog_bar.completed > len(dataset):
                        batch_size_all = len(dataset) - prog_bar.completed
                    for _ in range(batch_size_all):
                        prog_bar.update()
        

        if gpu_collect:
            select_list = collect_results_gpu(select_list, len(self.unlabelled_loader.dataset))
        else:
            select_list = collect_results_cpu(select_list, len(self.unlabelled_loader.dataset))
        
        # if rank !=0:
        #     dist.destroy_process_group()
        #     exit(0)

        if rank == 0:  
            select_dict = {}
            for select in select_list:
                select_dict[select[0]] = select[1]

            # sort and get selected_frames
            select_dict = dict(sorted(select_dict.items(), key=lambda item: item[1]))
            unlabelled_sample_num = len(select_dict.keys())

            selected_frames = list(select_dict.keys())[unlabelled_sample_num - self.cfg.active_train.select_nums:]
              
            # return selected_frames, [x['img_metas'].data['sample_idx'] for x in self.labelled_loader.dataset], [x['img_metas'].data['sample_idx'] for x in self.unlabelled_loader.dataset]
            
            
            mmcv.dump(selected_frames, os.path.join(self.active_label_dir, 'selected_frames_epoch_{}.pkl'.format(cur_epoch)))


            return None
        else:
            return None

