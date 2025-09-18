import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
from collections import deque
import copy
from einops import rearrange

from .zs_clip import ZeroShotCLIP
from clip.classifier import OODDetector
from utils.utils import *
from minisom import MiniSom
from utils.torchsom import TorchMiniSom
import math
import numpy as np
import torchvision
from typing import Iterable

import torch
from torch.optim._multi_tensor import SGD
import torch.nn.utils.prune as prune
import torch.nn.utils as utils

__all__ = ["SAMSGD"]

def _l2_normalize(x, dim=-1, eps=1e-12):
    return x / (x.norm(dim=dim, keepdim=True) + eps)
class SAMSGD(SGD):
    """ SGD wrapped with Sharp-Aware Minimization

    Args:
        params: tensors to be optimized
        lr: learning rate
        momentum: momentum factor
        dampening: damping factor
        weight_decay: weight decay factor
        nesterov: enables Nesterov momentum
        rho: neighborhood size

    """

    def __init__(self,
                 params: Iterable[torch.Tensor],
                 lr: float,
                 momentum: float = 0,
                 dampening: float = 0,
                 weight_decay: float = 0,
                 nesterov: bool = False,
                 rho: float = 0.05,
                 ):
        if rho <= 0:
            raise ValueError(f"Invalid neighborhood size: {rho}")
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        # todo: generalize this
        if len(self.param_groups) > 1:
            raise ValueError("Not supported")
        self.param_groups[0]["rho"] = rho

    @torch.no_grad()
    def step(self,
             closure
             ) -> torch.Tensor:
        """

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns: the loss value evaluated on the original point

        """
        closure = torch.enable_grad()(closure)
        loss = closure().detach()

        for group in self.param_groups:
            grads = []
            params_with_grads = []

            rho = group['rho']
            # update internal_optim's learning rate

            for p in group['params']:
                if p.grad is not None:
                    # without clone().detach(), p.grad will be zeroed by closure()
                    grads.append(p.grad.clone().detach())
                    params_with_grads.append(p)
            device = grads[0].device

            # compute \hat{\epsilon}=\rho/\norm{g}\|g\|
            grad_norm = torch.stack([g.detach().norm(2).to(device) for g in grads]).norm(2)
            epsilon = grads  # alias for readability
            torch._foreach_mul_(epsilon, rho / grad_norm)

            # virtual step toward \epsilon
            torch._foreach_add_(params_with_grads, epsilon)
            # compute g=\nabla_w L_B(w)|_{w+\hat{\epsilon}}
            closure()
            # virtual step back to the original point
            torch._foreach_sub_(params_with_grads, epsilon)

        super().step()
        return loss
## ood memory 为队列
class ZeroShotNTTA_queue(ZeroShotCLIP):
    def __init__(self, *args, **kwargs):
        super(ZeroShotNTTA_queue, self).__init__(*args, **kwargs)

        if self.args.model.arch == "ViT-L/14":
            feat_dim = 768
        elif self.args.model.arch == "ViT-B/16":
            feat_dim = 512
        else:
            feat_dim = 1024
        self.ood_net = OODDetector(feat_dim).to(self.model.image_encoder.conv1.weight.device)
        self.criterion = nn.CrossEntropyLoss() 

        self.optimizer = get_optimizer(self.args, self.ood_net.parameters(), lr=self.args.optim.lr)
       

        self.os_detector_queue = []
        self.loss_history = []
        self.ood_feature_queue = []
        self.id_feature_queue = []

        queue_length = self.args.inference.ttda_queue_length
        self.queues = {
            'ood_detector_out_queue': deque(maxlen=queue_length),
            'unseen_mask_queue': deque(maxlen=queue_length),
            'target_queue': deque(maxlen=queue_length),
            'clip_output_queue': deque(maxlen=queue_length)
        }
        self.ttda_queue = []

        ### no test time adaption!!!!
        self.text_features = self.model.get_text_features()
        ### memo para
        self.memory_size = 10
        self.set_id_memory_bank()


    def set_id_memory_bank(self):
        ## reset feat memory to vanilla text features, for each ID/OOD pair.
        text_features = self.text_features
        self.feat_memory = text_features.unsqueeze(1)  ## 1000 * 1 *512
        extented_empty_memory = torch.zeros_like(self.feat_memory).repeat(1, self.memory_size, 1)
        self.feat_memory = torch.cat((self.feat_memory, extented_empty_memory), dim=1)
        self.indice_memory = torch.ones(text_features.size(0))  ### 1000
        self.entropy_memory = torch.zeros(self.feat_memory.size(0), self.feat_memory.size(1)).to(
            text_features.device)  ## 1000*(1+memory_size)


    def insert_feat_memo(self,clip_output,class_num,unseen_mask_id,image_features):
        _, pred_all = torch.max(clip_output[:, :class_num], dim=1)
        prob_id = torch.softmax(clip_output[:, :class_num], dim=1)
        for i in range(unseen_mask_id.size(0)):
            if unseen_mask_id[i].item():
                predicted_cate = pred_all[i].item()
                predicted_prob = prob_id[i]
                current_instance_entropy = -(predicted_prob * (torch.log(predicted_prob + 1e-8))).sum()
                # pdb.set_trace()
                # self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                if self.indice_memory[predicted_cate] == self.memory_size:
                    if (current_instance_entropy < self.entropy_memory[predicted_cate]).sum() == 0:
                        pass  ## the entropy of current test image is very large.
                    else:
                        # replace the one with the maximum entropy!! to update. find the one with the maximum entropy.
                        _, indice = torch.sort(self.entropy_memory[predicted_cate])
                        to_replace_indice = indice[-1]  ## with max entropy, ascending.
                        self.feat_memory[predicted_cate][to_replace_indice] = image_features[i]
                        self.entropy_memory[predicted_cate][to_replace_indice] = current_instance_entropy
                else:
                    self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                    self.entropy_memory[predicted_cate][
                        self.indice_memory[predicted_cate].long()] = current_instance_entropy
                    self.indice_memory[predicted_cate] += 1
            else:
                pass
    def get_unseen_mask(self, clip_output, image, image_feature_raw, step, target):
        class_num = self.class_num

        unseen_mask,unseen_mask_ood, unseen_mask_id, clip_output,ood_score = super().get_unseen_mask(clip_output, image, image_feature_raw, step, target)
        self.ood_feature_queue.append(image_feature_raw[unseen_mask_ood])

        self.insert_feat_memo(clip_output,class_num,unseen_mask_id,image_feature_raw)

        if len(self.ood_feature_queue)>100:
            self.ood_feature_queue = self.ood_feature_queue[-100:]

        unseen_mask,visual_output= super().get_unseen_mask_visual_memo_queue(clip_output, image, image_feature_raw, self.id_feature_queue , self.ood_feature_queue, step, target,self.feat_memory)


        self.model.eval()
        self.ood_net.train()

        with torch.enable_grad():   
            if self.args.inference.batch_size == 1:
                ood_detector_out = self.ood_net(image_feature_raw)
                self.ttda_queue.extend(ood_detector_out)
                self.queues['ood_detector_out_queue'].append(ood_detector_out)
                self.queues['unseen_mask_queue'].append(unseen_mask)
                self.queues['target_queue'].append(target)
                self.queues['clip_output_queue'].append(clip_output)
                if step != 0 and step % self.args.inference.ttda_queue_length == 0:                    
                    batch_ood_detector_out = torch.stack(list(self.queues['ood_detector_out_queue']), dim=0).squeeze(1)
                    batch_unseen_mask = torch.stack(list(self.queues['unseen_mask_queue']), dim=0).squeeze(1)
                    batch_target = torch.stack(list(self.queues['target_queue']), dim=0).squeeze(1)
                    batch_clip = torch.stack(list(self.queues['clip_output_queue']), dim=0).squeeze(1)

                    self.update_detector(batch_ood_detector_out, batch_unseen_mask, batch_target, step, batch_clip)
            else:
                ood_detector_out = self.ood_net(image_feature_raw)
                self.update_detector(ood_detector_out, unseen_mask, target, step, clip_output)

        if self.args.inference.batch_size == 1:
            ttda_queue_length = self.args.inference.ttda_queue_length
        else:
            ttda_queue_length = self.args.inference.batch_size

        if step * self.args.inference.batch_size > self.args.inference.using_ttda_step * ttda_queue_length:
            predict = F.softmax(ood_detector_out, 1) # [bs, 2]

            ood_score = predict[:, 0]
            self.os_detector_queue.extend(ood_score.detach().cpu().tolist())
            self.os_detector_queue = self.os_detector_queue[-self.args.inference.queue_length:]
            if self.args.inference.threshold_type == 'adaptive':
                threshold_range = np.arange(0, 1, 0.01)
                criterias = [compute_os_variance(np.array(self.os_detector_queue), th) for th in threshold_range]
                best_threshold = threshold_range[np.argmin(criterias)]
            else:
                best_threshold = self.args.inference.fixed_threshold
            unseen_mask = (ood_score > best_threshold)

            conf = predict[:, 1]


            return unseen_mask, conf, visual_output
        
        return unseen_mask, visual_output , [1], [1], target
    
    
    def update_detector(self, ood_detector_out, unseen_mask, target, step, clip_output=None):
        logit = F.softmax(clip_output, dim=1)
        conf, _ = logit.max(1) # [bs] 
        
        # NOTE that we don't use ID/OOD samples' target here, we just use Gaussian noise samples' target
        # Artificially added Gaussian noise samples: target = -1000, we can obtain the target of these samples
        unseen_mask[target == -1000] = True

        select_ood_indices = torch.where(unseen_mask)[0]
        select_id_indices = torch.where(~unseen_mask)[0]
        selected_indices = torch.cat([select_id_indices, select_ood_indices])
        selected_labels = torch.cat([torch.ones(len(select_id_indices)), torch.zeros(len(select_ood_indices))]).cuda()
      
        loss = self.criterion(ood_detector_out[selected_indices], selected_labels.long())

        self.optimizer.zero_grad()
        loss.backward()
        self.loss_history.append(loss.item())
        self.optimizer.step()




### ood memory 使用som得到prototype
class ZeroShotNTTA(ZeroShotCLIP):
    def __init__(self, *args, **kwargs):
        super(ZeroShotNTTA, self).__init__(*args, **kwargs)

        if self.args.model.arch == "ViT-L/14":
            self.feat_dim = 768
        elif self.args.model.arch == "ViT-B/16":
            self.feat_dim = 512
        else:
            self.feat_dim = 1024
        self.ood_net = OODDetector(self.feat_dim).to(self.model.image_encoder.conv1.weight.device)
        self.criterion = nn.CrossEntropyLoss() 

        self.optimizer = get_optimizer(self.args, self.ood_net.parameters(), lr=self.args.optim.lr)
        #self.optimizer = SAMSGD(self.ood_net.parameters(), lr=0.1, rho=0.05)

        self.os_detector_queue = []
        self.loss_history = []

        self.ood_feature_queue = []

        self.id_feature_queue = []

        
        queue_length = self.args.inference.ttda_queue_length
        self.queues = {
            'ood_detector_out_queue': deque(maxlen=queue_length),
            'unseen_mask_queue': deque(maxlen=queue_length),
            'target_queue': deque(maxlen=queue_length),
            'clip_output_queue': deque(maxlen=queue_length)
        }
        self.ttda_queue = []

        ### no test time adaption!!!!
        self.text_features = self.model.get_text_features()
        ### memo para
        self.memory_size = 10
        self.set_id_memory_bank()



        ### som 只有numpy 后续改成tensor
        self.size = 15   ###   经验公式：决定输出层尺寸 math.ceil(np.sqrt(5 * np.sqrt(N)))  
        self.som_lr = 0.05
        self.som_sigma = 2
        self.som = MiniSom(self.size, self.size, self.feat_dim, self.som_sigma, self.som_lr, 
              neighborhood_function='gaussian')
        self.start_iter = -1  ## get enough sample to warmup
        self.start_queue = []  ## save sample 
        self.first_train_iter = 5000 ### train with start queue sample
        self.iter = 100

        self.count_winner = np.zeros((self.size,self.size))
        self.ood_som_prototypes = np.zeros((self.size,self.size,self.feat_dim))






    def set_id_memory_bank(self):
        ## reset feat memory to vanilla text features, for each ID/OOD pair.
        text_features = self.text_features
        self.feat_memory = text_features.unsqueeze(1)  ## 1000 * 1 *512
        extented_empty_memory = torch.zeros_like(self.feat_memory).repeat(1, self.memory_size, 1)
        self.feat_memory = torch.cat((self.feat_memory, extented_empty_memory), dim=1)
        self.indice_memory = torch.ones(text_features.size(0))  ### 1000
        self.entropy_memory = torch.zeros(self.feat_memory.size(0), self.feat_memory.size(1)).to(
            text_features.device)  ## 1000*(1+memory_size)


    def insert_feat_memo(self,clip_output,class_num,unseen_mask_id,image_features):
        _, pred_all = torch.max(clip_output[:, :class_num], dim=1)
        prob_id = torch.softmax(clip_output[:, :class_num], dim=1)
        for i in range(unseen_mask_id.size(0)):
            if unseen_mask_id[i].item():
                predicted_cate = pred_all[i].item()
                predicted_prob = prob_id[i]
                current_instance_entropy = -(predicted_prob * (torch.log(predicted_prob + 1e-8))).sum()
                # pdb.set_trace()
                # self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                if self.indice_memory[predicted_cate] == self.memory_size:
                    if (current_instance_entropy < self.entropy_memory[predicted_cate]).sum() == 0:
                        pass  ## the entropy of current test image is very large.
                    else:
                        # replace the one with the maximum entropy!! to update. find the one with the maximum entropy.
                        _, indice = torch.sort(self.entropy_memory[predicted_cate])
                        to_replace_indice = indice[-1]  ## with max entropy, ascending.
                        self.feat_memory[predicted_cate][to_replace_indice] = image_features[i]
                        self.entropy_memory[predicted_cate][to_replace_indice] = current_instance_entropy
                else:
                    self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                    self.entropy_memory[predicted_cate][
                        self.indice_memory[predicted_cate].long()] = current_instance_entropy
                    self.indice_memory[predicted_cate] += 1
            else:
                pass
    def get_unseen_mask(self, clip_output, image, image_feature_raw, step, target):
        class_num = self.class_num
        # if step < self.args.inference.using_ttda_step:
        
        unseen_mask,unseen_mask_ood,unseen_mask_id, visual_output,ood_score_vis = super().get_unseen_mask(clip_output, image, image_feature_raw, step, target)
        # else:
        #     unseen_mask,unseen_mask_ood,unseen_mask_id, visual_output,ood_score_vis = super().get_unseen_mask_visual_memo_gap(clip_output, image, image_feature_raw, self.id_feature_queue , self.ood_proxy, step, target,self.feat_memory)
        # else:
        #     unseen_mask,unseen_mask_ood,unseen_mask_id, visual_output = super().get_unseen_mask_visual_memo_gap(clip_output, image, image_feature_raw, self.id_feature_queue , self.ood_proxy, step, target,self.feat_memory)

        self.ood_feature_queue.append(image_feature_raw[unseen_mask_ood])
        
        # if step<self.start_iter:
        #     self.start_queue.append(image_feature_raw[unseen_mask_ood])
        # elif step == self.start_iter:
        #     print(len(self.start_queue))
        #     warm_features = torch.concat(self.start_queue).cpu().numpy()
        #     self.som.pca_weights_init(warm_features)
        #     self.som.train_batch(warm_features, self.first_train_iter, verbose=False)
        #     self.count_winner+=self.som.activation_response(warm_features)
        #     for key in self.som.win_map(warm_features).keys():
        #         i,j = key
                
        #         tmp = np.concatenate(self.som.win_map(warm_features)[key]).reshape(-1,self.feat_dim)
                
        #         self.ood_som_prototypes[i,j] += np.sum(tmp,axis=0)
        #         number = self.som.activation_response(warm_features)[i,j]
        #         if number>1:
        #             self.ood_som_prototypes[i,j] = self.ood_som_prototypes[i,j]/number
        # else:
        
        tf = image_feature_raw[unseen_mask_ood].cpu().numpy()
        self.som.train_batch(tf, self.iter, verbose=False)
        

        for key in self.som.win_map(tf).keys():
            i,j = key
            tmp = np.concatenate(self.som.win_map(tf)[key]).reshape(-1,self.feat_dim)
            tmp_ = np.sum(tmp,axis=0)
            number = self.som.activation_response(tf)[i,j]
            if number>0:
                self.ood_som_prototypes[i,j] = ((self.ood_som_prototypes[i,j]*self.count_winner[i,j])+tmp_)/(number+self.count_winner[i,j])
            
        self.count_winner+=self.som.activation_response(tf)



        self.insert_feat_memo(clip_output,class_num,unseen_mask_id,image_feature_raw)

        if step < self.start_iter:
            if len(self.ood_feature_queue)>1000:
                self.ood_feature_queue = self.ood_feature_queue[-1000:]
            unseen_mask, visual_output = super().get_unseen_mask_visual_memo_queue(clip_output, image, image_feature_raw, self.id_feature_queue , self.ood_feature_queue, step, target,self.feat_memory)

        else:
            # 将三维数组重塑为二维 (n*n, feat_dim)
            reshaped_prototypes = self.ood_som_prototypes.reshape(-1, self.feat_dim)
            # 创建掩码：标记哪些行（特征向量）至少有一个非零元素
            non_zero_mask = np.any(reshaped_prototypes != 0, axis=1)
            # 提取非零特征向量
            non_zero_prototypes = reshaped_prototypes[non_zero_mask]
            # 转换为 PyTorch 张量并发送到设备
            self.ood_proxy = torch.tensor(non_zero_prototypes, dtype=torch.float32).cuda()
            unseen_mask,unseen_mask_ood,unseen_mask_id, visual_output,ood_score_vis = super().get_unseen_mask_visual_memo_gap(clip_output, image, image_feature_raw, self.id_feature_queue , self.ood_proxy, step, target,self.feat_memory)





        self.model.eval()
        self.ood_net.train()

        with torch.enable_grad():   
            if self.args.inference.batch_size == 1:
                ood_detector_out = self.ood_net(image_feature_raw)
                self.ttda_queue.extend(ood_detector_out)
                self.queues['ood_detector_out_queue'].append(ood_detector_out)
                self.queues['unseen_mask_queue'].append(unseen_mask_ood)
                self.queues['target_queue'].append(target)
                self.queues['clip_output_queue'].append(clip_output)
                if step != 0 and step % self.args.inference.ttda_queue_length == 0:                    
                    batch_ood_detector_out = torch.stack(list(self.queues['ood_detector_out_queue']), dim=0).squeeze(1)
                    batch_unseen_mask = torch.stack(list(self.queues['unseen_mask_queue']), dim=0).squeeze(1)
                    batch_target = torch.stack(list(self.queues['target_queue']), dim=0).squeeze(1)
                    batch_clip = torch.stack(list(self.queues['clip_output_queue']), dim=0).squeeze(1)

                    self.update_detector(batch_ood_detector_out, batch_unseen_mask, batch_target, step, batch_clip)
            else:
                
                ood_detector_out = self.ood_net(image_feature_raw)


                self.update_detector(ood_detector_out, unseen_mask_ood,unseen_mask_id, target, step, clip_output)

        if self.args.inference.batch_size == 1:
            ttda_queue_length = self.args.inference.ttda_queue_length
        else:
            ttda_queue_length = self.args.inference.batch_size

        if step * self.args.inference.batch_size > self.args.inference.using_ttda_step * ttda_queue_length:
            vis_f = image_feature_raw / image_feature_raw.norm(dim=-1, keepdim=True) ### bs 512
             ### ood prototype  224 512
            # sim = torch.mean(vis_f@self.ood_proxy.T,dim=1)  ### bs 224 


            predict = F.softmax(ood_detector_out, 1) # [bs, 2]
            ood_score = predict[:, 0]
            self.os_detector_queue.extend(ood_score.detach().cpu().tolist())
            self.os_detector_queue = self.os_detector_queue[-self.args.inference.queue_length:]
            if self.args.inference.threshold_type == 'adaptive':
                threshold_range = np.arange(0, 1, 0.01)
                criterias = [compute_os_variance(np.array(self.os_detector_queue), th) for th in threshold_range]
                best_threshold = threshold_range[np.argmin(criterias)]
            else:
                best_threshold = self.args.inference.fixed_threshold

            # ood_score += 0.01*sim
            unseen_mask = (ood_score > best_threshold)

            conf = predict[:, 1]


            return unseen_mask, conf, visual_output
        
        return unseen_mask, visual_output
    
    
    def update_detector(self, ood_detector_out, unseen_mask_ood,unseen_mask_id,  target, step, clip_output=None):
        logit = F.softmax(clip_output, dim=1)
        conf, _ = logit.max(1) # [bs] 
        
        # NOTE that we don't use ID/OOD samples' target here, we just use Gaussian noise samples' target
        # Artificially added Gaussian noise samples: target = -1000, we can obtain the target of these samples
        unseen_mask_ood[target == -1000] = True
        unseen_mask_id[target == -1000] = True
        select_ood_indices = torch.where(unseen_mask_ood)[0]
        select_id_indices = torch.where(unseen_mask_id)[0]
        selected_indices = torch.cat([select_id_indices, select_ood_indices])
        selected_labels = torch.cat([torch.ones(len(select_id_indices)), torch.zeros(len(select_ood_indices))]).cuda()
      
        loss = self.criterion(ood_detector_out[selected_indices], selected_labels.long())
        self.optimizer.zero_grad()
        loss.backward()
        self.loss_history.append(loss.item())
        self.optimizer.step()


        # def closure():
        #     self.optimizer.zero_grad()
        #     loss = self.criterion(ood_detector_out[selected_indices], selected_labels.long())

        #     loss.backward(retain_graph=True)
        #     return loss
        # loss = self.optimizer.step(closure)
        # self.optimizer.zero_grad()








class ZeroShotNTTA_DEYO(ZeroShotCLIP):
    def __init__(self, *args, **kwargs):
        super(ZeroShotNTTA_DEYO, self).__init__(*args, **kwargs)

        if self.args.model.arch == "ViT-L/14":
            self.feat_dim = 768
        elif self.args.model.arch == "ViT-B/16":
            self.feat_dim = 512
        else:
            self.feat_dim = 1024
        self.ood_net = OODDetector(self.feat_dim).to(self.model.image_encoder.conv1.weight.device)
        self.criterion = nn.CrossEntropyLoss() 

        self.optimizer = get_optimizer(self.args, self.ood_net.parameters(), lr=self.args.optim.lr)
        #self.optimizer = SAMSGD(self.ood_net.parameters(), lr=0.1, rho=0.05)

        self.os_detector_queue = []
        self.loss_history = []

        self.ood_feature_queue = []

        self.id_feature_queue = []

        
        queue_length = self.args.inference.ttda_queue_length
        self.queues = {
            'ood_detector_out_queue': deque(maxlen=queue_length),
            'unseen_mask_queue': deque(maxlen=queue_length),
            'target_queue': deque(maxlen=queue_length),
            'clip_output_queue': deque(maxlen=queue_length)
        }
        self.ttda_queue = []

        ### no test time adaption!!!!
        self.text_features = self.model.get_text_features()
        ### memo para
        self.memory_size = 10
        self.set_id_memory_bank()

       

        ### som 只有numpy 后续改成tensor
        self.size = 15   ###   经验公式：决定输出层尺寸 math.ceil(np.sqrt(5 * np.sqrt(N)))  
        self.som_lr = 0.05
        self.som_sigma = 2
        self.som = MiniSom(self.size, self.size, self.feat_dim, self.som_sigma, self.som_lr, 
              neighborhood_function='gaussian',random_seed = 0)
        self.start_iter = -1  ## get enough sample to warmup
        self.start_queue = []  ## save sample 
        self.first_train_iter = 5000 ### train with start queue sample
        self.iter = 100

        self.count_winner = np.zeros((self.size,self.size))
        self.ood_som_prototypes = np.zeros((self.size,self.size,self.feat_dim))



        self.aug_type = 'patch'
        self.filter_ent = self.args.training.filter_ent
        self.filter_plpd = self.args.training.filter_plpd

        self.plpd_threshold = self.args.training.plpd_threshold
        self.deyo_margin = self.args.training.deyo_margin
        self.patch_len = 20
        print('plpd:' ,self.plpd_threshold)
        print('deyo', self.deyo_margin)

    def set_id_memory_bank(self):
        ## reset feat memory to vanilla text features, for each ID/OOD pair.
        text_features = self.text_features
        self.feat_memory = text_features.unsqueeze(1)  ## 1000 * 1 *512
        extented_empty_memory = torch.zeros_like(self.feat_memory).repeat(1, self.memory_size, 1)
        self.feat_memory = torch.cat((self.feat_memory, extented_empty_memory), dim=1)
        self.indice_memory = torch.ones(text_features.size(0))  ### 1000
        self.entropy_memory = torch.zeros(self.feat_memory.size(0), self.feat_memory.size(1)).to(
            text_features.device)  ## 1000*(1+memory_size)


    def insert_feat_memo(self,clip_output,class_num,unseen_mask_id,image_features):
        _, pred_all = torch.max(clip_output[:, :class_num], dim=1)
        prob_id = torch.softmax(clip_output[:, :class_num], dim=1)
        for i in range(unseen_mask_id.size(0)):
            if unseen_mask_id[i].item():
                predicted_cate = pred_all[i].item()
                predicted_prob = prob_id[i]
                current_instance_entropy = -(predicted_prob * (torch.log(predicted_prob + 1e-8))).sum()
                # pdb.set_trace()
                # self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                if self.indice_memory[predicted_cate] == self.memory_size:
                    if (current_instance_entropy < self.entropy_memory[predicted_cate]).sum() == 0:
                        pass  ## the entropy of current test image is very large.
                    else:
                        # replace the one with the maximum entropy!! to update. find the one with the maximum entropy.
                        _, indice = torch.sort(self.entropy_memory[predicted_cate])
                        to_replace_indice = indice[-1]  ## with max entropy, ascending.
                        self.feat_memory[predicted_cate][to_replace_indice] = image_features[i]
                        self.entropy_memory[predicted_cate][to_replace_indice] = current_instance_entropy
                else:
                    self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                    self.entropy_memory[predicted_cate][
                        self.indice_memory[predicted_cate].long()] = current_instance_entropy
                    self.indice_memory[predicted_cate] += 1
            else:
                pass


    def get_ent_plpd(self, clip_output, image, image_feature_raw, step, target):
        class_num = self.class_num
        outputs = clip_output
         # clip OUTPUT：MCM ouput
         # 计算ent

        temperature = 1   ## output 中已经100了
        temp_plpd = 0.1

        entropys = softmax_entropy(outputs)

        image = image.detach()
        x = image
        x_prime = x
        x_prime = x_prime.detach()
        ## patch
        resize_t = torchvision.transforms.Resize(((x.shape[-1]//self.patch_len)*self.patch_len,(x.shape[-1]//self.patch_len)*self.patch_len))
        resize_o = torchvision.transforms.Resize((x.shape[-1],x.shape[-1]))
        x_prime = resize_t(x_prime)
        x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=self.patch_len, ps2=self.patch_len)
        perm_idx = torch.argsort(torch.rand(x_prime.shape[0],x_prime.shape[1]), dim=-1)
        x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1),perm_idx]
        x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=self.patch_len, ps2=self.patch_len)
        x_prime = resize_o(x_prime)
        print('x_prime_shape:',x_prime.shape)
        ### MCM
        with torch.no_grad():
            outputs_prime,_ = self.model(x_prime)
        prob_outputs = (outputs *  temp_plpd).softmax(1)
        prob_outputs_prime = (outputs_prime * temp_plpd).softmax(1)
        cls1 = prob_outputs.argmax(dim=1)
        plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1,1))
        plpd = plpd.reshape(-1)
        return entropys, plpd

    def get_unseen_mask(self, clip_output, image, image_feature_raw, step, target):
        class_num = self.class_num
        outputs = clip_output
         # clip OUTPUT：MCM ouput
         # 计算ent
       
        entropys = softmax_entropy(outputs)
        if self.filter_ent:
            filter_ids_1 = torch.where(entropys > self.deyo_margin)
        else:    
            filter_ids_1 = torch.where((entropys >= math.log(1000)))

        x = image.clone()
        x_prime = x
        x_prime = x_prime.detach()
        ## patch
        if self.aug_type == 'patch':
            resize_t = torchvision.transforms.Resize(((x.shape[-1]//self.patch_len)*self.patch_len,(x.shape[-1]//self.patch_len)*self.patch_len))
            resize_o = torchvision.transforms.Resize((x.shape[-1],x.shape[-1]))
            x_prime = resize_t(x_prime)
            x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=self.patch_len, ps2=self.patch_len)
            perm_idx = torch.argsort(torch.rand(x_prime.shape[0],x_prime.shape[1]), dim=-1)
            x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1),perm_idx]
            x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=self.patch_len, ps2=self.patch_len)
            x_prime = resize_o(x_prime)

        ## pixel 
        elif self.aug_type == 'pixel':
            x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
            x_prime = x_prime[:,:,torch.randperm(x_prime.shape[-1])]
            x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=x.shape[-1], ps2=x.shape[-1])
        elif self.aug_type == 'gaussian':
            noise = torch.randn_like(x_prime) * 0.5  # 噪声强度可调
            x_prime = x_prime + noise
            x_prime = torch.clamp(x_prime, 0.0, 1.0)  # 假设输入在 [0,1] 范围


        ### MCM
        with torch.no_grad():
            outputs_prime,_ = self.model(x_prime)
        prob_outputs = (outputs * 0.1  ).softmax(1)
        prob_outputs_prime = (outputs_prime * 0.1 ).softmax(1)
        cls1 = prob_outputs.argmax(dim=1)
        plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1,1))
        plpd = plpd.reshape(-1)
    
        if self.filter_plpd:
            filter_ids_2 = torch.where((plpd < self.plpd_threshold))
        else:
            filter_ids_2 = torch.where(plpd <= -2.0)
        # entropys = entropys[filter_ids_2]
        # plpd = plpd[filter_ids_2]



        unseen_mask,unseen_mask_ood, unseen_mask_id, clip_output,ood_score = super().get_unseen_mask(clip_output, image, image_feature_raw, step, target)
 

 
        tf = image_feature_raw[unseen_mask_ood].cpu().numpy()
        self.som.train_batch(tf, self.iter, verbose=False)
        
        for key in self.som.win_map(tf).keys():
            i,j = key
            tmp = np.concatenate(self.som.win_map(tf)[key]).reshape(-1,self.feat_dim)
            tmp_ = np.sum(tmp,axis=0)
            number = self.som.activation_response(tf)[i,j]
            if number>0:
                self.ood_som_prototypes[i,j] = ((self.ood_som_prototypes[i,j]*self.count_winner[i,j])+tmp_)/(number+self.count_winner[i,j])
            
        self.count_winner+=self.som.activation_response(tf)

        self.insert_feat_memo(clip_output,class_num,unseen_mask_id,image_feature_raw)

    
        reshaped_prototypes = self.ood_som_prototypes.reshape(-1, self.feat_dim)
        non_zero_mask = np.any(reshaped_prototypes != 0, axis=1)
        non_zero_prototypes = reshaped_prototypes[non_zero_mask]
        self.ood_proxy = torch.tensor(non_zero_prototypes, dtype=torch.float32).cuda()
        
        unseen_mask,unseen_mask_ood,unseen_mask_id, visual_output,ood_score_vis = super().get_unseen_mask_visual_memo(clip_output, image, image_feature_raw, self.id_feature_queue , self.ood_proxy, step, target,self.feat_memory)


        self.model.eval()
        self.ood_net.train()
        with torch.enable_grad():   
            if self.args.inference.batch_size == 1:
                ood_detector_out = self.ood_net(image_feature_raw)
                self.ttda_queue.extend(ood_detector_out)
                self.queues['ood_detector_out_queue'].append(ood_detector_out)
                self.queues['unseen_mask_queue'].append(unseen_mask_ood)
                self.queues['target_queue'].append(target)
                self.queues['clip_output_queue'].append(clip_output)
                if step != 0 and step % self.args.inference.ttda_queue_length == 0:                    
                    batch_ood_detector_out = torch.stack(list(self.queues['ood_detector_out_queue']), dim=0).squeeze(1)
                    batch_unseen_mask = torch.stack(list(self.queues['unseen_mask_queue']), dim=0).squeeze(1)
                    batch_target = torch.stack(list(self.queues['target_queue']), dim=0).squeeze(1)
                    batch_clip = torch.stack(list(self.queues['clip_output_queue']), dim=0).squeeze(1)

                    self.update_detector(batch_ood_detector_out, batch_unseen_mask, batch_target, step, batch_clip)
            else:
                ood_detector_out = self.ood_net(image_feature_raw)
                self.update_detector(ood_detector_out, unseen_mask_ood,unseen_mask_id,target, step, filter_ids_1[0] ,filter_ids_2[0], clip_output)

        if self.args.inference.batch_size == 1:
            ttda_queue_length = self.args.inference.ttda_queue_length
        else:
            ttda_queue_length = self.args.inference.batch_size

        if step * self.args.inference.batch_size > self.args.inference.using_ttda_step * ttda_queue_length:

            predict = F.softmax(ood_detector_out, 1) # [bs, 2]
            ood_score = predict[:, 0]
            self.os_detector_queue.extend(ood_score.detach().cpu().tolist())
            self.os_detector_queue = self.os_detector_queue[-self.args.inference.queue_length:]
            if self.args.inference.threshold_type == 'adaptive':
                threshold_range = np.arange(0, 1, 0.01)
                criterias = [compute_os_variance(np.array(self.os_detector_queue), th) for th in threshold_range]
                best_threshold = threshold_range[np.argmin(criterias)]
            else:
                best_threshold = self.args.inference.fixed_threshold

            # ood_score += 0.01*sim
            unseen_mask = (ood_score > best_threshold)

            conf = predict[:, 1]


            return unseen_mask, conf, visual_output
        
        return unseen_mask, visual_output, plpd, entropys, target
    
    
    def update_detector(self, ood_detector_out, unseen_mask_ood,unseen_mask_id,  target, step,filter_ids_1, filter_ids_2,clip_output=None,):
        logit = F.softmax(clip_output, dim=1)
        conf, _ = logit.max(1) # [bs] 

        unseen_mask_ood[target == -1000] = True
        unseen_mask_id[target == -1000] = False

        # 确保是 tensor 并去重
        f1 = filter_ids_1.unique() if torch.is_tensor(filter_ids_1) else torch.tensor(filter_ids_1).unique()
        f2 = filter_ids_2.unique() if torch.is_tensor(filter_ids_2) else torch.tensor(filter_ids_2).unique()

        # 求交集
        common_indices = self.intersect1d(f1, f2)
        print(common_indices)
        # 修改 mask
        unseen_mask_ood[common_indices] = False
        unseen_mask_id[common_indices] = False

        select_ood_indices = torch.where(unseen_mask_ood)[0]
        select_id_indices = torch.where(unseen_mask_id)[0]
        selected_indices = torch.cat([select_id_indices, select_ood_indices])
        selected_labels = torch.cat([torch.ones(len(select_id_indices)), torch.zeros(len(select_ood_indices))]).cuda()

        loss = self.criterion(ood_detector_out[selected_indices], selected_labels.long())
        self.optimizer.zero_grad()
        loss.backward()
        self.loss_history.append(loss.item())
        self.optimizer.step()



    def intersect1d(self,tensor1, tensor2):
        mask = torch.isin(tensor1, tensor2)
        return tensor1[mask].unique()






class ZeroShotNTTA_DEYO_gap(ZeroShotCLIP):
    def __init__(self, *args, **kwargs):
        super(ZeroShotNTTA_DEYO_gap, self).__init__(*args, **kwargs)

        if self.args.model.arch == "ViT-L/14":
            self.feat_dim = 768
        elif self.args.model.arch == "ViT-B/16":
            self.feat_dim = 512
        else:
            self.feat_dim = 1024
        self.ood_net = OODDetector(self.feat_dim).to(self.model.image_encoder.conv1.weight.device)
        self.ood_net = utils.weight_norm(self.ood_net.fc, name="weight")  

        self.criterion = nn.CrossEntropyLoss() 

        self.optimizer = get_optimizer(self.args, self.ood_net.parameters(), lr=self.args.optim.lr)
        #self.optimizer = SAMSGD(self.ood_net.parameters(), lr=0.1, rho=0.05)

        self.os_detector_queue = []
        self.loss_history = []

        self.ood_feature_queue = []

        self.id_feature_queue = []

        
        queue_length = self.args.inference.ttda_queue_length
        self.queues = {
            'ood_detector_out_queue': deque(maxlen=queue_length),
            'unseen_mask_queue': deque(maxlen=queue_length),
            'target_queue': deque(maxlen=queue_length),
            'clip_output_queue': deque(maxlen=queue_length)
        }
        self.ttda_queue = []

        ### no test time adaption!!!!
        self.text_features = self.model.get_text_features()
        ### memo para
        self.memory_size = self.args.training.cache_len
        self.set_id_memory_bank()

       

        ### som 只有numpy 后续改成tensor
        self.size = self.args.training.size   ###   经验公式：决定输出层尺寸 math.ceil(np.sqrt(5 * np.sqrt(N)))  
        self.som_lr = self.args.training.som_lr
        self.som_sigma = self.args.training.som_sigma
        self.som = TorchMiniSom(self.size, self.size, self.feat_dim, self.som_sigma, self.som_lr, 
              neighborhood_function='gaussian',random_seed = 0,device = self.args.gpu)
        self.start_iter = -1  ## get enough sample to warmup
        self.start_queue = []  ## save sample 
        self.first_train_iter = 5000 ### train with start queue sample
        self.iter = self.args.training.som_iter

        self.count_winner = torch.zeros((self.size,self.size)).cuda()
        self.ood_som_prototypes = torch.zeros((self.size,self.size,self.feat_dim)).cuda()


        self.gap = self.args.training.gap

        self.aug_type = 'patch'
        self.filter_ent = self.args.training.filter_ent
        self.filter_plpd = self.args.training.filter_plpd

        self.plpd_threshold = self.args.training.plpd_threshold
        self.deyo_margin = self.args.training.deyo_margin
        self.patch_len = self.args.training.patch_size
        print('plpd:' ,self.plpd_threshold)
        print('deyo', self.deyo_margin)
        print('som_size:',self.size)




    def set_id_memory_bank(self):
        ## reset feat memory to vanilla text features, for each ID/OOD pair.
        text_features = self.text_features
        self.feat_memory = text_features.unsqueeze(1)  ## 1000 * 1 *512
        extented_empty_memory = torch.zeros_like(self.feat_memory).repeat(1, self.memory_size, 1)
        self.feat_memory = torch.cat((self.feat_memory, extented_empty_memory), dim=1)
        self.indice_memory = torch.ones(text_features.size(0))  ### 1000
        self.entropy_memory = torch.zeros(self.feat_memory.size(0), self.feat_memory.size(1)).to(
            text_features.device)  ## 1000*(1+memory_size)


    def insert_feat_memo(self,clip_output,class_num,unseen_mask_id,image_features):
        _, pred_all = torch.max(clip_output[:, :class_num], dim=1)
        prob_id = torch.softmax(clip_output[:, :class_num], dim=1)
        for i in range(unseen_mask_id.size(0)):
            if unseen_mask_id[i].item():
                predicted_cate = pred_all[i].item()
                predicted_prob = prob_id[i]
                current_instance_entropy = -(predicted_prob * (torch.log(predicted_prob + 1e-8))).sum()
                # pdb.set_trace()
                # self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                if self.indice_memory[predicted_cate] == self.memory_size:
                    if (current_instance_entropy < self.entropy_memory[predicted_cate]).sum() == 0:
                        pass  ## the entropy of current test image is very large.
                    else:
                        # replace the one with the maximum entropy!! to update. find the one with the maximum entropy.
                        _, indice = torch.sort(self.entropy_memory[predicted_cate])
                        to_replace_indice = indice[-1]  ## with max entropy, ascending.
                        self.feat_memory[predicted_cate][to_replace_indice] = image_features[i]
                        self.entropy_memory[predicted_cate][to_replace_indice] = current_instance_entropy
                else:
                    self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                    self.entropy_memory[predicted_cate][
                        self.indice_memory[predicted_cate].long()] = current_instance_entropy
                    self.indice_memory[predicted_cate] += 1
            else:
                pass



    def get_unseen_mask(self, clip_output, image, image_feature_raw, step, target):
        class_num = self.class_num
        outputs = clip_output
         # clip OUTPUT：MCM ouput
         # 计算ent

        entropys = softmax_entropy(outputs)
        if self.filter_ent:
            filter_ids_1 = torch.where(entropys > self.deyo_margin)
        else:    
            filter_ids_1 = torch.where((entropys >= math.log(1000)))

        x = image.clone()
        x_prime = x
        x_prime = x_prime.detach()
        ## patch
        if self.aug_type == 'patch':
            resize_t = torchvision.transforms.Resize(((x.shape[-1]//self.patch_len)*self.patch_len,(x.shape[-1]//self.patch_len)*self.patch_len))
            resize_o = torchvision.transforms.Resize((x.shape[-1],x.shape[-1]))
            x_prime = resize_t(x_prime)
            x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=self.patch_len, ps2=self.patch_len)
            perm_idx = torch.argsort(torch.rand(x_prime.shape[0],x_prime.shape[1]), dim=-1)
            x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1),perm_idx]
            x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=self.patch_len, ps2=self.patch_len)
            x_prime = resize_o(x_prime)

        ## pixel 
        elif self.aug_type == 'pixel':
            x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
            x_prime = x_prime[:,:,torch.randperm(x_prime.shape[-1])]
            x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=x.shape[-1], ps2=x.shape[-1])
        elif self.aug_type == 'gaussian':
            noise = torch.randn_like(x_prime) * 0.5  # 噪声强度可调
            x_prime = x_prime + noise
            x_prime = torch.clamp(x_prime, 0.0, 1.0)  # 假设输入在 [0,1] 范围


        ### MCM
        with torch.no_grad():
            outputs_prime,_ = self.model(x_prime)
        prob_outputs = (outputs * 0.1  ).softmax(1)
        prob_outputs_prime = (outputs_prime * 0.1 ).softmax(1)
        cls1 = prob_outputs.argmax(dim=1)
        plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1,1))
        plpd = plpd.reshape(-1)
    
        if self.filter_plpd:
            filter_ids_2 = torch.where((plpd < self.plpd_threshold))
        else:
            filter_ids_2 = torch.where(plpd <= -2.0)
        # entropys = entropys[filter_ids_2]
        # plpd = plpd[filter_ids_2]



        unseen_mask,unseen_mask_ood, unseen_mask_id, clip_output,ood_score = super().get_unseen_mask(clip_output, image, image_feature_raw, step, target)
 

 
        tf = image_feature_raw[unseen_mask_ood].cuda()
        self.som.train_batch(tf, self.iter)

        winmap = self.som.win_map(tf)              # 现在是 { (i,j): [tensor, tensor, ...] }
        activation = self.som.activation_response(tf)  # torch tensor

        for key, val in winmap.items():
            i, j = key
            if len(val) == 0:
                continue
            tmp = torch.stack(val, dim=0)   # (N, feat_dim)
            tmp_ = tmp.sum(dim=0)
            number = activation[i, j]
            if number > 0:
                self.ood_som_prototypes[i, j] = (
                    (self.ood_som_prototypes[i, j] * self.count_winner[i, j]) + tmp_
                ) / (number + self.count_winner[i, j])

        self.count_winner += activation


        self.insert_feat_memo(clip_output,class_num,unseen_mask_id,image_feature_raw)

    
        reshaped_prototypes = self.ood_som_prototypes.view(-1, self.feat_dim)
        non_zero_mask = torch.any(reshaped_prototypes != 0, dim=1)
        non_zero_prototypes = reshaped_prototypes[non_zero_mask]
        self.ood_proxy = non_zero_prototypes.to(self.args.gpu)




        # self.image_feature.append(image_feature_raw)
        # self.target.append(target)
        # if step>870:
        #     np.save('TSNE/som_proto.npy',non_zero_prototypes.cpu().numpy())
        #     sa_text_features = self.feat_memory.mean(1)
        #     sa_text_features /= sa_text_features.norm(dim=-1, keepdim=True)
        #     np.save('TSNE/cache_proto.npy',sa_text_features.cpu().numpy())
        #     tmp = torch.cat(self.image_feature).cpu().numpy()
        #     np.save('TSNE/image_sun.npy',tmp)
        #     tmp = torch.cat(self.target).cpu().numpy()
        #     np.save('TSNE/target.npy',tmp)



        if self.gap:
            if step * self.args.inference.batch_size > self.args.inference.using_ttda_step * self.args.inference.batch_size:
    
                unseen_mask,unseen_mask_ood,unseen_mask_id, visual_output,ood_score_vis = super().get_unseen_mask_visual_memo_gap(clip_output, image, image_feature_raw, self.id_feature_queue , self.ood_proxy, step, target,self.feat_memory)
            else:
                unseen_mask_,unseen_mask_ood,unseen_mask_id, visual_output,ood_score_vis = super().get_unseen_mask_visual_memo_gap(clip_output, image, image_feature_raw, self.id_feature_queue , self.ood_proxy, step, target,self.feat_memory)
        else:
            if step * self.args.inference.batch_size > self.args.inference.using_ttda_step * self.args.inference.batch_size:
    
                unseen_mask,unseen_mask_ood,unseen_mask_id, visual_output,ood_score_vis = super().get_unseen_mask_visual_memo(clip_output, image, image_feature_raw, self.id_feature_queue , self.ood_proxy, step, target,self.feat_memory)
            else:
                unseen_mask_,unseen_mask_ood,unseen_mask_id, visual_output,ood_score_vis = super().get_unseen_mask_visual_memo(clip_output, image, image_feature_raw, self.id_feature_queue , self.ood_proxy, step, target,self.feat_memory)


        self.model.eval()
        self.ood_net.train()
        with torch.enable_grad():   
            if self.args.inference.batch_size == 1:
                ood_detector_out = self.ood_net(image_feature_raw)
                self.ttda_queue.extend(ood_detector_out)
                self.queues['ood_detector_out_queue'].append(ood_detector_out)
                self.queues['unseen_mask_queue'].append(unseen_mask_ood)
                self.queues['target_queue'].append(target)
                self.queues['clip_output_queue'].append(clip_output)
                if step != 0 and step % self.args.inference.ttda_queue_length == 0:                    
                    batch_ood_detector_out = torch.stack(list(self.queues['ood_detector_out_queue']), dim=0).squeeze(1)
                    batch_unseen_mask = torch.stack(list(self.queues['unseen_mask_queue']), dim=0).squeeze(1)
                    batch_target = torch.stack(list(self.queues['target_queue']), dim=0).squeeze(1)
                    batch_clip = torch.stack(list(self.queues['clip_output_queue']), dim=0).squeeze(1)

                    self.update_detector(batch_ood_detector_out, batch_unseen_mask, batch_target, step, batch_clip)
            else:
                ood_detector_out = self.ood_net(image_feature_raw)
                self.update_detector(ood_detector_out, unseen_mask_ood,unseen_mask_id,target, step, filter_ids_1[0] ,filter_ids_2[0], clip_output)

        if self.args.inference.batch_size == 1:
            ttda_queue_length = self.args.inference.ttda_queue_length
        else:
            ttda_queue_length = self.args.inference.batch_size

        if step * self.args.inference.batch_size > self.args.inference.using_ttda_step * ttda_queue_length:

            predict = F.softmax(ood_detector_out, 1) # [bs, 2]
            ood_score = predict[:, 0]
            self.os_detector_queue.extend(ood_score.detach().cpu().tolist())
            self.os_detector_queue = self.os_detector_queue[-self.args.inference.queue_length:]
            if self.args.inference.threshold_type == 'adaptive':
                threshold_range = np.arange(0, 1, 0.01)
                criterias = [compute_os_variance(np.array(self.os_detector_queue), th) for th in threshold_range]
                best_threshold = threshold_range[np.argmin(criterias)]
            else:
                best_threshold = self.args.inference.fixed_threshold

            # ood_score += 0.01*sim
            unseen_mask = (ood_score > best_threshold)

            conf = predict[:, 1]


            return unseen_mask, conf, visual_output
        
        return unseen_mask, visual_output, plpd, entropys, target
    
    
    def update_detector(self, ood_detector_out, unseen_mask_ood,unseen_mask_id,  target, step,filter_ids_1, filter_ids_2,clip_output=None,):
        logit = F.softmax(clip_output, dim=1)
        conf, _ = logit.max(1) # [bs] 

        unseen_mask_ood[target == -1000] = True
        unseen_mask_id[target == -1000] = False

        # 确保是 tensor 并去重
        f1 = filter_ids_1.unique() if torch.is_tensor(filter_ids_1) else torch.tensor(filter_ids_1).unique()
        f2 = filter_ids_2.unique() if torch.is_tensor(filter_ids_2) else torch.tensor(filter_ids_2).unique()

        # 求交集
        common_indices = self.intersect1d(f1, f2)
        print(common_indices)
        # 修改 mask
        unseen_mask_ood[common_indices] = False
        unseen_mask_id[common_indices] = False

        select_ood_indices = torch.where(unseen_mask_ood)[0]
        select_id_indices = torch.where(unseen_mask_id)[0]
        selected_indices = torch.cat([select_id_indices, select_ood_indices])
        selected_labels = torch.cat([torch.ones(len(select_id_indices)), torch.zeros(len(select_ood_indices))]).cuda()

        loss = self.criterion(ood_detector_out[selected_indices], selected_labels.long())
        self.optimizer.zero_grad()
        loss.backward()
        self.loss_history.append(loss.item())
        self.optimizer.step()



    def intersect1d(self,tensor1, tensor2):
        mask = torch.isin(tensor1, tensor2)
        return tensor1[mask].unique()
