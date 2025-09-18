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

from utils.torchsom import TorchMiniSom
import math
import numpy as np
import torchvision
from typing import Iterable

import torch
from torch.optim._multi_tensor import SGD

import torch.nn.utils as utils

def _l2_normalize(x, dim=-1, eps=1e-12):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

class ZeroShotNTTA_Syde(ZeroShotCLIP):
    def __init__(self, *args, **kwargs):
        super(ZeroShotNTTA_Syde, self).__init__(*args, **kwargs)
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
        self.os_detector_queue = []
        self.loss_history = []

        self.text_features = self.model.get_text_features()
        ### memo para
        self.memory_size = self.args.training.cache_len
        self.set_id_memory_bank()
        self.size = self.args.training.size   
        self.som_lr = self.args.training.som_lr
        self.som_sigma = self.args.training.som_sigma
        self.som = TorchMiniSom(self.size, self.size, self.feat_dim, self.som_sigma, self.som_lr, 
              neighborhood_function='gaussian',random_seed = 0,device = self.args.gpu)
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
                if self.indice_memory[predicted_cate] == self.memory_size:
                    if (current_instance_entropy < self.entropy_memory[predicted_cate]).sum() == 0:
                        pass  
                    else:
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
            noise = torch.randn_like(x_prime) * 0.5  
            x_prime = x_prime + noise
            x_prime = torch.clamp(x_prime, 0.0, 1.0)  
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
        unseen_mask,unseen_mask_ood, unseen_mask_id, clip_output,ood_score = super().get_unseen_mask(clip_output, image, image_feature_raw, step, target)
        tf = image_feature_raw[unseen_mask_ood].cuda()
        self.som.train_batch(tf, self.iter)
        winmap = self.som.win_map(tf)              
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
        if step> self.args.inference.using_ttda_step:
            unseen_mask,unseen_mask_ood,unseen_mask_id, visual_output = super().get_unseen_mask_visual_memo_gap(clip_output, image, self.ood_proxy, self.feat_memory)
        else:
            unseen_mask_,unseen_mask_ood,unseen_mask_id, visual_output = super().get_unseen_mask_visual_memo_gap(clip_output, image,self.ood_proxy, self.feat_memory)
    
        self.model.eval()
        self.ood_net.train()
        with torch.enable_grad():   

            ood_detector_out = self.ood_net(image_feature_raw)
            self.update_detector(ood_detector_out, unseen_mask_ood,unseen_mask_id,target, step, filter_ids_1[0] ,filter_ids_2[0], clip_output)
    
        if step  > self.args.inference.using_ttda_step :
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
        return unseen_mask, visual_output, plpd, entropys, target
    
    def update_detector(self, ood_detector_out, unseen_mask_ood,unseen_mask_id,  target, step,filter_ids_1, filter_ids_2,clip_output=None,):
        logit = F.softmax(clip_output, dim=1)
        conf, _ = logit.max(1) # [bs] 
        unseen_mask_ood[target == -1000] = True
        unseen_mask_id[target == -1000] = False
        f1 = filter_ids_1.unique() if torch.is_tensor(filter_ids_1) else torch.tensor(filter_ids_1).unique()
        f2 = filter_ids_2.unique() if torch.is_tensor(filter_ids_2) else torch.tensor(filter_ids_2).unique()
        common_indices = self.intersect1d(f1, f2)
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
