import torch
from util.format_data import *
from util.sudo_labels import *

class GenSupConLossv2(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(GenSupConLossv2, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels, anc_mask = None):
        '''
        Args:
            feats: (anchor_features, contrast_features), each: [N, feat_dim]
            labels: (anchor_labels, contrast_labels) each: [N, num_cls]
            anc_mask: (anchors_mask, contrast_mask) each: [N]
        '''

        anchor_labels = torch.cat(labels, dim=0).float()
        contrast_labels = anchor_labels
        anchor_features = torch.cat(features, dim=0)
        contrast_features = anchor_features
        
        # 1. compute similarities among targets
        anchor_norm = torch.norm(anchor_labels, p=2, dim=-1, keepdim=True) # [anchor_N, 1]
        contrast_norm = torch.norm(contrast_labels, p=2, dim=-1, keepdim=True) # [contrast_N, 1]
        deno = torch.mm(anchor_norm, contrast_norm.T)
        mask = torch.mm(anchor_labels, contrast_labels.T) / deno # cosine similarity: [anchor_N, contrast_N]
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask
        
        # 2. compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_features, contrast_features.T),
            self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        if anc_mask:
            select = torch.cat( anc_mask )
            loss = loss[select]
            
        loss = loss.mean()

        return loss



class GenSupConLossv2_Queue(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07, queue_size=1024):
        super(GenSupConLossv2_Queue, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.queue_size = queue_size
        self.queue = None
        self.queue_labels = None

    # initalize larger queue with black negitive images?
    
    def _dequeue_and_enqueue(self, features, labels):
        if self.queue is None:
            self.queue = features
            self.queue_labels = labels
        else:
            self.queue = torch.cat((self.queue, features), dim=0)
            self.queue_labels = torch.cat((self.queue_labels, labels), dim=0)
            
        if self.queue.shape[0] > self.queue_size:
            self.queue = self.queue[-self.queue_size:]
            self.queue_labels = self.queue_labels[-self.queue_size:]

    def forward(self, features, labels, anc_mask=None):
        '''
        Args:
            feats: (anchor_features, contrast_features), each: [N, feat_dim]
            labels: (anchor_labels, contrast_labels) each: [N, num_cls]
            anc_mask: (anchors_mask, contrast_mask) each: [N]
        '''
        anchor_features = torch.cat(features, dim=0)
        anchor_labels = torch.cat(labels, dim=0).float()

        # Concatenate anchor features and queue
        contrast_features = torch.cat([anchor_features, self.queue], dim=0) if self.queue is not None else anchor_features
        contrast_labels = torch.cat([anchor_labels, self.queue_labels], dim=0) if self.queue is not None else anchor_labels
        
        # 1. compute similarities among targets
        anchor_norm = torch.norm(anchor_labels, p=2, dim=-1, keepdim=True)  # [anchor_N, 1]
        contrast_norm = torch.norm(contrast_labels, p=2, dim=-1, keepdim=True)  # [contrast_N, 1]
        deno = torch.mm(anchor_norm, contrast_norm.T)
        mask = torch.mm(anchor_labels, contrast_labels.T) / deno  # cosine similarity: [anchor_N, contrast_N]
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        # 2. compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_features, contrast_features.T), 
            self.temperature
        )  
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        if anc_mask:
            select = torch.cat(anc_mask)
            loss = loss[select]
        loss = loss.mean()

        # Update queue
        self._dequeue_and_enqueue(anchor_features.detach(), anchor_labels.detach())

        return loss
    