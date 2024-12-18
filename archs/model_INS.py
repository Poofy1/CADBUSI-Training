import torch
import torch.nn as nn
from fastai.vision.all import *
import torch.nn.functional as F
from archs.backbone import create_timm_body
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from loss.IWSCL import *
from archs.linear_classifier import *

class Embeddingmodel(nn.Module):
    def __init__(self, arch, pretrained_arch, num_classes=1, feat_dim=128, momentum=0.999, queue_size=8192):
        super(Embeddingmodel, self).__init__()
        
        self.num_classes = num_classes
        self.nf = 512
        self.momentum = momentum

        # Original encoder
        self.encoder_q = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        num_features = self.encoder_q.classifier[1].in_features
        self.encoder_q.classifier[1] = nn.Linear(num_features, self.nf)
        
        # Momentum encoder
        self.encoder_k = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.encoder_k.classifier[1] = nn.Linear(num_features, self.nf)
        
        self.aggregator = Linear_Classifier(nf=self.nf, num_classes=num_classes)
        
        # Original projector
        self.projector_q = nn.Sequential(
            nn.Linear(self.nf, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # dropout here
            nn.Linear(512, feat_dim)
        )
        
        # Momentum projector
        self.projector_k = nn.Sequential(
            nn.Linear(self.nf, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feat_dim)
        )
        
        self.ins_classifier = nn.Sequential(
            nn.Linear(self.nf, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )
        
        # Initialize the momentum encoder and projector
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Initialize the queue
        self.register_buffer("queue", torch.randn(feat_dim, queue_size))
        self.register_buffer("queue_labels", torch.zeros(queue_size).long())
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_size = queue_size
        
        # Add IWSCL
        self.iwscl = IWSCL(feat_dim=feat_dim, momentum=momentum)
    
    def get_queue(self):
        return self.queue.clone().detach(), self.queue_labels.clone().detach()
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # Replace the keys and labels at ptr (dequeue and enqueue)
        if ptr + batch_size > self.queue_size:
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
            self.queue_labels[ptr:] = labels[:remaining]
            self.queue_labels[:batch_size - remaining] = labels[remaining:]
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            self.queue_labels[ptr:ptr + batch_size] = labels
        
        ptr = (ptr + batch_size) % self.queue_size  # Move pointer
        self.queue_ptr[0] = ptr
            
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder and projector
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
        
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    def forward(self, img_q_input, im_k=None, true_label = None, projector=False, bag_on=False, val_on=False):
        if bag_on:
            num_bags = len(img_q_input)
            img_q = torch.cat(img_q_input, dim=0).cuda()
        else: 
            img_q = img_q_input

        # Calculate the embeddings for all images in one go
        feat_q = self.encoder_q(img_q)
        instance_predictions = self.ins_classifier(feat_q)

        bag_pred = None
        bag_instance_predictions = None
        if bag_on:
            # Split the embeddings back into per-bag embeddings
            split_sizes = [bag.size(0) for bag in img_q_input]
            h_per_bag = torch.split(feat_q, split_sizes, dim=0)
            y_hat_per_bag = torch.split(instance_predictions, split_sizes, dim=0)
            bag_pred = torch.empty(num_bags, self.num_classes).cuda()
            bag_instance_predictions = []
            for i, (h, y_h) in enumerate(zip(h_per_bag, y_hat_per_bag)):
                yhat_bag, yhat_ins = self.aggregator(h, y_h)
                bag_pred[i] = yhat_bag
                bag_instance_predictions.append(yhat_ins) 
        
        proj_q = None
        proj_k = None
        iwscl_loss = torch.tensor(0.0).cuda()
        pseudo_labels = None
        if projector:
            proj_q = self.projector_q(feat_q)
            proj_q = F.normalize(proj_q, dim=1)
            
            
            # 1. Calculate IWSCL loss using current queue state
            iwscl_loss, pseudo_labels = self.iwscl(proj_q, instance_predictions, true_label, 
                                                  self.queue, self.queue_labels, val_on)
            
            if im_k is not None and not val_on:
                # Momentum update
                self._momentum_update_key_encoder()
                
                # Compute key features
                with torch.no_grad():
                    feat_k = self.encoder_k(im_k)
                    proj_k = self.projector_k(feat_k)
                    proj_k = F.normalize(proj_k, dim=1)
                    
                # Modify the enqueue process
                if true_label is not None:
                    # Create a mask for known labels
                    known_mask = true_label != -1
                    
                    # Initialize labels with classifier predictions
                    enqueue_labels = instance_predictions.argmax(dim=1)
                    
                    # Replace known labels with ground truth
                    enqueue_labels[known_mask] = true_label[known_mask]
                else:
                    # If no true_label is provided, use classifier predictions for all
                    enqueue_labels = instance_predictions.argmax(dim=1)

                # Enqueue and dequeue
                self._dequeue_and_enqueue(proj_k, enqueue_labels)
                
        return bag_pred, bag_instance_predictions, instance_predictions.squeeze(), proj_q, iwscl_loss, pseudo_labels

