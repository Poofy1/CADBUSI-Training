import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F
import math

class AlexNet_CIFAR10_attention(nn.Module):
    def __init__(self, features, num_classes, init=True, withoutAtten=False, input_feat_dim=512):
        super(AlexNet_CIFAR10_attention, self).__init__()
        self.input_feat_dim = input_feat_dim
        self.withoutAtten = withoutAtten
        self.features = features
        self.classifier = nn.Sequential(nn.Dropout(0.5),
                            nn.Linear(input_feat_dim, 1024),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(1024, 1024),
                            nn.ReLU(inplace=True))
        self.L = 1024
        self.D = 512
        self.K = 1

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.headcount = len(num_classes)
        self.return_features = False
        if len(num_classes) == 1:
            self.top_layer = nn.Linear(1024, num_classes[0])
        else:
            for a,i in enumerate(num_classes):
                setattr(self, "top_layer%d" % a, nn.Linear(4096, i))
            self.top_layer = None  # this way headcount can act as switch.
        if init:
            self._initialize_weights()

    def forward(self, x, returnBeforeSoftMaxA=False, scores_replaceAS=None):
        if self.features is not None:
            x = x.squeeze(0)
            x = self.features(x)
        # print('x.shape',x.shape)
        # print('x.size',x.size(0))
        # print('input_feat_dim',self.input_feat_dim)
        x = x.view(x.size(0), self.input_feat_dim)
        # print('x.device',x.device)
        
        x = self.classifier(x)
        # print(self.classifier.device)
        # Attention module
        A_ = self.attention(x)  # NxK
        A_ = torch.transpose(A_, 1, 0)  # KxN
        A = F.softmax(A_, dim=1)  # softmax over N

        if scores_replaceAS is not None:
            A_ = scores_replaceAS
            A = F.softmax(A_, dim=1)  # softmax over N

        if self.withoutAtten:
            x = torch.mean(x, dim=0, keepdim=True)
        else:
            x = torch.mm(A, x)  # KxL

        if self.return_features: # switch only used for CIFAR-experiments
            return x

        x = self.top_layer(x)
        if returnBeforeSoftMaxA:
            # print('x.shape:',x.shape)
            return x, 0, A, A_
        return x, 0, A

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
def teacher_Attention_head(bn=True, num_classes=[2], init=True, input_feat_dim=512):
    model = AlexNet_CIFAR10_attention(features=None, num_classes=num_classes, init=init, input_feat_dim=input_feat_dim)
    return model


class INS(nn.Module):

    def __init__(self, args, base_encoder):
        super().__init__()
        
        pretrained = args.dataset == 'cub200' ## False in cifar-10
        # we allow pretraining for CUB200, or the network will not converge

        self.encoder_q = base_encoder(num_class=args.num_class, feat_dim=args.low_dim, name=args.arch, pretrained=pretrained)
        # momentum encoder
        self.encoder_k = base_encoder(num_class=args.num_class, feat_dim=args.low_dim, name=args.arch, pretrained=pretrained)

        # init a bag prediction head: FC layer
        self.bag_pred_head = None

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        self.model_teacherHead = teacher_Attention_head()
        # create the queue
        self.register_buffer("queue", torch.randn(args.moco_queue, args.low_dim))  ## (8192,128)
        self.register_buffer("queue_pseudo", torch.randn(args.moco_queue, args.num_class))  ## (8192,10)  伪标签
        self.register_buffer("queue_partial", torch.randn(args.moco_queue, args.num_class))  ## (8192,10) 偏标签
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))        
        self.register_buffer("prototypes", torch.zeros(args.num_class,args.low_dim))  ## 0(10,128)
        self.bag_classifier = nn.Linear(512, 2)
    @torch.no_grad()
    def _momentum_update_key_encoder(self, args):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, partial_Y, args):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)
        partial_Y = concat_all_gather(partial_Y)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # print('val:',args.moco_queue, batch_size)
        assert args.moco_queue % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_pseudo[ptr:ptr + batch_size, :] = labels
        self.queue_partial[ptr:ptr + batch_size, :] = partial_Y
        ptr = (ptr + batch_size) % args.moco_queue  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x, y, p_y):
        return x, y, p_y, None
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        y_gather = concat_all_gather(y)
        p_y_gather = concat_all_gather(p_y)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], y_gather[idx_this], p_y_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, y, p_y, idx_unshuffle):
        return x, y, p_y
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        y_gather = concat_all_gather(y)
        p_y_gather = concat_all_gather(p_y)

        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], y_gather[idx_this], p_y_gather[idx_this]

    def reset_prototypes(self, prototypes):
        self.prototypes = prototypes
        
    def bag_head(self, features):
        # mean-pooling attention-pooling
        bag_prediction, _, _, _ = self.model_teacherHead(features, returnBeforeSoftMaxA=True, scores_replaceAS=None)
        # print('bag_prediction:',bag_prediction)
        return bag_prediction
        # return features.mean(dim=0, keepdim=True)
    
    def forward(self, img_q, im_k=None, partial_Y=None, args=None, eval_only=False, bag_flag=False):
        if bag_flag:
            # print('img_q.shape:',img_q.shape) # 100*3*32*32
            # code for cal bag predictions, input is bag ()
            # features = encoder(all instance from a bag)
            features = self.encoder_q.encoder(img_q) # 100*512
            bag_prediction = self.bag_head(features)   # 512
        
            # or Max-pooling, Attention-pooling
            # bag_prediction = self.bag_classifier(bag_feature)
            # print(bag_prediction.shape, bag_prediction)
            return bag_prediction
            # return 0
        else:
            # 先提取q的特征，然后再提取im_k的特征时先声明不进行梯度更新，先进行momentum更新关键字网络，
            # 然后对于im_k的索引进行shuffle，然后再提取特征，提取完特征之后又恢复了特征k的顺序（unshuffle_ddp），
            # 因为q的顺序没有打乱，在计算损失时需要对应。

            # output是classifier的输出结果，q是MLP的输出结果
            # 两个样本的预测标签相同，则他们为正样本对，反之则为负样本对
            # if point == 0: 相当于是个mask ，每次都要对结果mask
            output, q = self.encoder_q(img_q)  ##([256,10]),([256,128])
            if eval_only:
                return output
            # for testing
            # 分类器的预测结果：torch.softmax(output, dim=1)
            # 所以病理的图片标签应该是 pos,neg 都为1
            predicted_scores = torch.softmax(output, dim=1) * partial_Y # 分类器结果，限制标签在候选集y当中
            max_scores, pseudo_labels = torch.max(predicted_scores, dim=1) ## values, index（预测标签）
            # using partial labels to filter out negative labels

            # compute protoypical logits
            prototypes = self.prototypes.clone().detach()  ## 0([10,128])
            logits_prot = torch.mm(q, prototypes.t())  ##([256,10])
            score_prot = torch.softmax(logits_prot, dim=1) ##([256,10]) # 成绩是平均的

            # update momentum prototypes with pseudo labels
            for feat, label in zip(concat_all_gather(q), concat_all_gather(pseudo_labels)): #concat_all_gather将多卡数据合并
                self.prototypes[label] = self.prototypes[label]*args.proto_m + (1-args.proto_m)*feat  ## 按feature_c更新prototypes
                # torch.set_printoptions(profile="full")
                # print((1-args.proto_m)*feat)
            # normalize prototypes
            self.prototypes = F.normalize(self.prototypes, p=2, dim=1)  ##(10,128)
            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder(args)  # update the momentum encoder with encoder_q
                # shuffle for making use of BN
                im_k, predicted_scores, partial_Y, idx_unshuffle = self._batch_shuffle_ddp(im_k, predicted_scores, partial_Y)
                _, k = self.encoder_k(im_k)  ## 输出k样本预测类别
                # print('img_k',im_k.shape,k.shape)
                # undo shuffle
                k, predicted_scores, partial_Y = self._batch_unshuffle_ddp(k, predicted_scores, partial_Y, idx_unshuffle)

            features = torch.cat((q, k, self.queue.clone().detach()), dim=0)  ##
            pseudo_scores = torch.cat((predicted_scores, predicted_scores, self.queue_pseudo.clone().detach()), dim=0)
            partial_target = torch.cat((partial_Y, partial_Y, self.queue_partial.clone().detach()), dim=0)
            # to calculate SupCon Loss using pseudo_labels and partial target

            # dequeue and enqueue
            self._dequeue_and_enqueue(k, predicted_scores, partial_Y, args)
            # 分类器预测结果，伪标签，原型的结果
            return output, features, pseudo_scores, partial_target, score_prot


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    return tensor # Using only one GPU
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output









import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # print(x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet18', head='mlp', feat_dim=128, num_class=0, pretrained=True):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        if pretrained:
            model = models.resnet18(pretrained=True)
            model.fc = Identity()
            self.encoder = model
            # Note: torchvision pretrained model is slightly different from ours, 
            # when training CUB, using torchvision model will be more memory efficient
        else:
            self.encoder = model_fun()
        # fc是classifier
        self.fc = nn.Linear(dim_in, num_class)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))     
        self.register_buffer("prototypes", torch.zeros(num_class, feat_dim))
        # self.encoder=nn.Sequential(
        #         nn.Linear(512,512),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(512,512)
        #     )
        # self.encoder = nn.Linear(512, 512)
        # self.encoder = nn.Linear(384,512)
        
    def forward(self, x):
        #print(x.shape)
        feat = self.encoder(x)  ##([256,512])
        feat_c = self.head(feat)  ## ([256,128]) 
        logits = self.fc(feat)  ## ([256,10]) 
        return logits, F.normalize(feat_c, dim=1)