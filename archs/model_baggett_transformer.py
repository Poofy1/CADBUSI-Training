import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# class CNNBackboneModel(nn.Module):
#     """
#     A flexible and configurable CNN backbone model that supports various architectures including ResNet, EfficientNet, and MobileNet.
#     This model can function as a standard classifier, a feature extractor, or output raw feature maps.

#     Parameters:
#     - outputs (int): The number of classes for classification. Special cases:
#         - If outputs = -1, the model outputs the last layer of feature maps.
#         - If outputs = 0, the model acts as a feature extractor, outputting a flat feature vector.
#         - If outputs > 0, a classifier head is added to output logits for the given number of outputs (classes).
#     - model_choice (str): Model architecture. Currently available choices are:
#         - 'efficientnet_b0', 'efficientnet_b1', ..., 'efficientnet_b7',
#         - 'mobilenet_v2'
#         - 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50'
#     - pretrained (bool): If True, initializes with pretrained weights.
#     """
#     def __init__(self, outputs, model_choice='resnet18', pretrained=True):
#         super(CNNBackboneModel, self).__init__()
#         self.outputs = outputs
#         self.model_choice = model_choice
#         self.pretrained = pretrained

#         efficientnets = [f'efficientnet_b{i}' for i in range(8)]
#         resnets = [f'resnet{i}' for i in [18, 34, 50, 101, 152]]
#         mobilenets = [f'mobilenet_{s}' for s in ['v2', 'v3_small', 'v3_large']]
#         self.available_models = efficientnets + resnets + mobilenets
        
#         self.model = self.initialize_model()
#         self.adjust_classifier()

#     def initialize_model(self):
#         # Determine the model and initialize it
#         if self.model_choice in self.available_models:
#             weights = "DEFAULT" if self.pretrained else None
#             if 'efficientnet' in self.model_choice:
#                 return models.__dict__[self.model_choice](weights=weights)
#             elif 'resnet' in self.model_choice:
#                 return models.__dict__[self.model_choice](weights=weights)
#             elif 'mobilenet' in self.model_choice:
#                 return models.__dict__[self.model_choice](weights=weights)
#             else:
#                 raise ValueError(f"Unsupported model choice: {self.model_choice}")
#         else:
#             raise ValueError(f"Unsupported model choice: {self.model_choice}")

#     def adjust_classifier(self):
#         # Adjust the classifier or network structure based on the number of classes
#         if 'resnet' in self.model_choice:
#             self.num_features = self.model.fc.in_features
#             self.model.fc = self.get_final_layer()
#         elif 'efficientnet' in self.model_choice:
#             self.num_features = self.model.classifier[1].in_features
#             self.model.classifier[1] = self.get_final_layer()
#         elif 'mobilenet_v2' in self.model_choice:
#             self.num_features = self.model.classifier[1].in_features
#             self.model.classifier[1] = self.get_final_layer()

#     def get_final_layer(self):
#         if self.outputs > 0:
#             return nn.Linear(self.num_features, self.outputs)
#         elif self.outputs == 0:
#             return nn.Identity()
#         elif self.outputs == -1:
#             # Remove layers for feature map output
#             return nn.Identity()

#     def forward(self, x):
#         return self.model(x)

class CNNBackboneModel(nn.Module):
    """
    A flexible and configurable CNN backbone model that supports various architectures including ResNet, EfficientNet, and MobileNet.
    This model can function as a standard classifier, a feature extractor, or output raw feature maps.

    Parameters:
    - outputs (int): The number of classes for classification. Special cases:
        - If outputs = -1, the model outputs the last layer of feature maps.
        - If outputs = 0, the model acts as a feature extractor, outputting a flat feature vector.
        - If outputs > 0, a classifier head is added to output logits for the given number of outputs (classes).
    - model_choice (str): Model architecture. Currently available choices are:
        - 'efficientnet_b0', 'efficientnet_b1', ..., 'efficientnet_b7',
        - 'mobilenet_v2'
        - 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50'
    - pretrained (bool): If True, initializes with pretrained weights.
    """
    def __init__(self, outputs, model_choice='resnet18', pretrained=True):
        super().__init__()
        self.outputs = outputs
        self.model_choice = model_choice
        self.pretrained = pretrained

        efficientnets = [f'efficientnet_b{i}' for i in range(8)]
        resnets = [f'resnet{i}' for i in [18, 34, 50, 101, 152]]
        mobilenets = [f'mobilenet_{s}' for s in ['v2', 'v3_small', 'v3_large']]
        self.available_models = efficientnets + resnets + mobilenets
        
        self.model = self.initialize_model()
        self.adjust_classifier()

    def initialize_model(self):
        # Determine the model and initialize it
        if self.model_choice in self.available_models:
            weights = "DEFAULT" if self.pretrained else None
            if 'efficientnet' in self.model_choice:
                return models.__dict__[self.model_choice](weights=weights)
            elif 'resnet' in self.model_choice:
                return models.__dict__[self.model_choice](weights=weights)
            elif 'mobilenet' in self.model_choice:
                return models.__dict__[self.model_choice](weights=weights)
            else:
                raise ValueError(f"Unsupported model choice: {self.model_choice}")
        else:
            raise ValueError(f"Unsupported model choice: {self.model_choice}")

    def adjust_classifier(self):
        # Adjust the classifier or network structure based on the number of classes
        if 'resnet' in self.model_choice:
            self.num_features = self.model.fc.in_features
            if self.outputs == -1:
                # Remove the avgpool and fc layers
                self.model = nn.Sequential(*(list(self.model.children())[:-2]))
            else:
                self.model.fc = self.get_final_layer()
        elif 'efficientnet' in self.model_choice:
            self.num_features = self.model.classifier[1].in_features
            if self.outputs == -1:
                # Remove the avgpool and classifier layers
                self.model = nn.Sequential(*(list(self.model.children())[:-2]))
            else:
                self.model.classifier[1] = self.get_final_layer()
        elif 'mobilenet_v2' in self.model_choice:
            self.num_features = self.model.classifier[1].in_features
            if self.outputs == -1:
                # Remove the avgpool and classifier layers
                self.model = nn.Sequential(*(list(self.model.children())[:-2]))
            else:
                self.model.classifier[1] = self.get_final_layer()

    def get_final_layer(self):
        if self.outputs > 0:
            return nn.Linear(self.num_features, self.outputs)
        elif self.outputs == 0:
            return nn.Identity()
        elif self.outputs == -1:
            return nn.Identity()

    def forward(self, x):
        return self.model(x.cuda())

class SimpleCNNFM(nn.Module):
    '''
    A simple CNN for turning 3-channel images into 128 channel feature maps.  Useful for cheaply testing frameworks.
    '''
    def __init__(self, num_classes=10):
        super(SimpleCNNFM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.num_features = 128
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return x

class GeneralMLP(nn.Module):
    def __init__(self, layer_sizes, use_batchnorm=False, leak_rate=0.1, normalize=False):
        """
        Initializes the GeneralMLP class.

        Args:
        layer_sizes (tuple): Tuple of integers specifying the sizes of the input layer,
                             hidden layers, and output layer.
        use_batchnorm (bool): If True, BatchNorm1d layers will be added after each Linear layer.
        leak_rate (float): The negative slope of the LeakyReLU activation function.
        normalize (bool): If True, the output vector will be normalized to have L2 norm 1.
        """
        super(GeneralMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.normalize = normalize
        self.epsilon = 1e-8
        
        # Iterate over the tuple to create layers dynamically
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:  # No activation on the last layer
                if use_batchnorm:
                    self.layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
                self.layers.append(nn.LeakyReLU(negative_slope=leak_rate))

    def forward(self, x):
        """
        Defines the forward pass of the MLP.

        Args:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Output of the MLP.
        """
        for layer in self.layers:
            x = layer(x)
        
        if self.normalize:
            x = F.normalize(x, p=2, dim=1, eps=self.epsilon)
        
        return x

class ListOfBagsProcessor(nn.Module):
    """
    A class to process bags of images through a given model and manage input-output transformations.

    This class accepts a list of tensors and a model.
    It concatenates the list of tensors into a single tensor, applies the model to extract
    features, and then splits the resulting tensor back into the original bag sizes.

    Attributes:
        model (nn.Module): A PyTorch model that accepts a single tensor of images and returns
                           a tensor of feature vectors.
    """
    def __init__(self, model):
        super(ListOfBagsProcessor, self).__init__()
        self.model = model

    def forward(self, bags):
        """
        Processes a list of bags of images through the model.

        Args:
            bags (list of Tensor): Each tensor has shape (bag_size, C, H, W),
                                   representing a bag of images.

        Returns:
            list of Tensor: Each tensor in the list corresponds to the output from the model
                            for each respective bag, with shapes restored to original bag sizes.
        """
        # Record the original sizes of each bag
        original_sizes = [bag.size(0) for bag in bags]

        # Concatenate all bags into a single batch
        concatenated = torch.cat(bags, dim=0)

        # Pass the concatenated tensor through the model
        features = self.model(concatenated)

        # Split the features tensor back into the original bag sizes
        split_features = torch.split(features, original_sizes, dim=0)

        return split_features

class BagOfFeaturesPadder(nn.Module):
    """
    This class takes a list of bags of feature vectors and pads each bag to the maximum bag size
    found in the batch. It returns a tensor of shape (B, S, D), where B is the batch size,
    S is the maximum bag size, and D is the fixed feature dimension.

    The class also returns a mask of size (B, S) that is False where rows are padded and True
    otherwise.
    """
    def __init__(self):
        super().__init__()

    def forward(self, bags):
        """
        Parameters:
            bags (list of Tensor): A list where each element is a tensor of shape (bag_size, D),
                                   representing a bag of feature vectors. `bag_size` can vary for each tensor.
                                   It can also handle 1D tensors, which will be treated as having a single feature dimension.

        Returns:
            Tensor: Padded tensor of shape (B, S, D).
            Tensor: Mask tensor of shape (B, S) indicating valid data (True) and padding (False).
        """
        # Check if input bags are 1D and add a trivial last dimension if needed
        for i, bag in enumerate(bags):
            if bag.dim() == 1:
                bags[i] = bag.unsqueeze(-1)

        # Determine the maximum bag size
        max_size = max(bag.size(0) for bag in bags)

        # Dimension of the feature vectors
        D = bags[0].size(1)

        # Initialize padded tensors and masks
        padded_tensors = []
        masks = []

        for bag in bags:
            # Current bag size
            current_size = bag.size(0)

            # Calculate padding needed
            padding_needed = max_size - current_size

            # Create padding
            if padding_needed > 0:
                padding_tensor = torch.zeros(padding_needed, D, dtype=bag.dtype, device=bag.device)
                padded_tensor = torch.cat([bag, padding_tensor], dim=0)
                mask = torch.cat([torch.ones(current_size, dtype=torch.bool, device=bag.device),
                                  torch.zeros(padding_needed, dtype=torch.bool, device=bag.device)], dim=0)
            else:
                padded_tensor = bag
                mask = torch.ones(current_size, dtype=torch.bool, device=bag.device)

            padded_tensors.append(padded_tensor)
            masks.append(mask)

        # Stack all the padded tensors and masks to get the final tensor and mask
        final_tensor = torch.stack(padded_tensors, dim=0)
        final_mask = torch.stack(masks, dim=0)

        return final_tensor, final_mask

class BagsTransformerEncoder(nn.Module):
    """
    A custom Transformer Encoder class that encapsulates the transformer encoder
    layer suitable for sequence processing tasks with batch-first input. This class
    can be used for sequence-to-sequence models, where the input consists of batches
    of sequences and accompanying padding masks.

    Parameters:
        D (int): The size of the input feature dimension (embedding dimension).
        nhead (int): The number of attention heads in the multihead attention models.
        num_layers (int): The number of sub-encoder-layers in the transformer.
        dim_feedforward (int): The dimension of the feedforward network model in the encoder. Default: 128
        dropout (float): The dropout value. Default: 0.1

        dim_feedforward is often 2 to 4 times larger than D
    """
    def __init__(self, D, nhead, num_layers, dim_feedforward=128, dropout=0.1):
        super(BagsTransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Ensuring the input and output are batch-first
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, src, padding_mask):
        """
        Forward pass of the custom transformer encoder.

        Parameters:
            src (Tensor): The input tensor of shape (B, S, D) where B is batch size,
                          S is sequence length, and D is the feature dimension.
            padding_mask (BoolTensor): The mask tensor of shape (B, S) where
                          False indicates positions that should be ignored (padded). 
                          (This is backwards from the usual src_key_padding_mask)
                          

        Returns:
            Tensor: The output tensor of shape (B, S, D) after encoding.
        """
        # Apply the transformer encoder directly on the batch-first input
        output = self.transformer_encoder(src, src_key_padding_mask=~padding_mask)
        return output
  

class ABMIL_Gated_Attention(nn.Module):
    '''
    Adapted version for batches of data. Input tensor should
    be (B, S, D) = (batch size, sequence length or bag size, embedding dimension).
    The mask, if provided, is of shape (B, S) where False indicates padding that should be zeroed out.
    If mask is None or batch size B is 1 and mask is not provided, no masking is applied.
    '''
    def __init__(self, nf=512, L=128):
        super().__init__()
        self.nf = nf  # Length of feature vectors
        self.L = L  # Latent dimension for attention gates

        self.U = nn.Sequential(
            nn.Linear(self.nf, self.L),
            nn.Sigmoid()
        )

        self.V = nn.Sequential(
            nn.Linear(self.nf, self.L),
            nn.Tanh()
        )

        self.W = nn.Sequential(
            nn.Linear(self.L, 1),
        )

    def forward(self, x, mask=None):
        # Apply gates U and V
        u = self.U(x)  # (B, S, L)
        v = self.V(x)  # (B, S, L)

        # Element-wise multiplication and compute attention scores
        attention_scores = self.W(u * v).squeeze(-1)  # (B, S)

        # Check if the mask is provided and applicable
        if mask is not None and x.size(0) > 1:
            # Use the mask to set scores of padding positions to a very large negative value
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))

        # Compute attention weights using softmax
        attention_weights = F.softmax(attention_scores, dim=1)  # Shape (B, S)

        return attention_weights


class FeatureAggregator(nn.Module):
    """
    A class to aggregate feature vectors using provided attention weights.

    This class takes a batch of feature vectors and a corresponding batch of
    attention weights, and computes the aggregated feature vectors for each
    bag in the batch.
    """
    def __init__(self):
        super(FeatureAggregator, self).__init__()

    def forward(self, features, attention_weights):
        """
        Forward pass to aggregate feature vectors using attention weights.

        Args:
            features (Tensor): Tensor of shape (B, S, D) containing the feature vectors.
                                B is the batch size, S is the sequence length,
                                and D is the feature dimension.
            attention_weights (Tensor): Tensor of shape (B, S) containing the normalized
                                        attention weights for each feature vector in the batch.

        Returns:
            Tensor: Aggregated feature vectors of shape (B, D).
        """
        # Validate that attention weights are normalized
        B = attention_weights.shape[0]
        if not torch.allclose(attention_weights.sum(dim=1),
                              torch.ones(B, dtype=attention_weights.dtype, device=attention_weights.device)):
            raise ValueError("Attention weights are not normalized properly")

        # Perform weighted sum of features
        # Unsqueezing attention weights to (B, S, 1) to make the dimensions compatible for multiplication
        weighted_features = features * attention_weights.unsqueeze(-1)
        aggregated_features = weighted_features.sum(dim=1)  # Summing over the sequence dimension

        return aggregated_features

class SimpleClassifier(nn.Module):
    """
    A simple classification module with one hidden layer and a LeakyReLU activation.

    Args:
        D (int): Dimensionality of the input feature vector.
        H (int): Number of nodes in the hidden layer.
        num_classes (int): Number of classes for classification output.

    Attributes:
        hidden_layer (nn.Linear): Linear layer mapping input features to hidden layer.
        activation (nn.LeakyReLU): LeakyReLU activation function with negative slope of 0.1.
        output_layer (nn.Linear): Linear layer mapping hidden layer outputs to class scores.
    """
    def __init__(self, D, H, num_classes):
        super(SimpleClassifier, self).__init__()
        self.classifier = GeneralMLP([D,H,num_classes])                           

    def forward(self, x):
        """
        Forward pass of the classifier.

        Args:
            x (Tensor): Input tensor of shape (N, D) where N is the batch size and D is the
                        dimensionality of the input features.

        Returns:
            Tensor: Output tensor of class scores of shape (N, num_classes).
        """
        return self.classifier(x)

class TopKPooling(nn.Module):
    def __init__(self, k):
        super(TopKPooling, self).__init__()
        if not (isinstance(k, int) and k > 0) and not (0 < k <= 1):
            raise ValueError("k must be a positive integer or a number between 0 and 1.")
        self.k = k

    def forward(self, x):
        
        # Get the shape of the input tensor
        batch_size, *dims = x.size()
        
        # Flatten the last two dimensions
        num_elements = dims[-1] * dims[-2]
        x_flat = x.view(batch_size, *dims[:-2], -1)

        if isinstance(self.k, int):
            k = self.k
        else:
            k = max(int(self.k * num_elements), 1)

        # Ensure k is not larger than the total number of elements
        k = min(k, num_elements)
        
        # Get the top k values
        topk_vals, _ = torch.topk(x_flat, k, dim=-1)
        
        # Average the top k values
        output = topk_vals.mean(dim=-1)

        return output

class Saliency_Classifier(nn.Module):
    """
    Takes feature maps as inputs produces a saliency map and instance prediction
    """
    
    def __init__(self, nf = 512, num_classes = 1, k = 0.2):
        super(Saliency_Classifier,self).__init__()
        self.nf = nf
        self.num_classes = num_classes # two for binary classification
        self.k = k
        
        self.saliency_layer = nn.Sequential(        
            nn.Conv2d( self.nf, self.num_classes, (1,1), bias = False),
            nn.Sigmoid() )

        self.topkpool = TopKPooling(k)
                    
    def forward(self, h):
        # input is a tensor with a bag of features, dim = bag_size x nf x h x w
    
        saliency_maps = self.saliency_layer(h)
        yhat_instance = self.topkpool(saliency_maps).squeeze()

        return yhat_instance, saliency_maps

class ImageABMIL(nn.Module):
    def __init__(self,backbone,nhead,num_classes,num_layers,
                 topk_features=1, feat_dim=128, L=128, use_encoder=False, normalize_features=False):
        super().__init__()
        self.backbone = backbone # extracts feature maps
        self.feat_dim = feat_dim # dim of projected feature vectors
        self.topk_features = topk_features # topk parameter for pooling feature maps to feature vectors, use 1 for max pooling, large integer for avg pooling, and or integer for top k
        
        self.L = L # latent dimension for gated-attention
        self.nf = backbone.num_features # number of channels in output
        self.nhead = nhead # number of attention heads in MIL transformer encoder
        self.num_classes = num_classes
        self.num_layers = num_layers # number of transformer layers
        
        self.use_encoder = use_encoder
        self.normalize_features = normalize_features

        self.hidden_mlp_dim= min(2*self.feat_dim,self.nf)
        self.hidden_classifier_dim = max(self.feat_dim//2,self.num_classes)
        
        self.pooler_projector = nn.Sequential(
            TopKPooling(self.topk_features),
            GeneralMLP( [self.nf, self.hidden_mlp_dim, self.feat_dim], normalize=self.normalize_features)
        )

        self.bags_encoder_model = ListOfBagsProcessor(self.backbone)

        self.bags_pooler_projector = ListOfBagsProcessor(self.pooler_projector)

        self.padder = BagOfFeaturesPadder()
        
        if use_encoder:
            self.transformer = BagsTransformerEncoder(
                self.feat_dim, 
                self.nhead, 
                self.num_layers, 
                dim_feedforward=2*self.feat_dim, 
                dropout=0.1
            )
        else:
            self.transformer = nn.Identity()

        self.gated_attention = ABMIL_Gated_Attention(nf=self.feat_dim,L=self.L)

        self.feature_aggregator = FeatureAggregator()

        self.classifier = GeneralMLP( [self.feat_dim, self.hidden_classifier_dim, self.num_classes] )

    def forward(self, x, mode='all'):
        if mode == 'all': # x should be a list of image tensors each [bag_size, 3, H, W]
            bags_of_feature_maps = self.bags_encoder_model(x)
            bags_of_features = self.bags_pooler_projector(bags_of_feature_maps)
            padded_bags_of_features, mask = self.padder(bags_of_features)
            if self.use_encoder:
               padded_bags_of_features = self.transformer(padded_bags_of_features,mask)
            attention_weights = self.gated_attention(padded_bags_of_features,mask)
            

            bag_features = self.feature_aggregator(padded_bags_of_features, attention_weights)
            logits = self.classifier(bag_features)

            return logits.cuda(), None, None, None
        
        elif mode == 'encode_project': # x should b a tensor of images [batch_size, 3, H, W]
            feature_maps = self.backbone(x)
            features = self.pooler_projector(feature_maps)
            return None, None, None, features.view(features.size(0))

class ImageABMILIns(nn.Module):
    def __init__(self,backbone,nhead,num_classes,num_layers,topk_features=1, feat_dim=128, L=128,use_encoder=False, normalize_features=False):
        super().__init__()
        self.backbone = backbone # extracts feature maps
        self.feat_dim = feat_dim # dim of projected feature vectors
        self.L = L # latent dimension for gated-attention
        self.nf = backbone.num_features # number of channels in output
        self.nhead = nhead # number of attention heads in MIL transformer encoder
        self.num_classes = num_classes
        self.num_layers = num_layers # number of transformer layers
        self.topk_features = topk_features # topk parameter for pooling feature maps to feature vectors, use 1 for max pooling, large integer for avg pooling, and or integer for top k
        self.use_encoder = use_encoder
        self.normalize_features = normalize_features

        self.hidden_mlp_dim= min(2*self.feat_dim,self.nf)
        self.hidden_classifier_dim = max(self.feat_dim//2,self.num_classes)

        self.bags_encoder_model = ListOfBagsProcessor(self.backbone)

        self.bags_pooler_projector = ListOfBagsProcessor(
            nn.Sequential(
                TopKPooling(self.topk_features),
                GeneralMLP( [self.nf, self.hidden_mlp_dim, self.feat_dim], normalize=self.normalize_features)
            )
        )

        self.padder = BagOfFeaturesPadder()
        
        if use_encoder:
            self.transformer = BagsTransformerEncoder(
                self.feat_dim, 
                self.nhead, 
                self.num_layers, 
                dim_feedforward=2*self.feat_dim, 
                dropout=0.1
            )
        else:
            self.transformer = nn.Identity()

        self.gated_attention = ABMIL_Gated_Attention(nf=self.feat_dim,L=self.L)

        self.feature_aggregator = FeatureAggregator()

        self.classifier = GeneralMLP( [self.feat_dim, self.hidden_classifier_dim, self.num_classes] )

        self.sigmoid = nn.Sigmoid()

    def forward(self,bags_of_images):
        bags_of_feature_maps = self.bags_encoder_model(bags_of_images)
        bags_of_features = self.bags_pooler_projector(bags_of_feature_maps)
        padded_bags_of_features, mask = self.padder(bags_of_features)
        if self.use_encoder:
           padded_bags_of_features = self.transformer(padded_bags_of_features,mask)
        attention_weights = self.gated_attention(padded_bags_of_features,mask)

        bags_of_instance_preds = self.sigmoid( self.classifier( padded_bags_of_features ) )
        
        bag_predictions = self.feature_aggregator(bags_of_instance_preds, attention_weights)

        return bag_predictions.view(bag_predictions.size(0))

class ImageABMILSaliency(nn.Module):
    def __init__(self,backbone,nhead,num_classes,num_layers,topk_features = 0.2, topk_saliency = 0.2, 
                 feat_dim=128,L=128,use_encoder=False, normalize_features=False):
        super().__init__()

        # set backbone
        self.backbone = backbone
        self.nf = backbone.num_features
        
        # model parameters
        self.feat_dim = feat_dim # dim of projected feature vectors
        self.hidden_mlp_dim= min(2*self.feat_dim,self.nf) # for MLP in transformer if used
        self.L = L # latent dimension for gated-attention
        self.nhead = nhead # number of attention heads in MIL transformer encoder
        self.normalize_features = normalize_features
        self.num_classes = num_classes
        self.num_layers = num_layers # number of transformer layers
        self.topk_features = topk_features # 1 for max pool, large int for avg pool, int for top k 
        self.topk_saliency = topk_saliency # topk parameter for aggregating saliency maps to instance predictions
        self.use_encoder = use_encoder
 
        # model layers
        self.pooler_projector = nn.Sequential(
            TopKPooling(self.topk_features),
            GeneralMLP( [self.nf, self.hidden_mlp_dim, self.feat_dim], normalize=self.normalize_features )
        )
        self.bags_encoder_model = ListOfBagsProcessor(self.backbone)
        self.bags_pooler_projector = ListOfBagsProcessor(self.pooler_projector)
        self.bags_saliency_model = ListOfBagsProcessor(
            nn.Sequential(
                nn.Conv2d( self.nf, self.num_classes, (1,1), bias = False),
                nn.Sigmoid()
            )
        )
        self.bags_topk_pooler = ListOfBagsProcessor(
            TopKPooling(self.topk_saliency)
        )
        self.padder = BagOfFeaturesPadder()
        if use_encoder:
            self.transformer = BagsTransformerEncoder(
                self.feat_dim, 
                self.nhead, 
                self.num_layers, 
                dim_feedforward=2*self.feat_dim, 
                dropout=0.1
            )
        else:
            self.transformer = nn.Identity()
        self.gated_attention = ABMIL_Gated_Attention(nf=self.feat_dim,L=self.L)
        self.feature_aggregator = FeatureAggregator()

    def forward(self, x, mode='all'):

        if mode=='all': # x should be a list of bags of images each [bag_size,C,H,W]
            # encoding and projection
            bags_of_feature_maps = self.bags_encoder_model(x)        
            bags_of_features = self.bags_pooler_projector(bags_of_feature_maps)
    
            # attention based-aggregation
            padded_bags_of_features, mask = self.padder(bags_of_features)
            if self.use_encoder:
               padded_bags_of_features = self.transformer(padded_bags_of_features,mask)
            attention_weights = self.gated_attention(padded_bags_of_features,mask)
            bags_of_saliency_maps = self.bags_saliency_model(bags_of_feature_maps)
            bags_of_instance_preds = self.bags_topk_pooler(bags_of_saliency_maps)
            padded_bags_of_instance_preds,_ = self.padder(bags_of_instance_preds) # generates same mask
            bag_predictions = self.feature_aggregator(padded_bags_of_instance_preds, attention_weights)
    
            return bag_predictions.view(bag_predictions.size(0))
        elif mode=='encode_project': # x should be a tensor batch of images [batch_size,C,H,W]
            feature_maps = self.backbone(x)
            feature_vectors = self.pooler_projector(feature_maps)
            return feature_vectors

    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.pooler_projector.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.pooler_projector.parameters():
            param.requires_grad = True

    def reset(self):
        def init_weights(m):
            if hasattr(m, 'weight') and m.weight is not None:
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        def apply_init(m, skip_modules):
            if not any(m is module for module in skip_modules):
                init_weights(m)

        # Modules to skip during initialization
        skip_modules = [self.backbone, self.pooler_projector]

        for name, module in self.named_children():
            apply_init(module, skip_modules)


# here is an example of having different modes for running pieces of the same model

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # Define the first group of layers
        self.group1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Define the second group of layers
        self.group2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Final fully connected layer
        self.fc = nn.Linear(32 * 8 * 8, 10)  # Assuming input image size is 32x32
        
    def forward(self, x, mode='all'):
        if mode == 'one':
            # Freeze the first group of layers
            with torch.no_grad():
                x = self.group1(x)
            x = self.group2(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            
        elif mode == 'two':
            x = self.group1(x)  # First group of layers with gradients
            return x
        
        elif mode == 'all':
            x = self.group1(x)
            x = self.group2(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            
        return x

# # Example usage
# model = CustomModel()
# input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor

# output_all = model(input_tensor, mode='all')
# print('Output (all layers):', output_all)

# output_one = model(input_tensor, mode='one')
# print('Output (one group frozen):', output_one)

# output_two = model(input_tensor, mode='two')
# print('Output (first group only):', output_two)

 