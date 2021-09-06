from typing import Optional, Union, List
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.unet.decoder import UnetDecoder

from .smp_modules import get_separate_encoder

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationModel
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import initialization as init

from albumentations import ShiftScaleRotate
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        
        conv = nn.Conv2d(in_channels, in_channels//2, 1)
        bn = nn.BatchNorm2d(in_channels//2)
        relu = nn.ReLU()
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = nn.Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels//2, classes, bias=True)
        activation = Activation(activation)
        super().__init__(conv, bn, relu, pool, flatten, dropout, linear, activation)
        
        
class Unet(SegmentationModel):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

        
class PretextClassifier(nn.Sequential):

    def __init__(self, in_channels, pretext_classes):
        pool = nn.AdaptiveAvgPool2d(1)
        flatten = nn.Flatten()
        dropout = nn.Dropout(0.2)
        linear = nn.Linear(in_channels, pretext_classes, bias=True)
        activation = nn.Softmax(dim=1)
        super().__init__(
            pool, flatten, 
            dropout,
            linear, activation)


class DomainUnet(SegmentationModel):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        pretext_classes = -1,
        domain_classes = 2,
        domain_layer = -2,
        domain_classifier='DomainClassifierFlatten',
        separate = False,
        input_shape=80,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        ) if separate is False else \
        get_separate_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.separate = separate

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if domain_classifier=='DomainClassifier':
            domain_classifier = DomainClassifier
        elif domain_classifier=='DomainClassifierFlatten':
            domain_classifier = DomainClassifierFlatten
        elif domain_classifier=='DomainClassifierFlattenSimple':
            domain_classifier = DomainClassifierFlattenSimple
        elif domain_classifier=='DomainClassifierFlattenCat':
            domain_classifier = DomainClassifierFlattenCat
        elif domain_classifier=='DomainClassifierReduceFlatten':
            domain_classifier = DomainClassifierReduceFlatten
            
        self.domain_layer = domain_layer
        
        if type(self.domain_layer)==list:
            self.domain_classification_head = domain_classifier(
                in_channels=sum([self.encoder.out_channels[idx_dl] 
                                 for idx_dl in self.domain_layer]), 
                domain_classes=domain_classes
            )
            if separate:
                self.encoder.set_domain_layer(self.domain_layer)
        else:
            self.domain_classification_head = domain_classifier(
                in_channels=self.encoder.out_channels[self.domain_layer], 
                domain_classes=domain_classes,
                input_shape=input_shape,
            )
        
        if pretext_classes==-1:
            raise ValueError(f'initialize pretext_classes')
        self.pretext_classification_head = PretextClassifier(
            in_channels=self.encoder.out_channels[-1], pretext_classes=pretext_classes
        )

        self.name = "u-{}".format(encoder_name)
        self.initialize()
        return
        

    def forward(self, x, alpha, domain_path=False):
        if type(self.domain_layer)==list:
            if self.separate:
                features, domain_features = self.encoder(x)
            else:
                features = self.encoder(x)
        else:
            features = self.encoder(x)
        
        if type(self.domain_layer)==list:
            _, _, W, H = features[self.domain_layer[0]].shape
            if self.separate:
                extracted_features = torch.cat([F.interpolate(domain_features[feat], size=(W,H), mode='bilinear') 
                                                for feat in range(len(self.domain_layer))], dim=1)
            else:
                extracted_features = torch.cat([F.interpolate(features[feat], size=(W,H), mode='bilinear') 
                                                for feat in self.domain_layer], dim=1)
        else:
            extracted_features = features[self.domain_layer]
        reverse_feature = ReverseLayerF.apply(extracted_features, alpha)
        domain_output = self.domain_classification_head(reverse_feature)
        
        if not domain_path:
            pretext_output = self.pretext_classification_head(features[-1])
            return pretext_output, domain_output
        else:
            decoder_output = self.decoder(*features)
            masks = self.segmentation_head(decoder_output)
            return masks, domain_output
    

    def predict(self, x, alpha=0):
        if self.training:
            self.eval()

        with torch.no_grad():
            y = self.forward(x, alpha)

        return y
    

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.domain_classification_head is not None:
            init.initialize_head(self.domain_classification_head)
        if self.pretext_classification_head is not None:
            init.initialize_head(self.pretext_classification_head)
        return
    
    
class DomainSSUnet(SegmentationModel):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        pretext_classes = -1,
        domain_classes = 2,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.domain_layer = -2
        self.domain_classification_head = DomainClassifier(
            in_channels=self.encoder.out_channels[self.domain_layer], domain_classes=domain_classes
        )
        
        if pretext_classes==-1:
            raise ValueError(f'initialize pretext_classes')
        self.pretext_classification_head = PretextClassifier(
            in_channels=self.encoder.out_channels[-1], pretext_classes=pretext_classes
        )

        self.name = "u-{}".format(encoder_name)
        self.initialize()
        return
        

    def forward(self, x, alpha, domain_path=False):
        features = self.encoder(x)
        
        reverse_feature = ReverseLayerF.apply(features[self.domain_layer], alpha) #### features from ? layer
        domain_output = self.domain_classification_head(reverse_feature)
        
        pretext_output = self.pretext_classification_head(features[-1])
        return pretext_output, domain_output

    

    def predict(self, x, alpha=0):
        if self.training:
            self.eval()

        with torch.no_grad():
            y = self.forward(x, alpha)

        return y
    

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.domain_classification_head is not None:
            init.initialize_head(self.domain_classification_head)
        if self.pretext_classification_head is not None:
            init.initialize_head(self.pretext_classification_head)
        return
    
