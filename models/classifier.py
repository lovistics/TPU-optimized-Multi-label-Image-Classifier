import torchvision
from torch import nn
import warnings
warnings.filterwarnings("ignore")

class ImprovedMultiLabelClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedMultiLabelClassifier, self).__init__()

        # Use EfficientNet-B2 as backbone
        self.backbone = torchvision.models.efficientnet_b2(weights='DEFAULT')
        backbone_out_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        # Advanced classifier head
        self.classifier = nn.Sequential(
            nn.Linear(backbone_out_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def unfreeze_layers(self, from_layer=6):
        # Freeze backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last few layers of backbone
        for param in list(self.backbone.children())[-from_layer:]:
            for p in param.parameters():
                p.requires_grad = True

        # Always train classifier
        for param in self.classifier.parameters():
            param.requires_grad = True
