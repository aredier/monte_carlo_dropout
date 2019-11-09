import torch
import torch.nn as nn


__all__ = ['AlexNet', 'alexnet']

from monte_carlo_dropout.mc_dropout import MCDropout

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class DropOutAlexnet(nn.Module):

    def __init__(self, og_alex: AlexNet, conv_dropout: float, dense_dropout: float, force_dropout: bool):

        super().__init__()
        self.dense_dropout = dense_dropout
        self.force_dropout = force_dropout

        # features
        feature_layers = list(og_alex.features.children())
        self.conv_1 = feature_layers[0]
        self.relu_1 = feature_layers[1]
        self.pool_1 = feature_layers[2]
        self.dropout_1 = MCDropout(conv_dropout)
        self.conv_2 = feature_layers[3]
        self.relu_2 = feature_layers[4]
        self.pool_2 = feature_layers[5]
        self.dropout_2 = MCDropout(conv_dropout)
        self.conv_3 = feature_layers[6]
        self.relu_3 = feature_layers[7]
        self.dropout_3 = MCDropout(conv_dropout)
        self.conv_4 = feature_layers[8]
        self.relu_4 = feature_layers[9]
        self.dropout_4 = MCDropout(conv_dropout)
        self.conv_5 = feature_layers[10]
        self.relu_5 = feature_layers[11]
        self.pool_3 = feature_layers[12]

        self.avgpool = og_alex.avgpool

        # classifier
        clasifier_layers = list(og_alex.classifier.children())
        self.dropout_5 = MCDropout(dense_dropout)
        self.dense_1 = clasifier_layers[1]
        self.relu_6 = clasifier_layers[2]
        self.dropout_6 = MCDropout(dense_dropout)
        self.dense_2 = clasifier_layers[4]
        self.relu_7 = clasifier_layers[5]
        self.dense_3 = clasifier_layers[6]

    def forward(self, x):
        out = self.conv_1(x)
        out = self.relu_1(out)
        out = self.pool_1(out)
        out = self.dropout_1(out)
        out = self.conv_2(out)
        out = self.relu_2(out)
        out = self.pool_2(out)
        out = self.dropout_2(out)
        out = self.conv_3(out)
        out = self.relu_3(out)
        out = self.dropout_3(out)
        out = self.conv_4(out)
        out = self.relu_4(out)
        out = self.dropout_4(out)
        out = self.conv_5(out)
        out = self.relu_5(out)
        out = self.pool_3(out)

        out = self.avgpool(out)

        out = self.dropout_5(out)
        out = self.dense_1(out)
        out = self.relu_6(out)
        out = self.dropout_6(out)
        out = self.dense_2(out)
        out = self.relu_7(out)
        out = self.dense_3(out)

        return out
