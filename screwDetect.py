import torch
import torch.nn as nn
from torchvision.transforms import functional as TF
from torchvision import transforms

class BuildModel(nn.Module):

    def __init__(self):
        
        super(BuildModel, self).__init__()
        
        # ----------------------------------------------
        # 初始化模型的 layer (input size: 224 * 224)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2)
        )

        # ----------------------------------------------
               
    def forward(self, x):
        
        # ----------------------------------------------
        # Forward (final output 1 probability)

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        # ----------------------------------------------

        return x

def get_scew_model(path=r'C:\Users\user\OneDrive\桌面\vertex\weights\Alexnet\screw_clf(2).pt'):

    return torch.load(path, map_location = torch.device('cpu'))

def screwPredict(seg_imgs, model, transformer = None):

    if transformer == None:
        transformer = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        ])

    data = torch.cat([TF.to_tensor(TF.resize(seg_img, [224,224])) for seg_img in seg_imgs])

    model.eval()

    return torch.argmax(model(data.unsqueeze(1)), axis=1)

