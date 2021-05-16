import json

with open('C:\\Users\\aaron\\OneDrive\\桌面\\DATA\\03877078\\03877078.json', 'r', encoding='utf-8') as f:
    projDict = json.load(f)
    print(projDict)



from torch.utils.data import dataset
from torchvision.datasets import ImageFolder
