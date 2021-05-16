import os
from PIL import Image
from cv2 import imread
import torchvision.transforms.functional as TF


os.mkdir('./content/multiangle')

img = Image.open('.\\content\\DATA\\00048035\\00048035_FILE1.bmp')

for angle in range(0, 90, 10):

    img2 = TF.rotate(img, angle, expand=True)
    img2.save(os.path.join('./content/multiangle/', '00048035_FILE1_rt' + str(angle) + '.bmp'))
