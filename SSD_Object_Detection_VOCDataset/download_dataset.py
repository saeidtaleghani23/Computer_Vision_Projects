# download VOC 2012 dataset using torch
import torchvision
import os

def main():
    # create dataset folder
    root= './dataset'
    if not os.path.exists(root):
        os.makedirs(root)
    torchvision.datasets.VOCDetection(
    root=root,
    year = '2012',
    image_set = 'trainval',
    download=True,
)
if __name__ == "__main__":
    main() 