from torchvision.datasets import MNIST
from omegaconf import OmegaConf
config = OmegaConf.load("./configs/config.yaml")
test_dataset = MNIST(root=config.data.root_dir, train=False)
def testImage() -> list:
    result = []
    for i, (image, label) in enumerate(test_dataset):
        if i < 5:
            image.save('{:1d}.jpg'.format(i))
            result.append(label)
        else:
            break
    return result