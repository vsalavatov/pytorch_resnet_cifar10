from resnet import resnet20, resnet32, resnet44
from fashion_cnn import fashion_cnn

model_names = ['resnet20', 'resnet32', 'resnet44', 'fashion_cnn']
def get_model(name):
    if name == 'resnet20':
        return resnet20()
    elif name == 'resnet32':
        return resnet32()
    elif name == 'resnet44':
        return resnet44()
    elif name == 'fashion_cnn':
        return fashion_cnn()
    else:
        raise ValueError("this network not implemented")