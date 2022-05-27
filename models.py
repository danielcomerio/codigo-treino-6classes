from torchvision import models
from typing import Any
import torch.nn as nn


def initialize_pretrained_model(model_name: str, num_classes: int, use_pretrained: bool=True) -> None: # typerror # tuple[Any, int]
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnext50":
        """ Resnext50
        """
        model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg16":
        """ VGG16
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        model_ft.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        model_ft.classifier[2] = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        input_size = 224

    elif model_name == "densenet121":
        """ Densenet121
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "densenet169":
        """ Densenet169
        """
        model_ft = models.densenet169(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size


# def initialize_pretrained_model(model_name: str, num_classes: int, use_pretrained: bool=True) -> None: # typerror # tuple[Any, int]
#     # Initialize these variables which will be set in this if statement. Each of these
#     #   variables is model specific.
#     model_ft = None
#     input_size = 0

#     if model_name == "resnet":
#         """ Resnet50
#         """
#         model_ft = models.resnet50(pretrained=use_pretrained)
#         num_ftrs = model_ft.fc.in_features
#         model_ft.fc = nn.Linear(num_ftrs, num_classes)
#         input_size = 224

#     elif model_name == "vgg":
#         """ VGG11_bn
#         """
#         model_ft = models.vgg16(pretrained=use_pretrained)
#         num_ftrs = model_ft.classifier[6].in_features
#         model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
#         input_size = 224

#     elif model_name == "squeezenet":
#         """ Squeezenet
#         """
#         model_ft = models.squeezenet1_0(pretrained=use_pretrained)
#         model_ft.classifier[1] = nn.Conv2d(
#             512, num_classes, kernel_size=(1, 1), stride=(1, 1))
#         model_ft.num_classes = num_classes
#         input_size = 224

#     elif model_name == "densenet":
#         """ Densenet
#         """
#         model_ft = models.densenet121(pretrained=use_pretrained)
#         num_ftrs = model_ft.classifier.in_features
#         model_ft.classifier = nn.Linear(num_ftrs, num_classes)
#         input_size = 224

#     elif model_name == "inception":
#         """ Inception v3
#         Be careful, expects (299,299) sized images and has auxiliary output
#         """
#         model_ft = models.inception_v3(pretrained=use_pretrained)
#         # Handle the auxilary net
#         num_ftrs = model_ft.AuxLogits.fc.in_features
#         model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
#         # Handle the primary net
#         num_ftrs = model_ft.fc.in_features
#         model_ft.fc = nn.Linear(num_ftrs, num_classes)
#         input_size = 299

#     else:
#         print("Invalid model name, exiting...")
#         exit()
    
#     return model_ft, input_size


def initialize_model(model_type: str, n_mpl_neurons: int) -> None: # tuple[Any, int]
    input_size = 2000 + 500

    if model_type == 'mlp':
        net = nn.Sequential(
            nn.Linear(input_size, n_mpl_neurons),
            nn.SELU(),
            nn.Linear(n_mpl_neurons, n_mpl_neurons),
            nn.SELU(),
            nn.Linear(n_mpl_neurons, 1),
        )
    elif model_type == 'cnn1d':
        net = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=64,
                kernel_size=5,
                stride=2
            ),
            nn.SELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=2
            ),
            nn.SELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=2
            ),
            nn.SELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1920, 512),
            nn.SELU(),
            nn.Linear(512, 256),
            nn.SELU(),
            nn.Linear(256, 1),
        )
    else:
        raise Exception(f"Model type '{model_type}' undefined.")
    
    return net, input_size