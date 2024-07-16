import torch
from model import generate_model
import options
from train import train
from test import test
from loader import loader_train_cifar, loader_test_cifar
from os.path import exists, join


if __name__ == '__main__':

    args = options.parser.parse_args()  # 对参数进行解析

    # parameters
    batch_sz = args.batch_sz
    lr = args.lr
    epochs = args.epochs
    plot = args.plot
    device = torch.device(f'cuda:{args.device}')
    model_name = args.model_name
    input_channels = args.input_channels
    model = generate_model(model_name, input_channels)
    pretrained_model_path = args.pretrained_model_path

    if args.training_scratch is True:
        train(model, device, lr, epochs, plot, loader_train_cifar)
        model = torch.load(pretrained_model_path + 'model.pth')
        acc = test(model, device, loader_test_cifar)
        print('acc:', acc)

    else:
        model = torch.load(join(pretrained_model_path, 'model.pth'))
        if not exists(join(pretrained_model_path, 'model.pth')):
            raise RuntimeError('no trained model!')
        acc = test(model, device, loader_test_cifar)
        print('acc:', acc)













