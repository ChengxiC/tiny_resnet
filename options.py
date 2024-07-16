# Chengxi Chu, Universiti Malaya
import argparse


parser = argparse.ArgumentParser('ResNet10 for cifar10')
parser.add_argument('--device', type=int, default=0, help='GPU ID')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--model_name', type=str, default='ResNet10')
parser.add_argument('--batch_sz', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='epochs for training')
parser.add_argument('--input_channels', type=int, default=3, help='the channels of the dataset images')
parser.add_argument('--training_scratch', type=bool, default=True, help='if need to train from scratch')
parser.add_argument('--pretrained_model_path', type=str, default='./pretrained_model/', help='the path of the pretrained model')
parser.add_argument('--plot', type=bool, default=True, help='whether to plot the loss curve')



