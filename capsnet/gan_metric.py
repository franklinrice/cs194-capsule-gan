import os, time
import pickle
import torch
from model import Net
from gan import generator
import torch
import torch.nn as nn
import numpy as np
import argparse
import utils
from torch.autograd import Variable
from collections import OrderedDict
from dcgan import base_generator, base_discriminator
from scipy.misc import imresize
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def calculate_r(G, D, size):
    batch = 50
    zs = Variable(torch.randn((batch, 100)).view(-1, 100, 1, 1))
    generated = G(zs)
    resized_generated = np.zeros((batch, 1, size, size))
    for i in range(batch):
        resized_generated[i][0] = imresize(generated.data.numpy()[i][0], size=(size, size))
    #acc = np.sum(D(resized_generated).data.numpy()) / batch
    print(np.amin(resized_generated), np.amax(resized_generated))
    acc = np.sum(torch.round(D(Variable(torch.from_numpy(resized_generated).float()))).data.numpy()) / batch

    # Code to visualize
    # rg = np.linspace(-1, 1, 10)
    # fig = plt.figure()
    # for i in range(len(rg)):
    #     a = fig.add_subplot(5, 6, i + 1)
    #     image = resized_generated[i][0]
    #     x=plt.imshow(image, cmap = 'gray')
    # fig.subplots_adjust(hspace=.3)
    # plt.show()


    return acc

def calculate_r_test(D, size):
    batch_size = 50

    transform = transforms.Compose([
            transforms.Scale(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)

    for x_, _ in test_loader:
        #print(D(x_).data.numpy())
        acc = np.sum(torch.round(D(x_)).data.numpy()) / batch_size
        break
    return acc

def load_model(model_type, dict_file):
    state_dict = torch.load(dict_file, map_location=lambda storage, loc: storage)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if k[:7] == 'module.':
            name = k[7:] # remove `module.`
        if name[:2] == 'fc':
            name = 'decoder.' + name
        new_state_dict[name] = v

    model = None
    if model_type == 'g':
        model = generator(128)
    elif model_type == 'd':
        model = Net(num_conv_in_channel=args.num_conv_in_channel,
                    num_conv_out_channel=args.num_conv_out_channel,
                    num_primary_unit=args.num_primary_unit,
                    primary_unit_size=args.primary_unit_size,
                    num_classes=args.num_classes,
                    output_unit_size=args.output_unit_size,
                    num_routing=args.num_routing,
                    use_reconstruction_loss=args.use_reconstruction_loss,
                    regularization_scale=args.regularization_scale,
                    input_width=args.input_width,
                    input_height=args.input_height,
                    cuda_enabled=args.cuda)
    elif model_type == 'b_g':
        model = base_generator(128)
    elif model_type == 'b_d':
        model = base_discriminator(128)

    model.load_state_dict(new_state_dict)
    return model

def main():

    global args

    # Setting the hyper parameters
    parser = argparse.ArgumentParser(description='Example of Capsule Network')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs. default=10')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate. default=0.01')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='training batch size. default=128')
    parser.add_argument('--test-batch-size', type=int,
                        default=128, help='testing batch size. default=128')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status. default=10')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training. default=false')
    parser.add_argument('--threads', type=int, default=4,
                        help='number of threads for data loader to use. default=4')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for training. default=42')
    parser.add_argument('--num-conv-out-channel', type=int, default=256,
                        help='number of channels produced by the convolution. default=256')
    parser.add_argument('--num-conv-in-channel', type=int, default=1,
                        help='number of input channels to the convolution. default=1')
    parser.add_argument('--num-primary-unit', type=int, default=8,
                        help='number of primary unit. default=8')
    parser.add_argument('--primary-unit-size', type=int,
                        default=1152, help='primary unit size is 32 * 6 * 6. default=1152')
    parser.add_argument('--num-classes', type=int, default=1,
                        help='number of digit classes. 1 unit for one MNIST digit. default=10')
    parser.add_argument('--output-unit-size', type=int,
                        default=1, help='output unit size. default=16')
    parser.add_argument('--num-routing', type=int,
                        default=3, help='number of routing iteration. default=3')
    parser.add_argument('--use-reconstruction-loss', type=utils.str2bool, nargs='?', default=True,
                        help='use an additional reconstruction loss. default=True')
    parser.add_argument('--regularization-scale', type=float, default=0.0005,
                        help='regularization coefficient for reconstruction loss. default=0.0005')
    parser.add_argument('--dataset', help='the name of dataset (mnist, cifar10)', default='mnist')
    parser.add_argument('--input-width', type=int,
                        default=28, help='input image width to the convolution. default=28 for MNIST')
    parser.add_argument('--input-height', type=int,
                        default=28, help='input image height to the convolution. default=28 for MNIST')
    parser.add_argument('--is-training', type=int,
                        default=1, help='Whether or not is training, default is yes')
    parser.add_argument('--weights', type=str,
                        default=None, help='Load pretrained weights, default is none')

    args = parser.parse_args()

    print(args)

    # Check GPU or CUDA is available
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Get reproducible results by manually seed the random number generator
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)


    # G_caps = generator(128)
    # D_caps = Net(num_conv_in_channel=args.num_conv_in_channel,
    #                 num_conv_out_channel=args.num_conv_out_channel,
    #                 num_primary_unit=args.num_primary_unit,
    #                 primary_unit_size=args.primary_unit_size,
    #                 num_classes=args.num_classes,
    #                 output_unit_size=args.output_unit_size,
    #                 num_routing=args.num_routing,
    #                 use_reconstruction_loss=args.use_reconstruction_loss,
    #                 regularization_scale=args.regularization_scale,
    #                 input_width=args.input_width,
    #                 input_height=args.input_height,
    #                 cuda_enabled=args.cuda)

    # #G_caps.load_state_dict(g_caps_dict)
    # G_caps_dict = torch.load('models/mnist_generator_param.pkl', map_location=lambda storage, loc: storage)
    # D_caps_dict = torch.load('models/mnist_discriminator_param.pkl', map_location=lambda storage, loc: storage)
    # #G_caps.load_state_dict(open("models/mnist_generator_param.pkl", 'rb'))
    # #D_caps.load_state_dict(d_caps_dict)
    G_caps = load_model('g', 'models/mnist_capsgan_generator_param.pkl')
    D_caps = load_model('d', 'models/mnist_capsgan_discriminator_param.pkl')

    G_original = load_model('b_g', 'models/mnist_dcgan_generator_param.pkl')
    D_original = load_model('b_d', 'models/mnist_dcgan_discriminator_param.pkl')

    r_try = calculate_r(G_caps, D_caps, 28)
    print("r try: {}".format(r_try))

    a = calculate_r(G_caps, D_original, 64)
    b = calculate_r(G_original, D_caps, 28)
    c = calculate_r_test(D_original, 64)
    d = calculate_r_test(D_caps, 28)
    print(a, b, c, d)

    r_samples = a / b
    r_test = c / d

    # r_samples = calculate_r(G_caps, D_original, 64) / calculate_r(G_original, D_caps, 28)

    # r_test = calculate_r_test(D_original, 64) / calculate_r_test(D_caps, 28)

    print("R sample is: {}".format(r_samples))
    print("R test is: {}".format(r_test))

if __name__ == "__main__":
    main()