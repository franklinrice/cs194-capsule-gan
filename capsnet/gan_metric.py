import os, time
import pickle
import torch
from model import Net
from gan import generator
import torch
import numpy as np

def calculate_r(G, D):
	batch = 500
	zs = torch.randn((batch, 100)).view(-1, 100, 1, 1)
	generated = G(zs)
	acc = np.sum(D(generated).data.numpy()) / batch
	return acc

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


	G_caps = generator(128)
	D_caps = Net(num_conv_in_channel=args.num_conv_in_channel,
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
	g_caps_dict = pickle.load(open("MNIST_DCGAN_results/generator_param.pkl", 'rb'))
	d_caps_dict = pickle.load(open("MNIST_DCGAN_results/discriminator_param.pkl", 'rb'))
	G_caps.load_state_dict(g_dict)
	D_caps.load_state_dict(d_dict)

	G_original = # fill in
	D_original = # fill in

	r_samples = calculate_r(G_caps, D_original) / calculate_r(G_original, D_caps)

	r_test = # fill in

if __name__ == "__main__":
    main()