import time
import argparse
import os

def make(parser):
    """Build the opt namespace used for training

    parser: parser object used to build opt
    """
    parser.add_argument('--model', default='Resnet_ae', help='model to train')
    parser.add_argument('--list', type=str, default='lists/intphys2017')
    parser.add_argument('--count', type=int, default=15000)
    parser.add_argument('--pattern', default='/mnt/20170407/train/%05d_block_O1_train',)
    parser.add_argument('--checkpoint', default='./checkpoints', help='path to checkpoint folder')
    parser.add_argument('--n_frames', type=int, default=100, help='numbers of frames in videos')
    parser.add_argument('--bsz', type=int, default=100, help='batch size')

    parser.add_argument('--input_seq', nargs='+', type=int, default=[1], help='pattern of input sequences')
    parser.add_argument('--target_seq', nargs='+', type=int, default=[1], help='pattern of input sequences')
    parser.add_argument('--full', action='store_true', help='use the maximal sequence as input and target, with a time delta between consecutive frames equal \"--delta\" and a prediction span equal \"--shift\"')
    parser.add_argument('--shift', type=int, default=0, help='prediction span, in the case \"full\" is used')
    parser.add_argument('--delta', type=int, default=1, help='distance between consecutive frames in input and target sequence, in the case \"--full\" is used')
    parser.add_argument('--mask_object', nargs='+', default=['object', 'occluder'], help='objects taken into account to build semantic masks')
    parser.add_argument('--input', default='input', help='type of image given as input, \"scene\" corresponds to the raw frame, \"mask\" corresponds to the semantic mask')
    parser.add_argument('--target', default='target', help='type of image given as input, \"scene\" corresponds to the raw frame, \"mask\" corresponds to the semantic mask')
    parser.add_argument('--px', default=0.5)
    parser.add_argument('--p_red', type=float, default=0, help='In case of semantic mask, probability to return a mask with only background. This is usefull for the discriminator of the GAN, because there are movies with no objects in the test set (and not in the training set)')
    parser.add_argument('--manualSeed', type=int, default=1)
    parser.add_argument('--nThreads', type=int, default=20, help='Number of threads used to load and process images')
    parser.add_argument('--num_channels', type=int, default=3, help='Number of channels in images')
    parser.add_argument('--frame_height', type=int, default=64)
    parser.add_argument('--frame_width', type=int, default=64)
    parser.add_argument('--mask_height', type=int, default=64)
    parser.add_argument('--mask_width', type=int, default=64)

    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--n_slices', type=int, default=1, help='Allows to split an epoch into several slices, and have intermediate checkpoints (useful when the training set is large, and each epochs lasts several hours)')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--image_save', action='store_true', help='Save image generations in the checkpoint folder')
    parser.add_argument('--image_save_interval', type=int, default=100)
    parser.add_argument('--visdom', action='store_true', help='Vizualisation with Visdom (https://github.com/facebookresearch/visdom)')
    parser.add_argument('--visdom_interval', type=int, default=10)
    parser.add_argument('--gpu', action='store_true', help='Use NVIDIA GPU')
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--load', action='append', type=lambda kv: kv.split("="), dest='load', help='Paths to trained models: for simple models use \"--load path/to/model.pth\", for GANs use \"--load netG=path/to/Generator.pth --load netD=path/to/Discriminator.pth')
    parser.add_argument('--maskPredictor', action='append', type=lambda kv: kv.split("="), dest='maskPredictor', help='Path to a mask predictor')
    parser.add_argument('--instanceNorm', action='store_true', help='Use instance normalization insead of batch normalization')
    parser.add_argument('--middleNL', default='sigmoid', help='Type of non-linearity at the middle of the network (bottleneck)')
    parser.add_argument('--finalNL', default='sigmoid', help='Type of non-linearity at the last layer of the network')
    parser.add_argument('--nc_in', type=int, default=3, help='number of channels of the input data')
    parser.add_argument('--nc_out', type=int, default=3, help='number of channels of the target data')
    parser.add_argument('--residual', action='store_true', help='use the residuals between frames as input and target')
    parser.add_argument('--rnn', default='RNN', help='type of recurrent neural net to use, in case \"--model linear_rnn\"')
    parser.add_argument('--n_hidden', type=int, default=128, help='Number of hidden units in the RNN')
    parser.add_argument('--n_layer', type=int, default=1, help='number of layer in the RNN')
    parser.add_argument('--initv', type=float, default=0.0001, help='initv parameter if the RNN')
    parser.add_argument('--nf', type=int, default=64)
    parser.add_argument('--latentDim', type=int, default=512, help='Dimension of the latent representation, at the middle of the network')

    parser.add_argument('--name', default=None, help='name given to the experiment (will appear if the name of the checkpoint folder)')

    parser.add_argument('--UpConv', default='SpatialFullConvolution', help='Type of upconvolution')
    parser.add_argument('--init_std', type=float, default=0.02, help='Standard deviation of the distribution of initial weights')
    parser.add_argument('--two_heads', action='store_true', help='Add a unconditioned head to the discriminator (in case \"--model Gan\"), see https://arxiv.org/abs/1611.06430')
    parser.add_argument('--weight_head0', default=0.5)
    parser.add_argument('--gen_bis', action='store_true', help='In the case of the GAN, discriminate between \"generated/real\" and \"generated/generated\", to enhance diversity of netG generations')
    parser.add_argument('--noiseDim', type=int, default=100)
    parser.add_argument('--lrG', type=float, default=0.0008)
    parser.add_argument('--learningRateDecayG', type=float, default=0)
    parser.add_argument('--beta1G', type=float, default=0.5)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--learningRateDecayD', type=float, default=0)
    parser.add_argument('--beta1D', type=float, default=0.5)
    parser.add_argument('--alpha1', type=float, default=1)
    parser.add_argument('--alpha2', type=float, default=1)
    parser.add_argument('--lambda', type=float, default=0)
    parser.add_argument('--target_real', type=float, default=0.9)
    parser.add_argument('--target_fake', type=float, default=0)

    opt = parser.parse_args()

    if opt.name is None:
        opt.name = opt.model

    if opt.full == 1:
        opt.input_seq = range(1, opt.n_frames - opt.shift, opt.delta)
        opt.target_seq = [x + opt.shift for x in opt.input_seq]
    opt.input_len = len(opt.input_seq)
    opt.target_len = len(opt.target_seq)

    if opt.name.find('%d%d%d%d%d%d_%d%d%d%d%d%d') == -1:
        append =  '_' + time.strftime('%y%m%d_%H%M%S')
    else:
        append = ''
    if not os.path.isdir(opt.checkpoint):
        print(opt.checkpoint, ' is not a valid directory! creating it!')
        os.mkdir(opt.checkpoint)
    opt.checkpoint = os.path.join(opt.checkpoint, opt.name + append)

    opt.m = opt.n_frames - max(opt.input_seq[-1], opt.target_seq[-1]) - int(opt.residual) + 1

    return opt

