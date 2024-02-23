import torch.utils.data as data
import torchvision.transforms as transforms
from model.utils import Reconstruction3DDataLoader, Reconstruction3DDataLoaderJump
from model.autoencoder import *
from model.discriminator_model import *
import torch.nn.functional as F
from utils import *

import argparse
import warnings
warnings.filterwarnings("ignore")

# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2" 
parser = argparse.ArgumentParser(description="gan_anomaly")
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--epochs', type=int, default=2, help='number of epochs for training')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--lr', type=float, default=1e-5, help='initial learning rate phase 1')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--dataset_type', type=str, default='ped2', choices=['ped2', 'one_people', 'five_people', 'one_people_frist', 'five_people_frist'], help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='dataset', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='basename of folder to save weights')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam','sgd'], help='adam or sgd with momentum and cosine annealing lr')

parser.add_argument('--g_model_dir', type=str, default=None, help='path of model for resume')
parser.add_argument('--old_g_model_dir', type=str, default=None, help='path of model for resume')
parser.add_argument('--d_model_dir', type=str, default=None, help='path of model for resume')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch. usually number in filename + 1')

# related to skipping frame pseudo anomaly
parser.add_argument('--pseudo_anomaly_raw', type=float, default=0.5, help='pseudo anomaly jump frame (skip frame) probability. 0 no pseudo anomaly')
parser.add_argument('--pseudo_anomaly_old', type=float, default=0.5, help='pseudo anomaly jump frame (skip frame) probability. 0 no pseudo anomaly')
parser.add_argument('--pseudo_anomaly_channle', type=float, default=0.0, help='pseudo anomaly jump frame (skip frame) probability. 0 no pseudo anomaly')
parser.add_argument('--pseudo_anomaly_mix', type=float, default=0.0, help='pseudo anomaly jump frame (skip frame) probability. 0 no pseudo anomaly')
parser.add_argument('--pseudo_anomaly_jump', type=float, default=0.0, help='pseudo anomaly jump frame (skip frame) probability. 0 no pseudo anomaly')
parser.add_argument('--pseudo_anomaly_jump_mix', type=float, default=0.0, help='pseudo anomaly jump frame (skip frame) probability. 0 no pseudo anomaly')
parser.add_argument('--jump', nargs='+', type=int, default=[3], help='Jump for pseudo anomaly (hyperparameter s)')  # --jump 2 3

parser.add_argument('--print_all', action='store_true', help='print all reconstruction loss')
'''
python train_two_phase_img.py --dataset_type one_people_frist  --jump 2 3 4 5 \
    --old_g_model_dir  /root/gan_anomaly/exp/one_people_frist/loglr0.0008weight_recon_pajump0.01_jump[2,3,4,5]/model_20.pth \
    --g_model_dir /root/gan_anomaly/exp/one_people_frist/loglr0.0008weight_recon_pajump0.01_jump[2,3,4,5]/model_59.pth \
    --lr 1e-5 --exp_dir modeImg30
    --old_g_model_dir  /root/gan_anomaly/exp/one_people_frist/loglr0.0008weight_recon_pajump0.01_jump[2,3,4,5]/model_20.pth \
    
    --d_model_dir  /root/gan_anomaly/code/pretain_model_3_00.pth \
'''
np.random.seed(0)
##################

args = parser.parse_args()
# assert 1 not in args.jump

exp_dir = args.exp_dir
exp_dir += '_lr' + str(args.lr) 
exp_dir += '_raw' + str(args.pseudo_anomaly_raw) + '_old' + str(args.pseudo_anomaly_old) + '_channle' + str(args.pseudo_anomaly_channle)
exp_dir += '_mix' + str(args.pseudo_anomaly_mix) + '_jump' + str(args.pseudo_anomaly_jump) + '_jump_mix' + str(args.pseudo_anomaly_jump_mix)

exp_dir += '_jump[' + ','.join([str(args.jump[i]) for i in range(0,len(args.jump))]) + ']' if args.pseudo_anomaly_jump != 0 else ''

print('exp_dir: ', exp_dir)
import torch

def gaussian(ins, is_training=1, mean=0, stddev=0.9**2):
    if is_training:
        noise = ins.data.new(ins.size()).normal_(mean, stddev)
        noisy_image = ins + noise
        if noisy_image.max().data > 1 or noisy_image.min().data < -1:
            noisy_image = torch.clamp(noisy_image, -1, 1)
            if noisy_image.max().data > 1 or noisy_image.min().data < -1:
                raise Exception('input image with noise has values larger than 1 or smaller than -1')
        return noisy_image
    return ins

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

train_folder = os.path.join(args.dataset_path, args.dataset_type, 'training', 'frames')

# Loading dataset
img_extension = '.tif' if args.dataset_type == 'ped1' else '.jpg'
train_dataset = Reconstruction3DDataLoader(train_folder, transforms.Compose([transforms.ToTensor()]),
                                           resize_height=args.h, resize_width=args.w, dataset=args.dataset_type, img_extension=img_extension)
train_dataset_jump = Reconstruction3DDataLoaderJump(train_folder, transforms.Compose([transforms.ToTensor()]),
                                                resize_height=args.h, resize_width=args.w, dataset=args.dataset_type, jump=args.jump, img_extension=img_extension)
# print(train_dataset.shape)
# print(train_dataset_jump.shape)
train_size = len(train_dataset)
print(train_size)
train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=True)
train_batch_jump = data.DataLoader(train_dataset_jump, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, drop_last=True)

# Report the training process
log_dir = os.path.join('/root/autodl-tmp/weight', args.dataset_type, exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
orig_stdout = sys.stdout
f = open(os.path.join(log_dir, 'log.txt'), 'a')
sys.stdout = f

torch.set_printoptions(profile="full")

print(args)
# loss_func_mse = nn.MSELoss(reduction='none')

if args.start_epoch < args.epochs:
    g_model = convAE()
    if args.old_g_model_dir is not None:
        old_g_model = convAE()
    d_model = d_net_img()
    g_model = nn.DataParallel(g_model)
    d_model = nn.DataParallel(d_model)

    if args.old_g_model_dir is not None:
        old_g_model = nn.DataParallel(old_g_model)
    g_model.cuda()
    d_model.cuda()
    g_optimizer = torch.optim.Adam(g_model.parameters(), lr=args.lr)
    d_optimizer = torch.optim.Adam(d_model.parameters(), lr=args.lr)

    # load_g_model
    if args.g_model_dir is not None:
        model_dict = torch.load(args.g_model_dir)
        model_weight = model_dict['model']
        g_model.load_state_dict(model_weight.state_dict())
        g_optimizer.load_state_dict(model_dict['optimizer'])
        g_model.cuda()
    
    # load_old_g_model
    if args.old_g_model_dir is not None:
        model_dict = torch.load(args.old_g_model_dir)
        model_weight = model_dict['model']
        old_g_model.load_state_dict(model_weight.state_dict())
        # d_optimizer.load_state_dict(model_dict['optimizer'])
        old_g_model.cuda()

    # load_d_model
    if args.d_model_dir is not None:
        model_dict = torch.load(args.d_model_dir)
        model_weight = model_dict['model']
        d_model.load_state_dict(model_weight.state_dict())
        # d_optimizer.load_state_dict(model_dict['optimizer'])
        d_model.cuda()

    resnet_transform = transforms.Compose([
        transforms.Resize(224)
    ])
    g_model.eval()
    for epoch in range(args.start_epoch, args.epochs):
        d_model.train()
        pseudolossepoch = 0
        lossepoch = 0
        pseudolosscounter = 0
        losscounter = 0

        for j, (imgs, imgsjump) in enumerate(zip(train_batch, train_batch_jump)):
            net_in = copy.deepcopy(imgs)
            old_net_in = copy.deepcopy(imgs)
            net_in = net_in.cuda()
            imgsjump = imgsjump.cuda()
            old_net_in = old_net_in.cuda()
            ########## TRAIN
            g_outputs = g_model(net_in)

            # raw
            label_real = torch.ones([args.batch_size], dtype=torch.float32).unsqueeze(1).cuda() * 1
            g_outputs_8 = torch.cat((g_outputs[:, :, 8].detach().clone(),\
                                    g_outputs[:, :, 8].detach().clone(),\
                                        g_outputs[:, :, 8].detach().clone()),1)
            g_outputs_real = resnet_transform(g_outputs_8)
            d_outputs_real = d_model(g_outputs_real.detach().clone())
            loss = args.pseudo_anomaly_raw * F.binary_cross_entropy(d_outputs_real, label_real)

            # old is glad
            if args.pseudo_anomaly_old > 0:
                label_fake_augmented_old = torch.ones([args.batch_size], dtype=torch.float32).unsqueeze(1).cuda() * 0
                old_g_outputs = old_g_model(old_net_in)
                g_outputs_8 = torch.cat((old_g_outputs[:, :, 8].detach().clone(),\
                                        old_g_outputs[:, :, 8].detach().clone(),\
                                            old_g_outputs[:, :, 8].detach().clone()),1)
                # g_outputs_8 = gaussian(g_outputs_8)
                g_outputs_old = resnet_transform(g_outputs_8)
                d_outputs_old = d_model(g_outputs_old.detach().clone())
                loss += args.pseudo_anomaly_old * F.binary_cross_entropy(d_outputs_old, label_fake_augmented_old)

            # channle
            if args.pseudo_anomaly_channle > 0:
                label_fake_augmented_channle = torch.ones([args.batch_size], dtype=torch.float32).unsqueeze(1).cuda() * 0
                g_outputs_channle = torch.cat((g_outputs[:, :, 0].detach().clone(),\
                                        g_outputs[:, :, 8].detach().clone(),\
                                            g_outputs[:, :, 15].detach().clone()),1)
                g_outputs_channle = resnet_transform(g_outputs_channle)
                d_outputs_channle = d_model(g_outputs_channle.detach().clone())
                loss += args.pseudo_anomaly_channle * F.binary_cross_entropy(d_outputs_channle, label_fake_augmented_channle)
            
            # mix
            if args.pseudo_anomaly_mix > 0:
                label_fake_augmented_mix = torch.ones([args.batch_size], dtype=torch.float32).unsqueeze(1).cuda() * 0
                g_outputs_0 = torch.cat((g_outputs[:, :, 0].detach().clone(),\
                                        g_outputs[:, :, 0].detach().clone(),\
                                            g_outputs[:, :, 0].detach().clone()),1)
                g_outputs_15 = torch.cat((g_outputs[:, :, 15].detach().clone(),\
                                        g_outputs[:, :, 15].detach().clone(),\
                                        g_outputs[:, :, 15].detach().clone()),1)
                g_outputs_mix = resnet_transform((g_outputs_0 + g_outputs_15)/2)
                d_outputs_mix = d_model(g_outputs_mix.detach().clone())
                loss += args.pseudo_anomaly_mix * F.binary_cross_entropy(d_outputs_mix, label_fake_augmented_mix)

            # jump
            if args.pseudo_anomaly_jump > 0:
                label_fake_augmented_jump = torch.ones([args.batch_size], dtype=torch.float32).unsqueeze(1).cuda() * 0
                g_outputs_jump = g_model(imgsjump)
                # 可能存在jump和mix混合的情况
                rand_number = np.random.rand()

                # skip frame pseudo anomaly
                pseudo_anomaly_jump_mix = 0 <= rand_number < args.pseudo_anomaly_jump_mix
                if pseudo_anomaly_jump_mix:
                    g_outputs_0 = torch.cat((g_outputs_jump[:, :, 0].detach().clone(),\
                                        g_outputs_jump[:, :, 0].detach().clone(),\
                                            g_outputs_jump[:, :, 0].detach().clone()),1)
                    g_outputs_15 = torch.cat((g_outputs_jump[:, :, 15].detach().clone(),\
                                            g_outputs_jump[:, :, 15].detach().clone(),\
                                            g_outputs_jump[:, :, 15].detach().clone()),1)
                    g_outputs_jump = resnet_transform((g_outputs_0 + g_outputs_15)/2)
                else:
                    g_outputs_8 = torch.cat((g_outputs[:, :, 8].detach().clone(),\
                                            g_outputs[:, :, 8].detach().clone(),\
                                                g_outputs[:, :, 8].detach().clone()),1)
                    g_outputs_jump = resnet_transform(g_outputs_8)
                d_outputs_jump = d_model(g_outputs_jump.detach().clone())
                loss += args.pseudo_anomaly_jump * F.binary_cross_entropy(d_outputs_jump, label_fake_augmented_jump)
            
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            loss.backward()
            d_optimizer.step()
            if j % 10 == 0:    
                model_dict = {
                    'model': d_model,
                    'optimizer': d_optimizer.state_dict(),
                }    
                torch.save(model_dict, os.path.join(log_dir, 'd_model_{:02d}_{:03d}.pth'.format(epoch, j)))
            if j % 10 == 0 or args.print_all:
                
                print("epoch {:d} iter {:d}/{:d}".format(epoch, j, len(train_batch)))
                print('Loss: {:.6f}'.format(loss.item()))

        print('----------------------------------------')
        print('Epoch:', epoch)
        # Save the model and the memory items
        model_dict = {
            'model': d_model,
            'optimizer': d_optimizer.state_dict(),
        }
        # if (epoch%2) == 0 :
        torch.save(model_dict, os.path.join(log_dir, 'd_model_{:02d}.pth'.format(epoch)))

print('Training is finished')
sys.stdout = orig_stdout
f.close()



'''
CUDA_VISIBLE_DEVICES=1,2 python train.py --dataset_type one_people_frist
CUDA_VISIBLE_DEVICES=3,4 python train.py --dataset_type one_people_frist --pseudo_anomaly_jump 0.01 --jump 2 3 4 5
CUDA_VISIBLE_DEVICES=5,6 python train.py --dataset_type five_people_frist
CUDA_VISIBLE_DEVICES=6,7 python train.py --dataset_type five_people_frist --pseudo_anomaly_jump 0.01 --jump 2 3 4 5
python train_two_phase_img.py --dataset_type five_people_frist  --pseudo_anomaly_jump 0.01 --jump 2 3 4 5 \
    --g_model_dir /root/gan_anomaly/exp/one_people_frist/loglr0.0008weight_recon_pajump0.01_jump[2,3,4,5]/model_59.pth \
    --old_g_model_dir  /root/gan_anomaly/exp/one_people_frist/loglr0.0008weight_recon_pajump0.01_jump[2,3,4,5]/model_20.pth \
    # --d_model_dir  /root/gan_anomaly/code/pretain_model_3_00.pth \
    --lr 1e-5 --exp_dir modeImg9
'''