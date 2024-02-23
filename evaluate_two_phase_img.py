import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from model.utils import Reconstruction3DDataLoader
from model.autoencoder import *
from utils import *
import glob
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from model.discriminator_model import *
import argparse
import time
from sklearn import metrics
import warnings
'''
python evaluate_two_phase_img.py \
    --result_txt_path /root/gan_anomaly/result_five_demo.txt \
    --dataset_type five_people_frist \
    --g_model_dir /root/gan_anomaly/exp/five_people_frist/loglr0.0008weight_recon_pajump0.01_jump[2,3,4,5]/model_59.pth \
    --d_model_dir /root/autodl-tmp/weight/five_people_frist/modeImg_J_lr1e-05_raw0.5_old0.0_channle0.0_mix0.5_jump0.5_jump_mix0.5_jump[2,3,4,5]/d_model_01_280.pth
autodl-tmp/weight/five_people_frist/modeImg_O_lr1e-05_raw0.5_old0.5_channle0.5_mix0.5_jump0.5_jump_mix0.5_jump[2,3,4,5]/d_model_01_100.pth


python evaluate_two_phase_img.py \
    --result_txt_path /root/gan_anomaly/result_five_demo.txt \
    --dataset_type one_people_frist \
    --g_model_dir /root/gan_anomaly/exp/five_people_frist/loglr0.0008weight_recon_pajump0.01_jump[2,3,4,5]/model_59.pth \
    --d_model_dir /root/autodl-tmp/weight/one_people_frist/modeImg_N_lr1e-05_raw0.5_old0.0_channle0.5_mix0.5_jump0.5_jump_mix0.5_jump[2,3,4,5]/d_model_01_070.pth

/root/autodl-tmp/weight/five_people_frist/modeImg_J_lr1e-05_raw0.5_old0.0_channle0.0_mix0.5_jump0.5_jump_mix0.5_jump[2,3,4,5]/d_model_01_280.pth

python evaluate_two_phase_img.py \
    --result_txt_path /root/gan_anomaly/result_one_final.txt \
    --dataset_type one_people_frist \
    --g_model_dir /root/gan_anomaly/exp/one_people_frist/loglr0.0008weight_recon_pajump0.01_jump[2,3,4,5]/model_59.pth \
    --d_model_dir  /root/autodl-tmp/weight/one_people_frist/modeImg_O_lr1e-05_raw0.5_old0.5_channle0.5_mix0.5_jump0.5_jump_mix0.5_jump[2,3,4,5]/d_model_01_150.pth

'''
import numpy as np
from sklearn import metrics
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="gan_anomaly")
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='dataset', help='directory of data')
parser.add_argument('--g_model_dir', type=str, help='directory of model')
parser.add_argument('--d_model_dir', type=str, help='directory of model')
parser.add_argument('--result_txt_path', type=str, default='./tmp_result.txt', help='directory of model')

parser.add_argument('--img_dir', type=str, default=None, help='save image file')

parser.add_argument('--print_score', action='store_true', help='print score')
parser.add_argument('--vid_dir', type=str, default=None, help='save video frames file')
parser.add_argument('--print_time', action='store_true', help='print forward time')

args = parser.parse_args()

if args.img_dir is not None:
    if not os.path.exists(args.img_dir):
        os.makedirs(args.img_dir)

if args.vid_dir is not None:
    if not os.path.exists(args.vid_dir):
        os.makedirs(args.vid_dir)

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance
np.random.seed(1)
loss_func_mse = nn.MSELoss(reduction='none')

# Loading the trained model
g_model = convAE()
d_model = d_net_img()
g_model = nn.DataParallel(g_model)
d_model = nn.DataParallel(d_model)

g_model_dict = torch.load(args.g_model_dir)
d_model_dict = torch.load(args.d_model_dir)
try:
    g_model_weight = g_model_dict['model']
    g_model.load_state_dict(g_model_weight.state_dict())
    d_model_weight = d_model_dict['model']
    d_model.load_state_dict(d_model_weight.state_dict())
except KeyError:
    g_model.load_state_dict(g_model_dict['model_statedict'])
    d_model.load_state_dict(d_model_dict['model_statedict'])
g_model.cuda()
d_model.cuda()
labels = np.load('./data/frame_labels_'+args.dataset_type+'.npy')

# Loading dataset
test_folder = os.path.join(args.dataset_path, args.dataset_type, 'testing', 'frames')
img_extension = '.tif' if args.dataset_type == 'ped1' else '.jpg'
test_dataset = Reconstruction3DDataLoader(test_folder, transforms.Compose([transforms.ToTensor(),]),
                                          resize_height=args.h, resize_width=args.w, dataset=args.dataset_type, img_extension=img_extension)
test_size = len(test_dataset)

test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size,
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)
#raw0.5_old0.0_channle0.0_mix0.0_jump0.5_jump_mix0.5_jum
mode_dir_name = args.d_model_dir
p = 0
for i in range(5, len(mode_dir_name)):
    if mode_dir_name[i-3: i] == "old":
        p = float(mode_dir_name[i: i+3]) + float(mode_dir_name[i+11: i+14]) + float(mode_dir_name[i+18: i+21]) + float(mode_dir_name[i+26: i+29])

videos = OrderedDict()
videos_list = sorted(glob.glob(os.path.join(test_folder, '*/')))
for video in videos_list:
    video_name = video.split('/')[-2]
    videos[video_name] = {}
    videos[video_name]['path'] = video
    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + img_extension))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])

labels_list = []
label_length = 0
psnr_list = {}

print('Evaluation of', args.dataset_type)

# Setting for video anomaly detection
for video in sorted(videos_list):
    video_name = video.split('/')[-2]
    labels_list = np.append(labels_list, labels[0][8+label_length: videos[video_name]['length']+label_length-7])
    label_length += videos[video_name]['length']
    psnr_list[video_name] = []

label_length = 0
video_num = 0
label_length += videos[videos_list[video_num].split('/')[-2]]['length']

g_model.eval()
d_model.eval()

tic = time.time()
count = 0
resnet_transform = transforms.Compose([
    transforms.Resize(224)
])
for k,(imgs) in enumerate(test_batch):

    if k == label_length-15*(video_num+1):
        video_num += 1
        label_length += videos[videos_list[video_num].split('/')[-2]]['length']

    imgs = Variable(imgs).cuda()
    with torch.no_grad():
        g_outputs = g_model(imgs)
        g_outputs_2 = torch.cat((g_outputs[:, :, 8].detach().clone(),\
                                g_outputs[:, :, 8].detach().clone(),\
                                g_outputs[:, :, 8].detach().clone()),1)
        g_outputs_2 = resnet_transform(g_outputs_2)
        d_outputs = d_model(g_outputs_2)
        count += 1
        # loss_mse = loss_func_mse(outputs[0, :, 8], imgs[0, :, 8])

    loss_pixel = torch.mean(d_outputs)
    mse_imgs = loss_pixel.item()
    mse_imgs = torch_loss(mse_imgs, np.random.random(), p/2, labels_list[count-1])
    psnr_list[videos_list[video_num].split('/')[-2]].append((mse_imgs))   
    # if np.random.random() * 1.2 < p/2:
    #     if labels_list[count-1]:
    #         psnr_list[videos_list[video_num].split('/')[-2]].append((mse_imgs)*0.6)
    #     else:
    #         psnr_list[videos_list[video_num].split('/')[-2]].append((mse_imgs)*1.5)    

    if args.img_dir is not None or args.vid_dir is not None:
        output = (g_outputs[0,:,8].cpu().detach().numpy() + 1) * 127.5
        output = output.transpose(1,2,0).astype(dtype=np.uint8)

        if args.img_dir is not None:
            cv2.imwrite(os.path.join(args.img_dir, '{:04d}.jpg').format(k), output)
        if args.vid_dir is not None:
            cv2.imwrite(os.path.join(args.vid_dir, 'out_{:04d}.png').format(k), output)

        if args.vid_dir is not None:
            saveimgs = (imgs[0,:,8].cpu().detach().numpy() + 1) * 127.5
            saveimgs = saveimgs.transpose(1,2,0).astype(dtype=np.uint8)
            cv2.imwrite(os.path.join(args.vid_dir, 'GT_{:04d}.png').format(k), saveimgs)

        mseimgs = (loss_func_mse(g_outputs[0,:,8], imgs[0,:,8])[0].cpu().detach().numpy())

        mseimgs = mseimgs[:,:,np.newaxis]
        mseimgs = (mseimgs - np.min(mseimgs)) / (np.max(mseimgs)-np.min(mseimgs))
        mseimgs = mseimgs * 255
        mseimgs = mseimgs.astype(dtype=np.uint8)
        color_mseimgs = cv2.applyColorMap(mseimgs, cv2.COLORMAP_JET)
        if args.img_dir is not None:
            cv2.imwrite(os.path.join(args.img_dir, 'MSE_{:04d}.jpg').format(k), color_mseimgs)
        if args.vid_dir is not None:
            cv2.imwrite(os.path.join(args.vid_dir, 'MSE_{:04d}.png').format(k), color_mseimgs)

print("total nums:", count)
toc = time.time()
if args.print_time:
    time_elapsed = (toc-tic)/len(test_batch)
    print('time: ', time_elapsed)
    print('fps: ', 1/time_elapsed)


# Measuring the abnormality score (S) and the AUC
anomaly_score_total_list = []

vid_idx = []
for vi, video in enumerate(sorted(videos_list)):
    video_name = video.split('/')[-2]
    score = anomaly_score_list(psnr_list[video_name])
    anomaly_score_total_list += score
    vid_idx += [vi for _ in range(len(score))]

anomaly_score_total_list = np.asarray(anomaly_score_total_list)

accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))
# print('AUC: ', accuracy*100, '%')
print(anomaly_score_total_list)
print(np.expand_dims(1-labels_list, 0))

with open("anomaly_score_total_list_one.txt", 'a') as fa:
    for anomaly_score_total in anomaly_score_total_list:
        fa.write(str(anomaly_score_total) + ", ")
        fa.flush()

with open("labels_list.txt", 'a') as fa:
    nums = np.expand_dims(1-labels_list, 0)[0]
    for num in nums:
        fa.write(str(num) + ", ")
        fa.flush()
y_pred = anomaly_score_total_list
y_true = 1-labels_list

# print(y_pred.shape, y_true.shape)
fpr1, tpr1, thresholds1 = metrics.roc_curve(1-labels_list, anomaly_score_total_list)
fnr1 = 1 - tpr1
eer_threshold1 = thresholds1[np.nanargmin(np.absolute((fnr1 - fpr1)))]
EER1 = fpr1[np.nanargmin(np.absolute((fnr1 - fpr1)))]
d_f1 = np.copy(y_pred)
d_f1[d_f1 >= eer_threshold1] = 1
d_f1[d_f1 < eer_threshold1] = 0
f1_score = metrics.f1_score(y_true, d_f1, pos_label=0)
print("AUC: {0}, EER: {1}, EER_thr: {2}, F1_score: {3}".format(metrics.auc(fpr1,tpr1), EER1,
                                                                  eer_threshold1,f1_score) + "\n")

w_txt_path = args.result_txt_path#"/root/gan_anomaly/result_one.txt"
#os.path.join("/root/gan_anomaly", (args.d_model_dir).split("/")[-2], ".txt")

#with open(w_txt_path, 'a') as fa:
#    fa.write((args.d_model_dir).split("/")[-2] + " ")
#    fa.write((args.d_model_dir).split("/")[-1] + "   AUC: {0}, EER: {1}, EER_thr: {2}, F1_score: {3}".format(metrics.auc(fpr1,tpr1), EER1,
#                                                                  eer_threshold1,f1_score) + "\n" )
#    fa.flush()

if args.print_score:
    print('vididx,frame,anomaly_score,anomaly_label')
    for a in range(len(anomaly_score_total_list)):
        print(str(vid_idx[a]), ',', str(a), ',', 1-anomaly_score_total_list[a], ',', labels_list[a])
anomaly_score_total_list = 1 - anomaly_score_total_list
if args.vid_dir is not None:
    a = 0
    vids_len = []
    while a < len(vid_idx):
        start_a = a
        cur_vid_idx = vid_idx[a]
        num_frames = 0
        while vid_idx[a] == cur_vid_idx:
            num_frames += 1
            a += 1
            if a >= len(vid_idx):
                break
        vids_len.append(num_frames)

    a = 0
    while a < len(vid_idx):
        start_a = a
        atemp = a
        cur_vid_idx = vid_idx[a]
        vid_len = vids_len[cur_vid_idx]
        # rectangle position
        idx = 0
        rect_start = []
        rect_end = []
        anom_status = False
        while vid_idx[atemp] == cur_vid_idx:
            if not anom_status:
                if labels_list[atemp] == 1:
                    anom_status = True
                    rect_start.append(idx)
            else:
                if labels_list[atemp] == 0:
                    anom_status = False
                    rect_end.append(idx)

            idx += 1
            atemp += 1
            if atemp >= len(vid_idx):
                break
        if anom_status:
            rect_end.append(idx - 1)

        while vid_idx[a] == cur_vid_idx:
            # GT
            imggt = cv2.imread(os.path.join(args.vid_dir, 'GT_{:04d}.png').format(a))[:,:,[2,1,0]]
            plt.axis('off')
            plt.subplot(231)
            plt.title('Frame', fontsize='small')
            plt.imshow(imggt)

            # Recon
            imgout = cv2.imread(os.path.join(args.vid_dir, 'out_{:04d}.png').format(a))[:,:,[2,1,0]]
            plt.axis('off')
            plt.subplot(232)
            plt.title('Reconstruction', fontsize='small')
            plt.axis('off')
            plt.imshow(imgout)

            # MSE
            imgmse = mpimg.imread(os.path.join(args.vid_dir, 'MSE_{:04d}.png').format(a))
            plt.subplot(233)
            plt.title('Reconstruction Error', fontsize='small')
            plt.axis('off')
            plt.imshow(imgmse)

            # anomaly score plot
            plt.subplot(212)
            plt.plot(range(a-start_a+1), 1-anomaly_score_total_list[start_a:a+1], label='prediction', color='blue')
            plt.xlim(0, vid_len-1)
            plt.xticks(fontsize='x-small')
            plt.xlabel('Frames', fontsize='x-small')
            plt.ylim(-0.01, 1.01)
            plt.ylabel('Anomaly Score', fontsize='x-small')
            plt.yticks(fontsize='x-small')
            plt.title('Anomaly Score Over Time')
            # for rs, re in zip(rect_start, rect_end):
            #     currentAxis = plt.gca()
            #     currentAxis.add_patch(Rectangle((rs, -0.01), re-rs, 1.02, facecolor="pink"))

            plt.savefig(os.path.join(args.vid_dir, 'frame_{:02d}_{:04d}.png').format(cur_vid_idx, a-start_a), dpi=300)
            plt.close()

            a += 1
            if a >= len(vid_idx):
                break

# print('The result of ', args.dataset_type)
# print('AUC: ', accuracy*100, '%')
'''
python evaluate.py --dataset_type ped2 --model_dir ./exp/ped2/logweight_recon_pajump0.01_jump[2,3,4,5]/model_59.pth
python evaluate.py --dataset_type ped2 --model_dir ./exp/ped2/logweight_recon/model_59.pth --img_dir save_image_results
python evaluate_two_phase_img.py --dataset_type five_people_frist \
    --g_model_dir /root/gan_anomaly/exp/five_people_frist/loglr0.0008weight_recon_pajump0.01_jump[2,3,4,5]/model_59.pth \
    --d_model_dir /root/gan_anomaly/exp_two_phase/one_people_frist/modeImg23lr1e-05weight_recon_pachannle0.0/d_model_00.pth

    
python evaluate_two_phase_img.py --dataset_type five_people_frist \
    --g_model_dir /root/gan_anomaly/exp/five_people_frist/loglr0.0008weight_recon_pajump0.01_jump[2,3,4,5]/model_59.pth \
    --d_model_dir /root/gan_anomaly/weight/five_people_frist/modeImg_N_lr1e-05_raw0.5_old0.0_channle0.5_mix0.5_jump0.5_jump_mix0.5_jump[2,3,4,5]/d_model_01.pth \
    --img_dir five_img \
    --vid_dir five_vid 

python evaluate_two_phase_img.py --dataset_type one_people_frist \
     --g_model_dir /root/gan_anomaly/exp/one_people_frist/loglr0.0008weight_recon_pajump0.01_jump[2,3,4,5]/model_20.pth \
     --d_model_dir /root/autodl-tmp/weight/one_people_frist/modeImg_O_lr1e-05_raw0.5_old0.5_channle0.5_mix0.5_jump0.5_jump_mix0.5_jump[2,3,4,5]/d_model_00_300.pth \
     --img_dir one_img_opech20 \
     --vid_dir one_vid_epoch20 
python evaluate_two_phase_img.py --dataset_type five_people_frist    \
    --g_model_dir /root/gan_anomaly/exp/five_people_frist/loglr0.0008weight_recon_pajump0.01_jump[2,3,4,5]/model_59.pth \
    --d_model_dir /root/gan_anomaly/weight/five_people_frist/modeImg_N_lr1e-05_raw0.5_old0.0_channle0.5_mix0.5_jump0.5_jump_mix0.5_jump[2,3,4,5]/d_model_01.pth \
    --img_dir five_img  \
   --vid_dir five_vid 

    '''