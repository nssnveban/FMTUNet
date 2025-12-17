import numpy as np
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random
import time
import itertools
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils import *
from torch.autograd import Variable
from IPython.display import clear_output
from model.modeling import VisionTransformer as ViT_seg
from model.modeling import create_model
from model.modeling import CONFIGS as CONFIGS_seg
try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pynvml import *
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(int(os.environ["CUDA_VISIBLE_DEVICES"]))
print("Device :", nvmlDeviceGetName(handle))
import warnings
warnings.filterwarnings("ignore")

config_vit = CONFIGS_seg['R50-ViT-B_16']
config_vit.n_classes = 6
config_vit.n_skip = 3
config_vit.patches.grid = (int(256 / 16), int(256 / 16))


net = create_model(
    config_name='R50-ViT-B_16',
    img_size=256,
    num_classes=6,
    skip_deep_fusion=False,  
    use_basic_skip=False  
).cuda()


net.load_from(weights=np.load(config_vit.pretrained_path))
print("Pre-trained weights loading completed")

params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print(params)
# Load the datasets

print("training : ", train_ids)
print("testing : ", test_ids)
print("BATCH_SIZE: ", BATCH_SIZE)
print("Stride Size: ", Stride_Size)
train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)

base_lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
# We define the scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 30, 40], gamma=0.1)


def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    ## Potsdam
    if DATASET == 'Potsdam':
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32') for id in test_ids)
        # test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, (3, 0, 1, 2)][:, :, :3], dtype='float32') for id in test_ids)
    ## Vaihingen
    else:
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []

    # Switch the network to inference mode
    with torch.no_grad():
        for img, dsm, gt, gt_e in tqdm(zip(test_images, test_dsms, test_labels, eroded_labels), total=len(test_ids), leave=False):
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                        leave=False)):
                # Build the tensor
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

                min = np.min(dsm)
                max = np.max(dsm)
                dsm = (dsm - min) / (max - min)
                dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                dsm_patches = np.asarray(dsm_patches)
                dsm_patches = Variable(torch.from_numpy(dsm_patches).cuda(), volatile=True)

                # Do the inference
                outs = net(image_patches, dsm_patches)
                outs = outs.data.cpu().numpy()

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt_e)
            clear_output()
            
    accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                       np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1):
    save_dir = "./results_posd"
    os.makedirs(save_dir, exist_ok=True) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    weights = weights.to(device)

    #criterion = nn.NLLLoss2d(weight=weights)
    criterion = nn.CrossEntropyLoss(weight=weights)    
    acc_best = 0.0
    miou_best = 0.0
    log_data = [] 
    batch_count = 0

    for e in range(1, epochs + 1):
        net.train()
        epoch_losses = []
        

        for batch_idx, (data, dsm, target) in enumerate(train_loader):
            data, dsm, target = data.to(device), dsm.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = net(data, dsm) 
            
            logits = output['logits'] if isinstance(output, dict) else output
            features = output.get('features') if isinstance(output, dict) else None
            
            loss = criterion(logits, target)
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            batch_count += 1
            
            if batch_idx % 100 == 0:
                if isinstance(output, dict):
                    pred = output['logits'].argmax(dim=1).cpu().numpy()
                else:
                    pred = output.argmax(dim=1).cpu().numpy()
                gt = target.cpu().numpy()
                acc = accuracy(pred, gt)
                print(f'Epoch {e} [Batch {batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}, Acc: {acc:.2f}%')
            
            if (batch_idx + 1) % 500 == 0:
                net.eval()
                metrics = test(net, test_ids, all=False, stride=Stride_Size)
                net.train()
                
                print(f'--- Validation at Epoch {e}, Batch {batch_idx+1} ---')
                print(f"Accuracy: {metrics['accuracy']:.2f}%, Mean F1: {metrics['mean_F1Score']:.4f}, Mean IoU: {metrics['mean_IoU']:.4f}")

                log_entry = {
                    'epoch': e,
                    'batch': batch_idx + 1,
                    'accuracy': metrics['accuracy'],
                    'mean_F1Score': metrics['mean_F1Score'],
                    'mean_IoU': metrics['mean_IoU']
                }
                log_data.append(log_entry)
                with open(f'{save_dir}/training_log.json', 'w') as f:
                    json.dump(log_data, f, indent=4)

                if metrics['accuracy'] > acc_best: #can switch to "metrics['mean_IoU']" and "miou_best"
                    torch.save(net.state_dict(), './results_posd/posd_epoch{}_{}'.format(e, metrics['accuracy']))
                    acc_best = metrics['accuracy']
                    print(f"New best accuracy: {acc_best:.2f}%, model saved.")
                else:
                    print(f"Current accuracy: {metrics['accuracy']:.2f}%, Best: {acc_best:.2f}%")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f'Epoch {e} Average Loss: {avg_loss:.4f}')

        if scheduler is not None:
            scheduler.step()
            print(f'Current learning rate: {scheduler.get_last_lr()[0]:.6f}')

    print(f'Training completed. Best accuracy: {acc_best:.2f}%')

#####   train   ####
time_start=time.time()
train(net, optimizer, 50, scheduler)
time_end=time.time()
print('Total Time Cost: ',time_end-time_start)

#####   test   ####
#net.load_state_dict(torch.load('./vg92.4.pth'))
#net.eval()
#acc, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
#print("Acc: ", acc)
#for p, id_ in zip(all_preds, test_ids):
    #img = convert_to_color(p)
    #plt.imshow(img) and plt.show()
    #io.imsave('./results/inference_tile{}.png'.format(id_), img)
#print("done!")




