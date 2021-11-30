import sys
sys.path.append("/content/drive/Shareddrives/ESS_Unicamp_CPqD/SER - projeto representation learning/Few-Shot_SER/")

from src.dataloader.meta_loader_lstm_mfcc import Dataloader4SER, NShotMAMLSampler, SERNShot
from src.audio.audio import AudioProcessor
from src.meta_learner.meta_learner_lstm_msl import Meta
import torch.utils.data as data
import torch
import matplotlib.pyplot as plt
import numpy as np

# DEFINING EXPERIMENT PARAMETERS
PAD_TO = 100
PAD_VALUE = -3
TRAIN_DF = '/content/drive/Shareddrives/ESS_Unicamp_CPqD/SER - projeto representation learning/Few-Shot_SER/experiments/agnostic_lstm/train.csv' # The path to a csv file with "wav_path" and "emotion" columns, for training (only two languages)
TEST_DF = '/content/drive/Shareddrives/ESS_Unicamp_CPqD/SER - projeto representation learning/Few-Shot_SER/experiments/agnostic_lstm/test.csv' # The same as training but with only data from the out-of-distribution language
UPDATE_LR = 0.05 # Learning rate of fast weight optimizations
META_LR = 0.001 # Learning rate of meta stage
N_WAY = 5 # How many classes
K_SPT = 5 # How many examples per class for training (support set)
K_QRY = 25 # How many example per class for validation (query set)
TASK_NUM = 16 # How many batches per sampling
UPDATE_STEP = 8 # How many times perform optimizations in meta stage 
UPDATE_STEP_TEST = 8 # How many times perform optimization in finetuning stage (test)
MEL_DIM = 20 # MEL DIM 
CHANNEL = 1 # Fixed
EPOCH = 3000 # How many epochs to run
LOG_PATH = "/content/drive/Shareddrives/ESS_Unicamp_CPqD/SER - projeto representation learning/Few-Shot_SER/experiments/agnostic_lstm" # Path to log the checkpointsmen
RESTORE_PATH = None
STEPS_EARLY_STOP = 200

device = 'cuda:0' # 'cpu' if dont have cuda

torch.autograd.set_detect_anomaly(True)

# Defining the audio processor
ap = AudioProcessor(fft_size = 512,
                    hop_length = 128,
                    win_length = 512,
                    pad_wav=False,
                    num_mels = 20,
                    mel_fmin = 80,
                    mel_fmax = 7600,
                    sample_rate = 16000,
                    duration = None,
                    resample = True,
                    signal_norm= True,
                    ref_level_db = 20,
                    min_level_db = -100,
                    symetric_norm = True,
                    max_norm = 4)

# Defining the dataloader for MAML
nshot = SERNShot(df_train_path = TRAIN_DF, df_test_path = TEST_DF, ap = ap, batch_size = 4, n_way = N_WAY, 
                k_shot = K_SPT, k_query = K_QRY, pad_to = PAD_TO, pad_value = PAD_VALUE)


print('experiment configs')

print(f'k_shot = {nshot.k_shot}, k_query = {nshot.k_query}, n_way = {nshot.n_way}')
# Defining args without argparse
class ARGS:
    
    def __init__(self):
    
        self.update_lr = UPDATE_LR
        self.meta_lr = META_LR
        self.n_way = N_WAY
        self.k_spt = K_SPT
        self.k_qry = K_QRY
        self.task_num = TASK_NUM
        self.update_step = UPDATE_STEP
        self.update_step_test = UPDATE_STEP_TEST
        self.imgsz = MEL_DIM
        self.imgc = CHANNEL
        self.epoch = EPOCH
        self.log_path = LOG_PATH
        self.restore_path = RESTORE_PATH
    
args = ARGS()

# Defining the config of the model

config = [
        ('conv2d', [256, 1, 3, 3, 2, 1]),
        ('relu', [True]),
        ('bn', [256]),
        ('conv2d', [256, 256, 3, 3, 2, 1]),
        ('relu', [True]),
        ('bn', [256]),
        ('flatten', []),
        ('linear', [4160,128000]),
        ('linear', [args.n_way, 4160])
    ]


# Loading meta
maml = Meta(args, config)
maml.net.to(device)


tmp = filter(lambda x: x.requires_grad, maml.parameters())
num = sum(map(lambda x: np.prod(x.shape), tmp))
print(maml)
print('Total trainable tensors:', num)

losses = []


early_stop_best_loss = 0 
early_stop_step = 0
early_stop = False

for step in range(args.epoch):

    print(f"Starting epoch {step}")

    x_spt, y_spt, x_qry, y_qry = nshot.next()
    x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).type(torch.LongTensor).to(device), \
                                    torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).type(torch.LongTensor).to(device)

    # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
    accs = maml(x_spt, y_spt, x_qry, y_qry)

    if step % 100 == 0:
        print('step:', step, '\ttraining acc:', accs)

    if step % 500 == 0:
        accs = []
        for _ in range(100//args.task_num):
            # test
            x_spt, y_spt, x_qry, y_qry = nshot.next('test')
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).type(torch.LongTensor).to(device), \
                                            torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).type(torch.LongTensor).to(device)

            # split to single task each time
            for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                accs.append( test_acc )

        # [b, update_step+1]
        stds = np.array(accs).std(axis=0).astype(np.float16)

        accs = np.array(accs).mean(axis=0).astype(np.float16)
        print('Test acc:', accs,'\tTest std:', stds) # Each position is the average test acc for each update_step_test (starting in no update i = 0)

    # Saving epoch checkpoint  
    if step % 500 == 0:
        torch.save({
                'model_state_dict': maml.net.state_dict(),
                'loss': maml.loss,
                'best_loss': maml.best_loss,
                }, args.log_path + f'/checkpoint_epoch{step}_loss{maml.loss}.pth')

    # if first best_loss (=0) then receive the first loss
    if(early_stop_best_loss == 0):
        early_stop_best_loss = maml.loss
    
    if(not early_stop):
        if(maml.loss < early_stop_best_loss):
            # print('New best loss')
            early_stop_best_loss = maml.loss   
            early_stop_step = 0
            torch.save({
            'model_state_dict': maml.net.state_dict(),
            'loss': maml.loss,
            'best_loss': maml.best_loss,
            'step': step,
            }, args.log_path + f"/best_model_early_stop.pth")

    early_stop_step += 1 

    if(early_stop_step > STEPS_EARLY_STOP):
        print(f"Early stop achieved on step {step}.")
        early_stop = True

    losses.append(maml.loss.item())

import pandas as pd
loss_df = pd.DataFrame({'losses': losses})
loss_df.to_csv(args.log_path + '/losses.csv', index = False)