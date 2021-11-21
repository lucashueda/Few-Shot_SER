import sys
sys.path.append("/content/drive/Shareddrives/ESS_Unicamp_CPqD/SER - projeto representation learning/Few-Shot_SER/")

from src.dataloader.meta_loader import Dataloader4SER, NShotMAMLSampler, SERNShot
from src.audio.audio import AudioProcessor
from src.meta_learner.meta_learner import Meta
import torch.utils.data as data
import torch
import matplotlib.pyplot as plt
import numpy as np

# DEFINING EXPERIMENT PARAMETERS
PAD_TO = 200 
PAD_VALUE = -3
TRAIN_DF = '/content/drive/Shareddrives/ESS_Unicamp_CPqD/SER - projeto representation learning/Few-Shot_SER/experiments/train-en-es/train.csv' # The path to a csv file with "wav_path" and "emotion" columns, for training (only two languages)
TEST_DF = '/content/drive/Shareddrives/ESS_Unicamp_CPqD/SER - projeto representation learning/Few-Shot_SER/experiments/train-en-es/test.csv' # The same as training but with only data from the out-of-distribution language
UPDATE_LR = 0.4 # Learning rate of fast weight optimizations
META_LR = 0.001 # Learning rate of meta stage
N_WAY = 5 # How many classes
K_SPT = 5 # How many examples per class for training (support set)
K_QRY = 2 # How many example per class for validation (query set)
TASK_NUM = 16 # How many batches per sampling
UPDATE_STEP = 5 # How many times perform optimizations in meta stage 
UPDATE_STEP_TEST = 50 # How many times perform optimization in finetuning stage (test)
MEL_DIM = 80 # MEL DIM 
CHANNEL = 1 # Fixed
EPOCH = 1000 # How many epochs to run
LOG_PATH = "/content/drive/Shareddrives/ESS_Unicamp_CPqD/SER - projeto representation learning/Few-Shot_SER/experiments/train-en-es" # Path to log the checkpointsmen
RESTORE_PATH = None

device = 'cuda:0' # 'cpu' if dont have cuda

# Defining the audio processor
ap = AudioProcessor(fft_size = 1024,
                    hop_length = 256,
                    win_length = 1024,
                    pad_wav=False,
                    num_mels = 80,
                    mel_fmin = 80,
                    mel_fmax = 7600,
                    sample_rate = 22050,
                    duration = None,
                    resample = True,
                    signal_norm= True,
                    ref_level_db = 20,
                    min_level_db = -100,
                    symetric_norm = True,
                    max_norm = 4)

# Defining the dataloader for MAML
nshot = SERNShot(df_train_path = TRAIN_DF, df_test_path = TEST_DF, ap = ap, batch_size = 2, n_way = N_WAY, 
                k_shot = K_SPT, k_query = K_QRY, pad_to = PAD_TO, pad_value = PAD_VALUE)

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
    ('conv2d', [32, 1, 3, 3, 1, 0]),
    ('relu', [True]),
    ('bn', [32]),
    ('max_pool2d', [2, 2, 0]),
    ('conv2d', [32, 32, 3, 3, 1, 0]),
    ('relu', [True]),
    ('bn', [32]),
    ('max_pool2d', [2, 2, 0]),
    ('conv2d', [32, 32, 3, 3, 1, 0]),
    ('relu', [True]),
    ('bn', [32]),
    ('max_pool2d', [2, 2, 0]),
    ('conv2d', [32, 32, 3, 3, 1, 0]),
    ('relu', [True]),
    ('bn', [32]),
    ('max_pool2d', [2, 1, 0]),
    ('flatten', []),
    ('linear', [args.n_way, 3200]) # Must fix if you change the default "PAD_TO = 200 and MEL_DIM = 80"
    ] 

# Loading meta
maml = Meta(args, config)
maml.net.to(device)


tmp = filter(lambda x: x.requires_grad, maml.parameters())
num = sum(map(lambda x: np.prod(x.shape), tmp))
print(maml)
print('Total trainable tensors:', num)

for step in range(args.epoch):

    print(f"Starting epoch {step}")

    x_spt, y_spt, x_qry, y_qry = nshot.next()
    x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device).type(torch.LongTensor), \
                                    torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device).type(torch.LongTensor)

    # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
    accs = maml(x_spt, y_spt, x_qry, y_qry)

    if step % 50 == 0:
        print('step:', step, '\ttraining acc:', accs)

    if step % 100 == 0:
        accs = []
        for _ in range(1000//args.task_num):
            # test
            x_spt, y_spt, x_qry, y_qry = nshot.next('test')
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device).type(torch.LongTensor), \
                                            torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device).type(torch.LongTensor)

            # split to single task each time
            for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                accs.append( test_acc )

        # [b, update_step+1]
        accs = np.array(accs).mean(axis=0).astype(np.float16)
        print('Test acc:', accs) # Each position is the average test acc for each update_step_test (starting in no update i = 0)

    # Saving epoch checkpoint  
    torch.save({
            'model_state_dict': maml.net.state_dict(),
            'loss': maml.loss,
            'best_loss': maml.best_loss,
            }, args.log_path + f'/checkpoint_epoch{step}_loss{maml.net.loss}.pth')