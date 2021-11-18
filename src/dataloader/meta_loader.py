'''
Dataloader for SER in meta-learning approach
'''

import numpy as np
import pandas as pd
import torch.utils.data as data
import torch
import random
from collections import defaultdict

# First we define a global dataloader which simply loads
# the pairs of wav files and emotion labels
class Dataloader4SER(data.Dataset):
    '''
        Given a csv file with wav_path and label columns,
        it returns pairs of raw wav_file path and labels.
    
        TODO: Process directly a disered audio transformation
        (instead of pass the raw wav path pass the melspectrogram or the mfccs)
    '''
    def __init__(self, df_path, ap, pad_to=100):
        self.df_path = df_path # Must have wav_path and labels columns
        self.ap = ap
        self.pad_to = pad_to

        self.all_data = pd.read_csv(self.df_path) # Must be comma delimited

        cols = self.all_data.columns
        if('wav_path' not in cols):
            raise RuntimeError('There is no > wav_path < column.')

        if('emotion' not in cols):
            raise RuntimeError('There is no > emotion < column.')

        # if('language' not in cols):
        #     raise RuntimeError('There is no > language < column.')

        # If dataset is complete then get the lists 
        self.x = self.all_data['wav_path'].to_list()
        self.y = self.all_data['emotion'].to_list()
        # self.l = self.all_data['language'].to_list()

    def load_wav(self, filename):
        audio = self.ap.load_wav(filename)
        return audio

    def load_data(self, index):
        wav = self.x[index]
        emotion = self.y[index]

        # w = np.asarray(self.load_wav(wav), dtype=np.float32)
        w = self.load_wav(wav)

        mel = self.ap.melspectrogram(w).astype('float32')

        mel_lengths = mel.shape[1]


        # mel = prepare_tensor(mel, 1)
        mel = mel.transpose(1, 0)


        # log mel  
        mel = np.log(mel)

        # pad mel
        if(mel_lengths >= self.pad_to):
            mel = mel[:self.pad_to, :]
        else:
            N = self.pad_to - mel_lengths
            zeros = np.zeros((N,80))
            mel = np.concatenate((mel, zeros), axis = 0)

        # print(mel.shape)


        mel = torch.FloatTensor(mel).contiguous()
        # mel_lengths = torch.LongTensor(mel_lengths)


        # return {'wav': w, 'emotion': np.asarray(emotion)}
        return {'mel': mel, 'emotion': emotion}

    def collate_fn(self, batch):

        # w = batch[:,0]
        # emotion = batch[:,1]
        # print(w, emotion)
        print(batch)

        mel = [self.ap.melspectrogram(w).astype("float32") for w in batch['wav']]
        # mel = self.ap.melspectrogram(w).astype('float32')
        mel_lengths = mel.shape[1]

        mel = prepare_tensor(mel, self.outputs_per_step)
        mel = mel.transpose(0, 2, 1)

        mel = torch.FloatTensor(mel).contiguous()
        mel_lengths = torch.LongTensor(mel_lengths)

        return {'melspec': mel, 'mel_len': mel_lengths, 'emotion': torch.tensor(batch['emotion'])}

    def __getitem__(self, index):
        return self.load_data(index)

    def __len__(self):
        return len(self.x)


class NShotMAMLSampler(object):
    '''
        NShotMAMLSampler returns batches of meta samples
        in a N-samples way of each class 
    '''

    def __init__(self, targets, n_way, k_shot, lang, n_episodes_per_iter, shuffle, shuffle_once, include_query):
        '''
          n_way: number of examples for each of the k classes
        '''
        self.targets = targets
        self.n_way = n_way
        self.k_shot = k_shot
        self.language = lang
        self.batch_size = n_way*k_shot
        self.episodes = n_episodes_per_iter
        self.shuffle = shuffle
        self.shuffle_once = shuffle_once
        self.include_query = include_query

        # Organize samples per target value
        self.classes = torch.unique(self.targets).tolist()
        self.num_classes = len(self.classes)

        self.idx_per_class = {}
        self.batches_per_class = {}

        for c in self.classes:
          self.idx_per_class[c] = torch.where(self.targets == c)[0]
          self.batches_per_class[c] = self.idx_per_class[c].shape[0]//self.k_shot


        # Create a list of classes from which we select the N classes per batch
        self.iterations = sum(self.batches_per_class.values()) // self.n_way
        self.class_list = [c for c in self.classes for _ in range(self.batches_per_class[c])]
        if shuffle_once or self.shuffle:
            self.shuffle_data()
        else:
            # For testing, we iterate over classes instead of shuffling them
            sort_idxs = [i+p*self.num_classes for i,
                         c in enumerate(self.classes) for p in range(self.batches_per_class[c])]
            self.class_list = np.array(self.class_list)[np.argsort(sort_idxs)].tolist()

    def shuffle_data(self):
        # Shuffle the examples per class
        for c in self.classes:
            perm = torch.randperm(self.idx_per_class[c].shape[0])
            self.idx_per_class[c] = self.idx_per_class[c][perm]
        # Shuffle the class list from which we sample. Note that this way of shuffling
        # does not prevent to choose the same class twice in a batch. However, for
        # training and validation, this is not a problem.
        random.shuffle(self.class_list)

    def __iter__(self):
        # Shuffle data
        if self.shuffle:
            self.shuffle_data()

        # Sample few-shot batches
        start_index = defaultdict(int)
        for it in range(self.iterations):
            class_batch = self.class_list[it*self.n_way:(it+1)*self.n_way]  # Select N classes for the batch
            index_batch = []
            for c in class_batch:  # For each class, select the next K examples and add them to the batch
                index_batch.extend(self.idx_per_class[c][start_index[c]:start_index[c]+self.k_shot])
                start_index[c] += self.k_shot
            if self.include_query:  # If we return support+query set, sort them so that they are easy to split
                index_batch = index_batch[::2] + index_batch[1::2]
            yield index_batch

    def __len__(self):
        return self.iterations
      

        