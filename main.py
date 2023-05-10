import torch
import glob
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import torch
from torch import nn
# import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import miditoolkit
# import modules
import pickle
import utils
import time
from miditok import REMI
from miditok.utils import get_midi_programs
from miditoolkit import MidiFile
from pathlib import Path
from transformers import TransfoXLConfig, TransfoXLTokenizer, TransfoXLModel
import os
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # training opts
    parser.add_argument('--is_train', type=bool,
                        help='True for training, False for testing.', default=True)
    parser.add_argument('--is_continue', type=bool,
                        help='True for continue training, False for training from scratch.', default=False)
    parser.add_argument('--continue_pth', type=str,
                        help='Continue training checkpoint path.', default='')
    parser.add_argument('--dict_path', type=str,
                        help='Decide using chord or not.', default='./dictionary/dictionary_REMI-tempo-checkpoint.pkl')
    
    # testing opts
    parser.add_argument('--prompt', type=bool,
                        help='False for generating from scratch, True for continue generating.', default=False)
    parser.add_argument('--prompt_path', type=str,
                        help='if prompt is True, you have to specify the continue generating midi file path.', default='')
        # './data/evaluation/000.midi'
    parser.add_argument('--n_target_bar', type=int,
                        help='Controll the generate result.', default=16)
    parser.add_argument('--temperature', type=float,
                        help='Controll the generate result.', default=1.2)
    parser.add_argument('--topk', type=int,
                        help='Controll the generate result.', default=5)
    parser.add_argument('--output_path', type=str,
                        help='output path', default='./results/from_scratch.midi')
    parser.add_argument('--model_path', type=str,
                        help='model path', default='./checkpoints/epoch_200.pkl')
    args = parser.parse_args()
    return args

class NewsDataset(Dataset):
    def __init__(self, midi_l = [], dict_pth = './dictionary/dictionary_REMI-tempo-checkpoint.pkl'):
        self.midi_l = midi_l
        self.tokenizer = REMI()
        self.checkpoint_path = dict_pth
        self.x_len = 512
        self.dictionary_path = dict_pth
        self.event2word, self.word2event = pickle.load(open(self.dictionary_path, 'rb'))
        self.parser = self.prepare_data(self.midi_l)
    
    def __len__(self):
        return len(self.parser)
    
    def __getitem__(self, index):
        if self.train:
            return self.parser[index]
        else:
            return self.words
            
    def extract_events(self, input_path):
        note_items, tempo_items = utils.read_items(input_path)
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end
        if 'chord' in self.checkpoint_path:
            chord_items = utils.extract_chords(note_items)
            items = chord_items + tempo_items + note_items
        else:
            items = tempo_items + note_items
        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups)
        return events
        
    def prepare_data(self, midi_paths):
        # extract events
        all_events = []
        for path in midi_paths:
            events = self.extract_events(path)
            all_events.append(events)
        # event to word
        all_words = []
        for events in all_events:
            words = []
            for event in events:
                e = '{}_{}'.format(event.name, event.value)
                if e in self.event2word:
                    words.append(self.event2word[e])
                else:
                    # OOV
                    if event.name == 'Note Velocity':
                        # replace with max velocity based on our training data
                        words.append(self.event2word['Note Velocity_21'])
                    else:
                        # something is wrong
                        # you should handle it for your own purpose
                        print('something is wrong! {}'.format(e))
            all_words.append(words)
        # to training data
        self.group_size = 5
        segments = []
        for words in all_words:
            pairs = []
            for i in range(0, len(words)-self.x_len-1, self.x_len):
                x = words[i:i+self.x_len]
                y = words[i+1:i+self.x_len+1]
                pairs.append([x, y])
            pairs = np.array(pairs)
            # abandon the last
            for i in np.arange(0, len(pairs)-self.group_size, self.group_size*2):
                data = pairs[i:i+self.group_size]
                if len(data) == self.group_size:
                    segments.append(data)
        segments = np.array(segments)
#         print(pairs.shape)
#         print(type(segments))
        print(segments.shape)
        return segments

class Model(nn.Module):
    def __init__(self, checkpoint, is_training=False):
        super(Model, self).__init__()
        # load dictionary
        self.dictionary_path = checkpoint
        self.checkpoint_path = checkpoint
        self.event2word, self.word2event = pickle.load(open(self.dictionary_path, 'rb'))
        # model settings
        self.x_len = 512
        self.mem_len = 512
        self.n_layer = 12
        self.d_embed = 512
        self.d_model = 512
        self.dropout = 0.1
        self.n_head = 8
        self.d_head = self.d_model // self.n_head
        self.d_ff = 2048
        self.n_token = len(self.event2word)
        self.learning_rate = 0.0002
        # load model
#         super(TransfoXLModel, self.model).__init__()
        self.configuration = TransfoXLConfig(   attn_type = 0,
                                                adaptive = False,
                                                clamp_len = -1,
                                                cutoffs = [],
                                                d_embed = self.d_embed,
                                                d_head = self.d_head,
                                                d_inner = self.d_ff,
                                                d_model = self.d_model,
                                                div_val = -1,
                                                dropatt = self.dropout,
                                                dropout = self.dropout,
                #                                 eos_token_id = ,
                                                init = 'normal',
                #                                 init_range = ,
                                                init_std = 0.02,
                                                layer_norm_epsilon = 0.001,
                                                mem_len = self.mem_len,
                                                n_head = self.n_head,
                                                n_layer = self.n_layer,
                                                pre_lnorm = 'normal',
                                                proj_init_std = 0.01,
                                                same_length = False,
                #                                 sample_softmax = ,
                                                tie_projs = [],
                                                vocab_size = self.n_token,
                                                untie_r = False)
#         
        # Initializing a model (with random weights) from the configuration
        self.xl = TransfoXLModel(self.configuration)
        self.drop = nn.Dropout(p=self.dropout)
        self.linear = nn.Linear(self.d_embed, self.n_token)

    def forward(self, x):
        outputs = self.xl(input_ids = x)
        output = self.drop(outputs['last_hidden_state']) # dropout
        output_logit = self.linear(output)
        return output_logit

def temperature_sampling(logits, temperature, topk):
        # probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        logits = torch.Tensor(logits)
        probs = nn.Softmax(dim=0)(logits / temperature)
        probs = np.array(probs)
        if topk == 1:
            prediction = np.argmax(probs)
        else:
            sorted_index = np.argsort(probs)[::-1]
            candi_index = sorted_index[:topk]
            candi_probs = [probs[i] for i in candi_index]
            # normalize probs
            candi_probs /= sum(candi_probs)
            # choose by predicted probs
            prediction = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return prediction
    
def test(prompt_path = './data/evaluation/000.midi', prompt = True, n_target_bar = 16,
         temperature = 1.2, topk = 5, output_path = '', model_path = ''):
    
    # check path folder
    try:
        os.makedirs('./results', exist_ok=True)
        print("dir \'./results\' is created")
    except:
        pass

    with torch.no_grad():
        # load model
        checkpoint = torch.load(model_path)
        model = Model(checkpoint=opt.dict_path)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        test_data = NewsDataset(midi_l = [prompt_path], dict_pth = opt.dict_path)
        batch_size = 1
        
        # if prompt, load it. Or, random start
        if prompt:
            events = test_data.extract_events(prompt_path)
            words = [[test_data.event2word['{}_{}'.format(e.name, e.value)] for e in events]]
            words[0].append(test_data.event2word['Bar_None'])
        else:
            words = []
            for _ in range(batch_size):
                ws = [test_data.event2word['Bar_None']]
                if 'chord' in model.checkpoint_path:
                    tempo_classes = [v for k, v in test_data.event2word.items() if 'Tempo Class' in k]
                    tempo_values = [v for k, v in test_data.event2word.items() if 'Tempo Value' in k]
                    chords = [v for k, v in test_data.event2word.items() if 'Chord' in k]
                    ws.append(test_data.event2word['Position_1/16'])
                    ws.append(np.random.choice(chords))
                    ws.append(test_data.event2word['Position_1/16'])
                    ws.append(np.random.choice(tempo_classes))
                    ws.append(np.random.choice(tempo_values))
                else:
                    tempo_classes = [v for k, v in test_data.event2word.items() if 'Tempo Class' in k]
                    tempo_values = [v for k, v in test_data.event2word.items() if 'Tempo Value' in k]
                    ws.append(test_data.event2word['Position_1/16'])
                    ws.append(np.random.choice(tempo_classes))
                    ws.append(np.random.choice(tempo_values))
                words.append(ws)

        # generate
        original_length = len(words[0])
        initial_flag = 1
        current_generated_bar = 0
        print('Start generating')
        while current_generated_bar < n_target_bar:
            # input
            if initial_flag:
                temp_x = np.zeros((batch_size, original_length))
                for b in range(batch_size):
                    for z, t in enumerate(words[b]):
                        temp_x[b][z] = t
                initial_flag = 0
            else:
                temp_x_new = np.zeros((batch_size, 1))
                for b in range(batch_size):
                    temp_x_new[b][0] = words[b][-1]
                temp_x = np.array([np.append(temp_x[0], temp_x_new[0])])
            # model (prediction)
            temp_x = torch.Tensor(temp_x).long()
            # temp_x = (1_batch, 4_length)
            # print('temp_x shape =', temp_x.shape)
            output_logits = model(temp_x)
            # print('output_logits shape =', output_logits.shape)
            # output_logits = output_logits.permute(0,2,1)
            # sampling
            _logit = output_logits[0, -1].detach().numpy()
            # print('_logit shape =', _logit.shape)
            # break

            # print('_logit =',_logit.shape)
            word = temperature_sampling(
                logits=_logit, 
                temperature=temperature,
                topk=topk)

            words[0].append(word)

            # if bar event (only work for batch_size=1)
            if word == test_data.event2word['Bar_None']:
                current_generated_bar += 1
            # re-new mem
    #         batch_m = _new_mem
        # write
        if prompt:
            utils.write_midi(
                words=words[0][original_length:],
                word2event=test_data.word2event,
                output_path=output_path,
                prompt_path=prompt_path)
        else:
            utils.write_midi(
                words=words[0],
                word2event=test_data.word2event,
                output_path=output_path,
                prompt_path=None)
    
# train
def train(is_continue = False, checkpoints_path = ''):
    epochs = 200
    # create data list
    train_list = glob.glob('./data/train/*.midi')
    print('train list len =', len(train_list))
    # dataset
    train_dataset = NewsDataset(train_list, dict_pth = opt.dict_path)
    # dataloader
    BATCH_SIZE = 4
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    print('Dataloader is created')

    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    # create model
    if not is_continue:
        start_epoch = 1
        model = Model(checkpoint=opt.dict_path).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.0002)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400000, eta_min=0.004*0.0002)
    else:
        # wheather checkpoint_path is exist
        if os.path.isfile(checkpoints_path):
            checkpoint = torch.load(checkpoints_path)
        else:
            os._exit()
        start_epoch = checkpoint['epoch'] + 1

        model = Model(checkpoint=opt.dict_path).to(device)
        model.load_state_dict(checkpoint['model'])

        optimizer = torch.optim.Adam(model.parameters(), lr = 0.0002)
        optimizer.load_state_dict(checkpoint['optimizer'])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400000, eta_min=0.004*0.0002)
        scheduler.load_state_dict(checkpoint['scheduler'])


    

    print('Model is created \nStart training')
    
    model.train()
    losses = []
    try:
        os.makedirs('./checkpoints', exist_ok=True)
        print("dir is created")
    except:
        pass
    
    for epoch in range(start_epoch, epochs+1):
        single_epoch = []
        for i in tqdm(train_dataloader):
            for g in range(5):
                x = i[:, g, 0, :].to(device).long()
                # x =(batch, 512)
                y = i[:, g, 1, :].to(device).long()
                output_logit = model(x)
                # print(output_logit.shape, y.shape)
                # output_logit = (4, 512(length), 274(vocab))
                loss = nn.CrossEntropyLoss()(output_logit.permute(0,2,1), y)
                loss.backward()
                single_epoch.append(loss.to('cpu').mean().item())
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        single_epoch = np.array(single_epoch)
        losses.append(single_epoch.mean())
        print('>>> Epoch: {}, Loss: {:.5f}'.format(epoch,losses[-1]))
        # torch.save(model.state_dict(), './checkpoints/epoch_%03d.pkl'%epoch)
        torch.save({'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'loss': losses[-1],
                    }, './checkpoints/123epoch_%03d.pkl'%epoch)
    losses = np.array(losses)
    np.save('training_losses_his_from_77.npy', losses)

def main(opt):

    # train
    if opt.is_train:
        # train from scratch
        if not opt.is_continue:
            train()
        # continue training
        else:
            train(is_continue = opt.is_continue, checkpoints_path = opt.continue_pth)

    else:
        # generate from screatch
        if not opt.prompt:
            test(prompt = opt.prompt, n_target_bar = opt.n_target_bar, temperature = opt.temperature, topk = opt.topk, 
                output_path = opt.output_path, model_path = opt.model_path)
        
        # continue generate
        else:
            test(prompt_path = opt.prompt_path, n_target_bar = opt.n_target_bar, temperature = opt.temperature, topk = opt.topk,
                output_path = opt.output_path, model_path = opt.model_path)

if __name__ == '__main__':
    opt = parse_opt()
    # print(opt)
    main(opt)
