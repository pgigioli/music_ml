import os
import librosa
import numpy as np
import random
import boto3
import json
import io
import sklearn
import scipy.io.wavfile as sciwav

import torch
import torchvision
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset

def load_audio(fname, sample_rate=16000, t_start=0):
    audio, sr = torchaudio.load(fname)
    if sr != sample_rate:
        audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
    return audio[:, int(t_start*sample_rate):]

class Mono:
    def __init__(self, dim=0):
        self.dim = dim
        
    def __call__(self, x):
        return torch.mean(x, dim=self.dim, keepdim=True)
    
class TimeCrop:
    def __init__(self, length, random=False, time_dim=-1):
        self.length = length
        self.random = random
        self.time_dim = time_dim
        
    def __call__(self, x):
        if x.shape[self.time_dim] > self.length:
            if self.random:
                start = random.randint(0, x.shape[self.time_dim] - self.length - 1)
            else:
                start = 0
            return torch.narrow(x, self.time_dim, start, self.length)
        else:
            return x
    
class TimePad:
    def __init__(self, length, time_dim=-1, pad_val='zero'):
        if pad_val not in ['min', 'mean', 'zero']:
            raise Exception('mask_val must be one of [min, mean, zero]')
        
        self.length = length
        self.time_dim = time_dim
        self.pad_val = pad_val
        
    def __call__(self, x):
        if self.pad_val == 'min':
            val = x.min()
        elif self.pad_val == 'zero':
            val = 0.0
        elif self.pad_val == 'mean':
            val = x.mean()
        
        if x.shape[self.time_dim] < self.length:
            padding = [0, 0]*len(x.shape)
            padding[self.time_dim*2] = self.length - x.shape[self.time_dim]
            padding = padding[::-1]
            return F.pad(x, padding, mode='constant', value=val)
        else:
            return x
        
class Resize:
    def __init__(self, size):
        if type(size) == list:
            size = tuple(size)
        self.size = size
        
    def __call__(self, x):
        return F.interpolate(x.unsqueeze(0), self.size).squeeze(0)

class Normalize:
    def __call__(self, x):
        return (x - x.min()) / (x.max() - x.min())

class Standardize:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
        
    def __call__(self, x):
        if self.mean is None:
            self.mean = x.mean()
            
        if self.std is None:
            self.std = x.std()
        
        return (x - self.mean) / self.std

class RandomMask:
    def __init__(self, max_width, dim, mask_val='min'):
        if mask_val not in ['min', 'mean', 'zero']:
            raise Exception('mask_val must be one of [min, mean, zero]')
        
        self.max_width = max_width
        self.dim = dim
        self.mask_val = mask_val
        
    def __call__(self, x):
        width = random.randrange(1, self.max_width)
        start = random.randrange(0, x.shape[self.dim]-width)
        end = start + width
        mask_range = torch.arange(start, end, dtype=torch.long)
        
        if self.mask_val == 'min':
            val = x.min()
        elif self.mask_val == 'zero':
            val = 0.0
        elif self.mask_val == 'mean':
            val = x.mean()
            
        return x.index_fill_(self.dim, mask_range, val)
    
def create_audio_transform(sample_rate, n_samples, random_crop=False, feature_type='mel', resize=None, normalize=False, 
                           standardize=False, standardize_mean=None, standardize_std=None, spec_augment=False):
    transform = [
        Mono(),
        TimeCrop(n_samples, random=random_crop), 
        TimePad(n_samples)
    ]
    
    if feature_type == 'mel':
        transform.append(torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=256
        ))
    elif feature_type == 'mfcc':
        transform.append(torchaudio.transforms.MFCC(
            sample_rate=sample_rate, n_mfcc=40
        ))
    else:
        raise Exception('feature_type invalid')
        
    transform.append(torchaudio.transforms.AmplitudeToDB())
    
    if resize:
        transform.append(Resize(resize))
    
    if normalize:
        transform.append(Normalize())
        
    if standardize:
        transform.append(Standardize(mean=standardize_mean, std=standardize_std))
        
#     if image_time_pad:
#         transform.append(TimePad(image_time_pad, pad_val='min'))
        
    if spec_augment:
        transform.append(RandomMask(10, -2))
        transform.append(RandomMask(10, -1))
        
    return torchvision.transforms.Compose(transform)

class NSynthDataset(Dataset):
    def __init__(self, nsynth_path, s3_bucket=None, include_meta=False, instrument_source=(0, 1, 2), sample_rate=16000, n_samples=64000, 
                 feature_type='mel', random_crop=False, resize=None, normalize=False, standardize=False, standardize_mean=None, standardize_std=None, 
                 spec_augment=False, remove_synth_lead=False, n_samples_per_class=None):
        self.nsynth_path = nsynth_path
        self.s3_bucket = s3_bucket
        self.include_meta = include_meta
        self.instrument_source = instrument_source
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.feature_type = feature_type 
        self.random_crop = random_crop
        self.resize = resize
        self.normalize = normalize
        self.standardize = standardize
        self.standardize_mean = standardize_mean
        self.standardize_std = standardize_std
        self.spec_augment = spec_augment
        self.remove_synth_lead = remove_synth_lead
        self.n_samples_per_class = n_samples_per_class
        
        if self.resize and type(self.resize) != tuple:
            raise Exception('resize must be tuple')
        
        if self.s3_bucket:
            self.s3_client = boto3.client('s3')
            self.s3_resource = boto3.resource('s3')
        
            meta_obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=os.path.join(self.nsynth_path, 'examples.json'))
            meta = json.loads(meta_obj['Body'].read().decode('utf-8'))
        else:
            with open(os.path.join(self.nsynth_path, 'examples.json'), 'r') as read_file:
                meta = json.load(read_file)
        
        self.meta = {}
        self.class_cts = {}
        for k, v in meta.items():
            if v['instrument_source'] not in self.instrument_source:
                continue
            if self.remove_synth_lead and (v['instrument_family_str'] == 'synth_lead'):
                continue
            
            label = v['instrument_family_str']
            self.class_cts[label] = self.class_cts.get(label, 0) + 1
            if self.n_samples_per_class and (self.class_cts[label] > self.n_samples_per_class):
                continue
                
            self.meta[k] = v
            
        self.files = list(self.meta.keys())
        
        self.transform = create_audio_transform(
            self.sample_rate,
            self.n_samples,
            random_crop=self.random_crop,
            feature_type=self.feature_type,
            resize=self.resize,
            normalize=self.normalize,
            standardize=self.standardize,
            standardize_mean=self.standardize_mean,
            standardize_std=self.standardize_std,
            spec_augment=self.spec_augment
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav_fname = self.files[idx] + '.wav'
        features = self.extract_features(wav_fname)
        
        if self.include_meta:
            return features, self.meta[self.files[idx]]
        else:
            return features
    
    def extract_features(self, wav_fname):
        if self.s3_bucket:
            obj = self.s3_resource.Object(self.s3_bucket, os.path.join(self.nsynth_path, 'audio/{}'.format(wav_fname)))
            sample_rate, audio = sciwav.read(io.BytesIO(obj.get()['Body'].read()))
            audio = torch.tensor(audio, dtype=torch.float32)
            
            if sample_rate != self.sample_rate:
                audio = torchaudio.transforms.Resample(sample_rate, self.sample_rate)(audio)
        else:
            audio = load_audio(os.path.join(self.nsynth_path, 'audio/{}'.format(wav_fname)), sample_rate=self.sample_rate)
    
        features = self.transform(audio)
        return features