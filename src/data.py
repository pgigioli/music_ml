import os
import librosa
import numpy as np
import boto3
import json
import io
import torch
import torchaudio
import sklearn
import scipy.io.wavfile as sciwav
from skimage.transform import resize

from torch.utils.data import Dataset

class NSynthDataset(Dataset):
    def __init__(self, nsynth_path, s3_bucket=None, instrument_source=(0, 1, 2), feature_type='mfcc', scaling=None, include_meta=False,
                 resize=None, mu_law_companding=False, remove_synth_lead=False, n_samples_per_class=None):
        if scaling not in [None, 'standardize', 'normalize']:
            raise Exception('scaling must be one of: None, "standardize", or "normalize"')
        
        self.nsynth_path = nsynth_path
        self.s3_bucket = s3_bucket
        self.feature_type = feature_type 
        self.scaling = scaling
        self.include_meta = include_meta
        self.resize = resize
        self.mu_law_companding = mu_law_companding
        
        if resize and type(resize) != tuple:
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
        class_cts = {}
        for k, v in meta.items():
            if v['instrument_source'] not in instrument_source:
                continue
            if remove_synth_lead and (v['instrument_family_str'] == 'synth_lead'):
                continue
            
            label = v['instrument_family_str']
            class_cts[label] = class_cts.get(label, 0) + 1
            if n_samples_per_class and (class_cts[label] > n_samples_per_class):
                continue
                
            self.meta[k] = v
            
        self.files = list(self.meta.keys())

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
            sample_rate, X = sciwav.read(io.BytesIO(obj.get()['Body'].read()))
        else:
            sample_rate, X = sciwav.read(os.path.join(self.nsynth_path, 'audio/{}'.format(wav_fname)))
        X = X.astype(np.float32)

        if self.mu_law_companding:
            X = torchaudio.transforms.MuLawEncoding()(torch.tensor(X, dtype=torch.float32)).numpy().astype(np.float32)

        if self.feature_type == 'mfcc':
            features = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
        elif self.feature_type == 'mel':
            features = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_fft=1024, hop_length=256)
        elif self.feature_type == 'raw':
            features = X
        else:
            raise Exception('feat must be "raw", "mfcc", "mel"')
            
        if self.resize:
            features = resize(features, self.resize)
            
        if self.scaling == 'standardize':
            features = sklearn.preprocessing.scale(features, axis=1)
        elif self.scaling == 'normalize':
            features = (features - features.min()) / (features.max() - features.min())

        return features