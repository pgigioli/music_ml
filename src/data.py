import os
import librosa
import numpy as np
import boto3
import json
import io
import sklearn
import scipy.io.wavfile as sciwav

from torch.utils.data import Dataset

class NSynthDataset(Dataset):
    def __init__(self, bucket, nsynth_path, instrument_source=(0, 1, 2), feature_type='mfcc', scaling=None):
        if scaling not in [None, 'standardize', 'normalize']:
            raise Exception('scaling must be one of: None, "standardize", or "normalize"')
        
        self.bucket = bucket
        self.nsynth_path = nsynth_path
        self.feature_type = feature_type 
        self.scaling = scaling
        
        self.s3_client = boto3.client('s3')
        self.s3_resource = boto3.resource('s3')
        
        meta_obj = self.s3_client.get_object(Bucket=self.bucket, Key=os.path.join(self.nsynth_path, 'examples.json'))
        self.meta = json.loads(meta_obj['Body'].read().decode('utf-8'))
        self.meta = dict([(k, v) for k, v in self.meta.items() if v['instrument_source'] in instrument_source])
        self.files = list(self.meta.keys())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav_fname = self.files[idx] + '.wav'
        features = self.extract_features(wav_fname)
        return features
    
    def extract_features(self, wav_fname):
        obj = self.s3_resource.Object(self.bucket, os.path.join(self.nsynth_path, 'audio/{}'.format(wav_fname)))
        sample_rate, X = sciwav.read(io.BytesIO(obj.get()['Body'].read()))
        X = X.astype(np.float32)


        if self.feature_type == 'mfcc':
            features = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
        elif self.feature_type == 'mel':
            features = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_fft=1024, hop_length=256)
        else:
            raise Exception('feat must be "mfcc" or "mel"')
            
        if self.scaling == 'standardize':
            features = sklearn.preprocessing.scale(features, axis=1)
        elif self.scaling == 'normalize':
            features = (features - features.min()) / (features.max() - features.min())

        return features