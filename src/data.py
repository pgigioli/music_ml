import os
import librosa
import numpy as np
import random
import boto3
import json
import io
import sklearn
import pandas as pd
import scipy.io.wavfile as sciwav
from sklearn.model_selection import train_test_split

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
    
def create_audio_transform(sample_rate, n_samples=None, random_crop=False, feature_type='mel', resize=None, normalize=False, 
                           standardize=False, standardize_mean=None, standardize_std=None, spec_augment=False):
    if (type(spec_augment) == str) and (spec_augment not in ['freq', 'time']):
        raise Exception('spec_augment must be bool or one of: freq, time')
    
    transform = [
        Mono()
    ]
    
    if n_samples:
        transform.append(TimeCrop(n_samples, random=random_crop))
        transform.append(TimePad(n_samples))
    
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
        
    if spec_augment == True:
        transform.append(RandomMask(10, -2))
        transform.append(RandomMask(10, -1))
    elif spec_augment == 'freq':
        transform.append(RandomMask(10, -2))
    elif spec_augment == 'time':
        transform.append(RandomMask(10, -1))
        
    return torchvision.transforms.Compose(transform)

class NSynth(Dataset):
    def __init__(self, nsynth_path, split, s3_bucket=None, include_meta=False, instrument_source=(0, 1, 2), sample_rate=16000, n_samples=64000, 
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
            
        if split not in ['train', 'val', 'test']:
            raise Exception('split must be one of: train, val, test')
            
        if split == 'val': split = 'valid'
            
        self.data_dir = os.path.join(self.nsynth_path, 'nsynth-{}'.format(split))
        
        if self.s3_bucket:
            self.s3_client = boto3.client('s3')
            self.s3_resource = boto3.resource('s3')
        
            meta_obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=os.path.join(self.data_dir, 'examples.json'))
            meta = json.loads(meta_obj['Body'].read().decode('utf-8'))
        else:
            with open(os.path.join(self.data_dir, 'examples.json'), 'r') as read_file:
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
        self.files = [os.path.join(self.data_dir, 'audio/{}.wav'.format(x)) for x in self.files]
        
        self.transform = create_audio_transform(
            self.sample_rate,
            n_samples=self.n_samples,
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
        wav_fname = self.files[idx]
        features = self.extract_features(wav_fname)
        
        if self.include_meta:
            return features, self.meta[self.files[idx]]
        else:
            return features
    
    def extract_features(self, wav_fname):
        if self.s3_bucket:
            obj = self.s3_resource.Object(self.s3_bucket, wav_fname)
            sample_rate, audio = sciwav.read(io.BytesIO(obj.get()['Body'].read()))
            audio = torch.tensor(audio, dtype=torch.float32)
            
            if sample_rate != self.sample_rate:
                audio = torchaudio.transforms.Resample(sample_rate, self.sample_rate)(audio)
        else:
            audio = load_audio(wav_fname, sample_rate=self.sample_rate)
    
        features = self.transform(audio)
        return features
    
class Freesound(Dataset):
    def __init__(self, freesound_dir, split, binary=False, sample_rate=16000, n_samples=64000, random_crop=False, feature_type='mel', 
                 resize=None, normalize=False, standardize=False, standardize_mean=None, standardize_std=None, spec_augment=False, 
                 val_split=0.1, random_state=0, no_label=False):
        
        self.freesound_dir = freesound_dir
        self.split = split
        self.binary = binary
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.random_crop = random_crop
        self.feature_type = feature_type
        self.resize = resize
        self.normalize = normalize
        self.standardize = standardize
        self.standardize_mean = standardize_mean
        self.standardize_std = standardize_std
        self.spec_augment = spec_augment
        self.val_split = val_split
        self.no_label = no_label

        if resize and type(resize) != tuple:
            raise Exception('resize must be tuple')
        
        if (split == 'train') or (split == 'val'):
            self.data_dir = os.path.join(freesound_dir, 'FSDKaggle2018.audio_train')
            self.meta = pd.read_csv(os.path.join(freesound_dir, 'FSDKaggle2018.meta/train_post_competition.csv'))
            
            if split == 'train':
                self.meta, _ = train_test_split(self.meta, test_size=val_split, random_state=random_state)
            else:
                _, self.meta = train_test_split(self.meta, test_size=val_split, random_state=random_state)
                
            self.meta = self.meta.reset_index(drop=True)
        elif split == 'test':
            self.data_dir = os.path.join(freesound_dir, 'FSDKaggle2018.audio_test')
            self.meta = pd.read_csv(os.path.join(freesound_dir, 'FSDKaggle2018.meta/test_post_competition_scoring_clips.csv'))
        else:
            raise Exception('split must be train, val, or test')
            
        self.meta['fname'] = self.meta['fname'].apply(lambda x : os.path.join(self.data_dir, x))
            
        if self.binary:
            instruments = [
                'Acoustic_guitar', 'Violin_or_fiddle', 'Trumpet', 'Cello', 'Double_bass', 'Saxophone', 'Clarinet',
                'Bass_drum', 'Flute', 'Hi-hat', 'Snare_drum', 'Oboe', 'Gong', 'Tambourine', 'Cowbell', 'Harmonica',
                'Electric_piano', 'Chime', 'Glockenspiel'
            ]
            
            self.meta['label'] = self.meta['label'].apply(lambda x : 'instrument' if x in instruments else 'non_instrument')
            
        self.transform = create_audio_transform(
            self.sample_rate,
            n_samples=self.n_samples,
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
        return len(self.meta)

    def __getitem__(self, idx):
        wav_fname = self.meta.iloc[idx]['fname']
        features = self.extract_features(wav_fname)
        
        if self.no_label:
            return features
        else:
            return features, self.meta.iloc[idx]['label']
        
    def extract_features(self, wav_fname):
        audio = load_audio(wav_fname, sample_rate=self.sample_rate)
    
        features = self.transform(audio)
        return features
    
class TUTAcousticScenes(Dataset):
    def __init__(self, data_dir, split, sample_rate=16000, n_samples=160000, random_crop=False, feature_type='mel', resize=None,
                 normalize=False, standardize=False, standardize_mean=None, standardize_std=None, spec_augment=False, no_label=False):
        self.split = split
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.random_crop = random_crop
        self.feature_type = feature_type
        self.resize = resize
        self.normalize = normalize
        self.standardize = standardize
        self.standardize_mean = standardize_mean
        self.standardize_std = standardize_std
        self.spec_augment = spec_augment
        self.no_label = no_label
        
        if resize and type(resize) != tuple:
            raise Exception('resize must be tuple')
        
        if split == 'train':
            self.data_dir = os.path.join(data_dir, 'TUT-urban-acoustic-scenes-2018-development')
            self.meta = pd.read_csv(os.path.join(self.data_dir, 'meta.csv'), delimiter='\t')
            
            # remove broken file
            idx = self.meta[self.meta['filename'] == 'audio/airport-london-5-226-a.wav'].index
            self.meta = self.meta.drop(idx).reset_index(drop=True)
            
            self.meta['filename'] = self.meta['filename'].apply(lambda x : os.path.join(self.data_dir, x))
        elif split == 'val':
            self.data_dir = os.path.join(data_dir, 'TUT-urban-acoustic-scenes-2018-evaluation')
        elif split == 'test':
            self.data_dir = os.path.join(data_dir, 'TUT-urban-acoustic-scenes-2018-leaderboard')
        else:
            raise Exception('split must be train, val, or test') 
            
        if split == 'val' or split == 'test':
            audio_dir = os.path.join(self.data_dir, 'audio')
            self.meta = pd.DataFrame([
                {'filename' : f, 'scene_label' : None, 'identifier' : None, 'source_label' : None} 
                for f in os.listdir(audio_dir)
                
            ])
            self.meta['filename'] = self.meta['filename'].apply(lambda x : os.path.join(audio_dir, x))
            
            
        self.transform = create_audio_transform(
            self.sample_rate,
            n_samples=self.n_samples,
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
        return len(self.meta)

    def __getitem__(self, idx):
        wav_fname = self.meta.iloc[idx]['filename']
        features = self.extract_features(wav_fname)
        
        if self.no_label:
            return features
        else:
            return features, self.meta.iloc[idx]['scene_label']
        
    def extract_features(self, wav_fname):
        audio = load_audio(wav_fname, sample_rate=self.sample_rate)
    
        features = self.transform(audio)
        return features
    
class UrbanSED(Dataset):
    def __init__(self, data_dir, split, sample_rate=16000, n_samples=160000, random_crop=False, feature_type='mel', resize=None,
                 normalize=False, standardize=False, standardize_mean=None, standardize_std=None, spec_augment=False, no_label=False):
        self.data_dir = data_dir
        self.split = split
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.random_crop = random_crop
        self.feature_type = feature_type
        self.resize = resize
        self.normalize = normalize
        self.standardize = standardize
        self.standardize_mean = standardize_mean
        self.standardize_std = standardize_std
        self.spec_augment = spec_augment
        self.no_label = no_label
        
        if resize and type(resize) != tuple:
            raise Exception('resize must be tuple')
            
        if self.split == 'val':
            key = 'validate'
        else:
            key= self.split
            
        self.audio_dir = os.path.join(self.data_dir, 'audio/{}'.format(key))
        self.annotations_dir = os.path.join(self.data_dir, 'annotations/{}'.format(key))
        self._create_meta()
            
        self.transform = create_audio_transform(
            self.sample_rate,
            n_samples=self.n_samples,
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
        return len(self.meta)

    def __getitem__(self, idx):
        wav_fname = self.meta.iloc[idx]['filename']
        features = self.extract_features(wav_fname)
        
        if self.no_label:
            return features
        else:
            return features, self.meta.iloc[idx]['events']
        
    def _create_meta(self):
        self.meta = []
        for f in os.listdir(self.audio_dir):
            if f[:2] == '._':
                continue

            audio_path = os.path.join(self.audio_dir, f)
            meta_path = f.strip('.wav') + '.jams'
            meta_path = os.path.join(self.annotations_dir, meta_path)

            annotation = jams.load(meta_path)
            annotation = annotation.annotations[0]

            events = []
            for event in annotation.data:
                if event.value['label'] != 'noise':
                    events.append({
                        'label' : event.value['label'],
                        'start' : event.value['event_time'],
                        'end' : event.value['event_time'] + event.duration
                    })

            self.meta.append({
                'filename' : audio_path,
                'events' : events
            })
        self.meta = pd.DataFrame(self.meta)
        
    def extract_features(self, wav_fname):
        audio = load_audio(wav_fname, sample_rate=self.sample_rate)
    
        features = self.transform(audio)
        return features
    
class AcousticSceneMusicSegmentation(Dataset):
    def __init__(
        self, tut_dir, nsynth_dir, split, freesound_dir=None, sample_rate=16000, n_samples=160000, random_crop=False, feature_type='mel',
        resize=None, normalize=False, standardize=False, standardize_mean=None, standardize_std=None, spec_augment=False, random_volume_reduction=False
    ):
        self.tut_dir = tut_dir
        self.freesound_dir = freesound_dir
        self.nsynth_dir = nsynth_dir
        self.split = split
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.random_crop = random_crop
        self.feature_type = feature_type
        self.resize = resize
        self.normalize = normalize
        self.standardize = standardize
        self.standardize_mean = standardize_mean
        self.standardize_std = standardize_std
        self.spec_augment = spec_augment
        self.random_volume_reduction = random_volume_reduction
        
        tut_dataset = TUTAcousticScenes(self.tut_dir, self.split)
        nsynth_dataset = NSynth(nsynth_dir, self.split)
        
        self.scene_files = tut_dataset.meta['filename'].tolist()
        self.music_files = nsynth_dataset.files
        
        if self.freesound_dir:
            freesound_dataset = Freesound(self.freesound_dir, self.split, binary=True)
            music_meta = freesound_dataset.meta[freesound_dataset.meta['label'] == 'instrument'].reset_index(drop=True)
            self.music_files += music_meta['fname'].tolist()
        
        self.transform = create_audio_transform(
            self.sample_rate,
            n_samples=self.n_samples,
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
        return len(self.scene_files)

    def __getitem__(self, idx):
        scene_fname = self.scene_files[idx]
        music_fname = random.choice(self.music_files)
        
        audio, start, end = self.combine_audio(scene_fname, music_fname, random_volume_reduction=self.random_volume_reduction)
        features = self.transform(audio)
        
        # create segmentation mask
        scale = (features.shape[-1] / audio.shape[-1])
        start = int(scale*start)
        end = int(scale*end)
        mask = torch.zeros(features.shape[-1], device=features.device, dtype=torch.float32)
        mask[start:end] = 1.0
        
        return features, mask
    
    def combine_audio(self, scene_fname, music_fname, random_volume_reduction=False):
        scene = load_audio(scene_fname, sample_rate=self.sample_rate)
        music = load_audio(music_fname, sample_rate=self.sample_rate)
        
        # convert to mono
        scene = scene.mean(0, keepdim=True)
        music = music.mean(0, keepdim=True)
        
        # trim leading/trailing zeros from music audio
        music = torch.from_numpy(np.trim_zeros(music.squeeze().numpy())).unsqueeze(0)
        
        # truncate music audio if longer than scene audio
        if music.shape[-1] > (scene.shape[-1] - self.sample_rate*2):
            music = music[..., :scene.shape[-1] - self.sample_rate*2]
            
        # normalize music volume to match scene volume
        if random_volume_reduction:
            scene_to_music_ratio = random.random()*3 + 1.0
        else:
            scene_to_music_ratio = 1.0

        new_max = scene.max() / scene_to_music_ratio
        new_min = scene.min() / scene_to_music_ratio
        music = (((music - music.min()) * (new_max - new_min)) / (music.max() - music.min())) + new_min
        
        # insert music into scene
        start = random.randint(0, scene.shape[-1] - music.shape[-1] - 1)
        end = start+music.shape[-1]
        scene[..., start:end] = np.clip(scene[..., start:end] + music, scene.min(), scene.max())
        
        return scene, start, end
    
class MusicVsNoise(Dataset):
    def __init__(self, nsynth_dir, tut_scenes_dir, split, sample_rate=16000, n_samples=64000, random_crop=False, feature_type='mel', resize=None,
                 normalize=False, standardize=False, standardize_mean=None, standardize_std=None, spec_augment=False):
        self.nsynth_dir = nsynth_dir
        self.tut_scenes_dir = tut_scenes_dir
        self.split = split
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.random_crop = random_crop
        self.feature_type = feature_type
        self.resize = resize
        self.normalize = normalize
        self.standardize = standardize
        self.standardize_mean = standardize_mean
        self.standardize_std = standardize_std
        self.spec_augment = spec_augment
        
        if resize and type(resize) != tuple:
            raise Exception('resize must be tuple')
            
        nsynth_dataset = NSynth(self.nsynth_dir, self.split)
        tut_dataset = TUTAcousticScenes(self.tut_scenes_dir, self.split)
        
        self.music_files = nsynth_dataset.files
        self.noise_files = tut_dataset.meta['filename'].tolist()
        
        self.meta = pd.DataFrame()
        self.meta['filename'] = self.music_files + self.noise_files
        self.meta['label'] = ['music']*len(self.music_files) + ['noise']*len(self.noise_files)
        self.meta = self.meta.sample(frac=1).reset_index(drop=True)
        
        self.transform = create_audio_transform(
            self.sample_rate,
            n_samples=self.n_samples,
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
        return len(self.meta)

    def __getitem__(self, idx):
        wav_fname = self.meta.iloc[idx]['filename']
        features = self.extract_features(wav_fname)
        
        return features, self.meta.iloc[idx]['label']
        
    def extract_features(self, wav_fname):
        audio = load_audio(wav_fname, sample_rate=self.sample_rate)
    
        features = self.transform(audio)
        return features