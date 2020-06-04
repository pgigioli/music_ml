import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, n_classes, h_dim=1024):
        super(Classifier, self).__init__()
        
        # (1, 128, 128)
        self.encode = nn.Sequential(
            nn.Conv2d(1, 128, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 128, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 256, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 512, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            
            nn.Conv2d(512, 512, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            
            nn.Conv2d(512, 1024, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            
            nn.Conv2d(1024, h_dim, 1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(h_dim)
        )

        self.out = nn.Linear(h_dim, n_classes, bias=True)
        
    def forward(self, x):
        h = self.encode(x)
        h = h.view(h.size(0), -1)
        return self.out(h)
    
class WavClassifier(nn.Module):
    def __init__(self, n_classes, h_dim=1024):
        super(WavClassifier, self).__init__()
        
        # (1, 65536)
        self.encode = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=9, stride=4, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Conv1d(128, 128, kernel_size=9, stride=4, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Conv1d(128, 128, kernel_size=9, stride=4, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Conv1d(128, 256, kernel_size=9, stride=4, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Conv1d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            nn.Conv1d(1024, 1024, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            nn.Conv1d(1024, h_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(h_dim),
        )

        self.out = nn.Linear(h_dim, n_classes, bias=True)
        
    def forward(self, x):
        h = self.encode(x)
        h = h.view(h.size(0), -1)
        return self.out(h)

class Autoencoder(nn.Module):
    def __init__(self, h_dim=1024):
        super(Autoencoder, self).__init__()
        
        # (1, 128, 128)
        self.encode = nn.Sequential(
            nn.Conv2d(1, 128, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 128, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 256, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 512, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            
            nn.Conv2d(512, 512, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            
            nn.Conv2d(512, 1024, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            
            nn.Conv2d(1024, h_dim, 1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(h_dim)
        )

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(h_dim, 1024, 1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            
            nn.ConvTranspose2d(1024, 512, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            
            nn.ConvTranspose2d(512, 512, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            
            nn.ConvTranspose2d(512, 256, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.ConvTranspose2d(256, 256, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.ConvTranspose2d(256, 128, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 128, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 1, 4, padding=1, stride=2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        h = self.encode(x)
        return self.decode(h)
    
class WavAutoencoder(nn.Module):
    def __init__(self, h_dim=1024):
        super(WavAutoencoder, self).__init__()
        
        # (1, 65536)
        self.encode = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=9, stride=4, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Conv1d(128, 128, kernel_size=9, stride=4, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Conv1d(128, 128, kernel_size=9, stride=4, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Conv1d(128, 256, kernel_size=9, stride=4, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Conv1d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            nn.Conv1d(1024, 1024, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            nn.Conv1d(1024, h_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(h_dim),
        )

        self.decode = nn.Sequential(
            nn.ConvTranspose1d(h_dim, 1024, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            nn.ConvTranspose1d(1024, 1024, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            nn.ConvTranspose1d(1024, 512, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.ConvTranspose1d(512, 512, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.ConvTranspose1d(512, 512, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.ConvTranspose1d(512, 256, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.ConvTranspose1d(256, 256, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.ConvTranspose1d(256, 256, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.ConvTranspose1d(256, 128, 8, padding=2, stride=4),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.ConvTranspose1d(128, 128, 8, padding=2, stride=4),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.ConvTranspose1d(128, 128, 8, padding=2, stride=4),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.ConvTranspose1d(128, 1, 8, padding=2, stride=4),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        h = self.encode(x)
        return self.decode(h)
    
class AutoencoderClassifier(nn.Module):
    def __init__(self, n_classes, h_dim=128):
        super(AutoencoderClassifier, self).__init__()
        
        self.n_classes = n_classes
        self.h_dim = h_dim
        
        # (1, 128, 128)
        self.encode = nn.Sequential(
            nn.Conv2d(1, 64, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 64, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 128, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 256, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 512, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            
            nn.Conv2d(512, h_dim, 1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(h_dim)
        )
        
        self.out = nn.Linear(h_dim, n_classes, bias=True)

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(h_dim, 512, 1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            
            nn.ConvTranspose2d(512, 256, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.ConvTranspose2d(256, 256, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.ConvTranspose2d(256, 128, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 128, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 64, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 1, 4, padding=1, stride=2),
            nn.Sigmoid()
        )
        
    def classify(self, x):
        h = self.encode(x)
        return self.out(h.view(h.size(0), -1))
        
    def forward(self, x):
        h = self.encode(x)
        recon = self.decode(h)
        logits = self.out(h.view(h.size(0), -1))
        return logits, recon
    
class AutoencoderLite(nn.Module):
    def __init__(self, h_dim=512):
        super(AutoencoderLite, self).__init__()
        
        # (1, 128, 128)
        self.encode = nn.Sequential(
            nn.Conv2d(1, 64, 8, padding=2, stride=4),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, 8, padding=2, stride=4),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 512, 4, padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            
            nn.Conv2d(512, h_dim, 1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(h_dim)
        )

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(h_dim, 512, 1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            
            nn.ConvTranspose2d(512, 256, 4, padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.ConvTranspose2d(256, 128, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, 8, padding=2, stride=4),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 1, 8, padding=2, stride=4),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        h = self.encode(x)
        return self.decode(h)
    
class SpatialTimeAutoencoder(nn.Module):
    def __init__(self, h_dim=512):
        super(SpatialTimeAutoencoder, self).__init__()
        
        # (1, 128, 128)
        self.time_encode = nn.Sequential(
            nn.Conv2d(1, 1, (128, 3), padding=(0, 1), stride=1),
            nn.ReLU()
        )
        
        self.spatial_encode = nn.Sequential(
            nn.Conv2d(1, 64, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 64, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 128, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 256, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 512, 4, padding=1, stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )

        self.dense_concat = nn.Conv2d(128+512, h_dim, 1, stride=1)
        
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(h_dim, 512, 1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            
            nn.ConvTranspose2d(512, 256, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.ConvTranspose2d(256, 256, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.ConvTranspose2d(256, 128, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 128, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 64, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 1, 4, padding=1, stride=2),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        t_h = self.time_encode(x)
        s_h = self.spatial_encode(x)
        h = torch.cat([s_h, t_h.permute(0, 3, 1, 2)], 1)
        return F.relu(self.dense_concat(h))
        
    def forward(self, x):
        h = self.encode(x)
        return self.decode(h)

# class Autoencoder(nn.Module):
#     def __init__(self, h_dim=1024):
#         super(Autoencoder, self).__init__()

#         # (1, 128, 251)
#         self.conv1 = nn.Conv2d(1, 32, 4, padding=(1, 4), stride=(2, 2))
#         self.conv2 = nn.Conv2d(32, 64, 4, padding=1, stride=(2, 2))
#         self.conv3 = nn.Conv2d(64, 128, 4, padding=1, stride=(2, 2))
#         self.conv4 = nn.Conv2d(128, 256, 4, padding=1, stride=(2, 2))
#         self.conv5 = nn.Conv2d(256, 512, 4, padding=1, stride=(2, 2))

#         self.fc_enc = nn.Linear(512*4*8, h_dim, bias=True)
#         self.fc_dec = nn.Linear(h_dim, 512*4*8, bias=True)

#         self.deconv1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
#         self.deconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
#         self.deconv3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
#         self.deconv4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
#         self.deconv5 = nn.ConvTranspose2d(32, 1, (2, 3), padding=(0, 3), stride=2)
        
#         # (1, 40, 126)
# #         self.conv1 = nn.Conv2d(1, 32, 4, padding=1, stride=(2, 2))
# #         self.conv2 = nn.Conv2d(32, 64, 4, padding=1, stride=(2, 2))
# #         self.conv3 = nn.Conv2d(64, 128, 4, padding=1, stride=(2, 2))
# #         self.conv4 = nn.Conv2d(128, 256, 4, padding=1, stride=(2, 2))
        
# #         self.fc_enc = nn.Linear(3584, h_dim, bias=True)
# #         self.fc_dec = nn.Linear(h_dim, 3584, bias=True)

# #         self.deconv1 = nn.ConvTranspose2d(256, 128, 3, stride=2)
# #         self.deconv2 = nn.ConvTranspose2d(128, 64, (2, 3), stride=2)
# #         self.deconv3 = nn.ConvTranspose2d(64, 32, (2, 3), stride=2)
# #         self.deconv4 = nn.ConvTranspose2d(32, 1, 2, stride=2)
        
#     def encode(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = F.relu(self.conv5(x))
    
#         x = x.view(x.size(0), -1)
#         return F.relu(self.fc_enc(x))
        
#     def decode(self, h):
#         h = F.relu(self.fc_dec(z))
#         h = h.view(h.size(0), 512, 4, 8)
        
#         h = F.relu(self.deconv1(h))
#         h = F.relu(self.deconv2(h))
#         h = F.relu(self.deconv3(h))
#         h = F.relu(self.deconv4(h))
#         h = self.deconv5(h)
#         return torch.sigmoid(h)
        
#     def forward(self, x):
#         h = self.encode(x)
#         outputs = self.decode(h)
#         return outputs