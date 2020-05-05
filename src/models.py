import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, h_dim=512):
        super(Autoencoder, self).__init__()
        
        # (1, 128, 128)
        self.encode = nn.Sequential(
            nn.Conv2d(1, 64, 4, padding=1, stride=(2, 2)),
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
            
            nn.ConvTranspose2d(64, 1, 4, padding=1, stride=2),
            nn.Sigmoid()
        )
        
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