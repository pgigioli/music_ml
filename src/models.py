import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalAvg1D(nn.Module):
    def __init__(self, dim):
        super(GlobalAvg1D, self).__init__()
        
        self.dim = dim
    
    def __call__(self, tensor):
        if len(tensor.shape) != 4:
            raise Exception('tensor must be rank of 4')
            
        return tensor.mean(self.dim, keepdim=True)

class GlobalAvg2D(nn.Module):
    def __call__(self, tensor):
        if len(tensor.shape) != 4:
            raise Exception('tensor must be rank of 4')
            
        return tensor.mean([2, 3], keepdim=True)
    
def straight_through_estimator(logits):
    argmax = torch.eq(logits, logits.max(-1, keepdim=True).values).to(logits.dtype)
    return (argmax - logits).detach() + logits

def gumbel_softmax(logits, temperature=1.0, eps=1e-20):
    u = torch.rand(logits.size(), dtype=logits.dtype, device=logits.device)
    g = -torch.log(-torch.log(u + eps) + eps)
    return F.softmax((logits + g) / temperature, dim=-1)
    
class Classifier(nn.Module):
    def __init__(self, n_classes, h_dim=512, depths=(64, 64, 128, 128, 256, 256)):
        super(Classifier, self).__init__()
        
        self.n_classes = n_classes
        self.h_dim = h_dim
        self.depths = depths
        
        # (1, 128, 251)
        self.encode = nn.Sequential(
#             nn.Conv2d(1, 64, (3, 4), padding=(1, 4), stride=1),
            nn.Conv2d(1, depths[0], 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(depths[0]),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(depths[0], depths[1], 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(depths[1]),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(depths[1], depths[2], 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(depths[2]),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(depths[2], depths[3], 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(depths[3]),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(depths[3], depths[4], 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(depths[4]),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(depths[4], depths[5], 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(depths[5]),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(depths[5], h_dim, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(h_dim),
            GlobalAvg2D()
        )
        
        self.out = nn.Linear(h_dim, n_classes, bias=True)
        
    def forward(self, x):
        h = self.encode(x)
        h = h.view(h.size(0), -1)
        return self.out(h)

class Segmenter1d(nn.Module):
    def __init__(self, h_dim=512, sigmoid=True, depths=(64, 64, 128, 128, 256, 256)):
        super(Segmenter1d, self).__init__()
        
        self.h_dim = h_dim
        self.sigmoid = sigmoid
        self.depths = depths
        
        # (1, 128, 251)
        self.encode = nn.Sequential(
            nn.Conv2d(1, depths[0], (3, 9), padding=(1, 4), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(depths[0]),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            
            nn.Conv2d(depths[0], depths[1], (3, 9), padding=(1, 4), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(depths[1]),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            
            nn.Conv2d(depths[1], depths[2], (3, 9), padding=(1, 4), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(depths[2]),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            
            nn.Conv2d(depths[2], depths[3], (3, 9), padding=(1, 4), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(depths[3]),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            
            nn.Conv2d(depths[3], depths[4], (3, 9), padding=(1, 4), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(depths[4]),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            
            nn.Conv2d(depths[4], depths[5], (3, 9), padding=(1, 4), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(depths[5]),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            
            nn.Conv2d(depths[5], h_dim, (3, 9), padding=(1, 4), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(h_dim),
            GlobalAvg1D(-2)
        )
        
        self.out = nn.Conv2d(h_dim, 1, (1, 1))
        
    def forward(self, x):
        h = self.encode(x)
        outputs = self.out(h)
        outputs = outputs.view(outputs.shape[0], -1)
        
        if self.sigmoid:
            return torch.sigmoid(outputs)
        else:
            return outputs

class Autoencoder(nn.Module):
    def __init__(self, h_dim=512, sigmoid=False, depths=(64, 64, 128, 128, 256, 256, 512)):
        super(Autoencoder, self).__init__()
        
        self.h_dim = h_dim
        self.sigmoid = sigmoid
        self.depths = depths
        
        # (1, 128, 251)
        self.encode = nn.Sequential(
            nn.Conv2d(1, depths[0], (3, 4), padding=(1, 4), stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(depths[0]),
            
            nn.Conv2d(depths[0], depths[1], 3, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(depths[1]),
            
            nn.Conv2d(depths[1], depths[2], 3, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(depths[2]),
            
            nn.Conv2d(depths[2], depths[3], 3, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(depths[3]),
            
            nn.Conv2d(depths[3], depths[4], 3, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(depths[4]),
            
            nn.Conv2d(depths[4], depths[5], 3, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(depths[5]),
            
            nn.Conv2d(depths[5], depths[6], (2, 4), padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(depths[6]),
            
            nn.Conv2d(depths[6], h_dim, 1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(h_dim)
        )

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(h_dim, depths[6], 1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(depths[6]),
            
            nn.ConvTranspose2d(depths[6], depths[5], (4, 6), padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(depths[5]),
            
            nn.ConvTranspose2d(depths[5], depths[4], 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(depths[4]),
            
            nn.ConvTranspose2d(depths[4], depths[3], 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(depths[3]),
            
            nn.ConvTranspose2d(depths[3], depths[2], 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(depths[2]),
            
            nn.ConvTranspose2d(depths[2], depths[1], 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(depths[1]),
            
            nn.ConvTranspose2d(depths[1], depths[0], 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(depths[0]),
            
            nn.ConvTranspose2d(depths[0], 1, (4, 5), padding=(1, 4), stride=2)
        )
        
    def forward(self, x):
        h = self.encode(x)
        
        if self.sigmoid:
            return torch.sigmoid(self.decode(h))
        else:
            return self.decode(h)

class VAE(Autoencoder):
    def __init__(self, h_dim=512, sigmoid=False, depths=(64, 64, 128, 128, 256, 256, 512)):
        super(VAE, self).__init__(h_dim=h_dim, sigmoid=sigmoid, depths=depths)
        
        self.fc_mu = nn.Linear(h_dim, h_dim)
        self.fc_log_var = nn.Linear(h_dim, h_dim)
        
    def _reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def _latent_layer(self, h, sample=True, temperature=1.0):
        mu, log_var = self.fc_mu(h), self.fc_log_var(h)
        
        if sample:
            z = self._reparameterize(mu, log_var)
        else:
            z = mu
        return z, mu, log_var
        
    def forward(self, x, sample=True, temperature=1.0):
        h = self.encode(x)
        h = h.view(h.size(0), -1)
        
        z, mu, log_var = self._latent_layer(h, sample=sample, temperature=temperature)
        z = z[..., None, None]
        
        outputs = self.decode(z)
        
        if self.sigmoid:
            return torch.sigmoid(outputs), mu, log_var
        else:
            return outputs, mu, log_var

class CVAE(VAE):
    def __init__(self, n_classes, h_dim=512, sigmoid=False, depths=(64, 64, 128, 128, 256, 256, 512)):
        super(CVAE, self).__init__(h_dim=h_dim, sigmoid=sigmoid, depths=depths)
        
        self.n_classes = n_classes

        self.fc_cond = nn.Linear(h_dim, n_classes)
        self.fc_merge = nn.Linear(h_dim + n_classes, h_dim)
        
    def _latent_layer(self, h, sample=True, temperature=1.0):
        mu, log_var = self.fc_mu(h), self.fc_log_var(h)
        c_logits = self.fc_cond(h)
        
        if sample:
            z = self._reparameterize(mu, log_var)
            c_dist = gumbel_softmax(c_logits, temperature=temperature)
        else:
            z = mu
            c_dist = F.softmax(c_logits, dim=-1)            

        c = straight_through_estimator(c_dist)
        
        # merge
        y = F.relu(self.fc_merge(torch.cat([z, c], dim=1)))
        return z, y, c_logits, mu, log_var

    def forward(self, x, sample=True, temperature=1.0):
        h = self.encode(x)
        h = h.view(h.size(0), -1)
        
        z, y, c_logits, mu, log_var = self._latent_layer(h, sample=sample, temperature=temperature)
        y = y[..., None, None]
        
        outputs = self.decode(y)
        
        if self.sigmoid:
            return torch.sigmoid(outputs), c_logits, mu, log_var
        else:
            return outputs, c_logits, mu, log_var
        
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