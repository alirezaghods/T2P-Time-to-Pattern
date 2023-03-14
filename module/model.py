import torch
from torch import nn, Tensor

# define encoder network
class Encoder(nn.Module):
    def __init__(self, timesteps, n_patterns):
        super(Encoder, self).__init__()
        
        self.timesteps = timesteps
        self.n_patterns = n_patterns
        
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(2,stride=2)
        self.softplus = nn.Softplus()

        # Encoder layers
        self.conv1 = nn.Conv1d(1,12,3)
        self.conv2 = nn.Conv1d(12,24,3)
        self.conv3 = nn.Conv1d(24,32,3)
        remaining_timesteps = int(((timesteps - 3 + 1 - 3 + 1)/2 - 3 + 1))
        self.encoded = nn.Conv3d(32, 1*2, (n_patterns,remaining_timesteps,1),padding=(int(n_patterns-1),0,0))
   
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        x = self.relu(self.conv3(x))
        x = self.softplus(self.encoded(x[:,:,None,:,None]))
        alpha_1 = x[:,:1]
        alpha_2 = x[:,1:]
        return alpha_1, alpha_2

# define T2P

class T2P(nn.Module):
    def __init__(self, timesteps: int, n_patterns: int, lambda_1: float, device:str=None):
        super(T2P, self).__init__()
        """
        T2P: It a neural network that can extract frequent pattern from time series
        args:
            - n_frames: number of input frames
            - timesteps: length of timeseris in each frames
            - lambda_1: temprature of the variational posterior q(z|x)
            - dtype: data type
            - device: the device to run the model (cpu or gpu)
        """

        self.encoder = Encoder(timesteps=timesteps, 
                                n_patterns=n_patterns)
                             

        self.decoder = nn.ConvTranspose3d(1, 1, (n_patterns,timesteps,1),padding=(int(n_patterns-1),0,0))

        self.lambda_1 = lambda_1 
        self.dtype = torch.float
        self.device = device
        self.tanhshrink = nn.Tanhshrink()
        self.softmax = nn.Softmax(dim=2)
    def repar(self, alpha1:Tensor, alpha2:Tensor)-> Tensor:
        """
        Reprametrization trick for concrete distribution
        args:
            - alpha 1
            - alpha 2
        """
        eps = 1e-7
        self.alpha = alpha1.div(alpha2+eps)
        U = torch.rand(self.alpha.size(), dtype=self.dtype, device=self.device, requires_grad=False)
        y = (self.alpha.mul(U).div(1-U+eps)).pow(1/self.lambda_1)
        z = y.div(1+y).mul(alpha1)
        return z

    def forward(self,inputs:Tensor) -> Tensor:
        a1, a2 = self.encoder(inputs)
        self.z = self.repar(a1,a2)
        self.z = self.softmax(self.z)
        x_ = self.decoder(self.z)
        return x_