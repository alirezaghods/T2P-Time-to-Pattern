from torch import nn

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