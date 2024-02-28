import torch
import os
import torch.nn as nn
import numpy as np

# data = torch.from_numpy(np.random.rand(1,1,598))
# data = data.to(torch.float32)
# class LSTMEncoder(nn.Module):
#     def __init__(self, input_dim=598, hidden_dim=128, num_layers=4, latent_dims=4):
#         super(LSTMEncoder, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.linear1 = nn.Linear(128, 8)
#         self.linear2 = nn.Linear(128, 8)
#
#         self.N = torch.distributions.Normal(0, 1)
#         self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
#         self.N.scale = self.N.scale.cuda()
#         self.kl = 0
#
#         self.ncnn = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=8, kernel_size=8, stride=1, padding=2),
#             nn.LeakyReLU(negative_slope=0.01),
#             nn.Conv1d(in_channels=8, out_channels=16, kernel_size=8, stride=1, padding=2),
#             nn.LeakyReLU(negative_slope=0.01),
#             nn.Dropout(p=0.6),
#             nn.MaxPool1d(2,2),
#
#             nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8, stride=2, padding=2),
#             nn.LeakyReLU(negative_slope=0.01),
#             nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, stride=2, padding=2),
#             nn.LeakyReLU(negative_slope=0.01),
#             nn.Dropout(p=0.5),
#             nn.MaxPool1d(2,2),
#
#             nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=2, padding=2),
#             nn.LeakyReLU(negative_slope=0.01),
#
#             nn.Flatten(),
#             # nn.Linear(128*32, 512), #7 or 31
#             # nn.LeakyReLU(negative_slope=0.01),
#             # nn.Linear(512, 128),
#             # nn.LeakyReLU(negative_slope=0.01)
#         )
#
#         self.wcnn = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=8, kernel_size=16, stride=1, padding=2),
#             nn.LeakyReLU(negative_slope=0.01),
#             nn.Conv1d(in_channels=8, out_channels=16, kernel_size=16, stride=1, padding=2),
#             nn.LeakyReLU(negative_slope=0.01),
#             nn.Dropout(p=0.6),
#             nn.MaxPool1d(2,2),

#             nn.Conv1d(in_channels=16, out_channels=32, kernel_size=16, stride=2, padding=2),
#             nn.LeakyReLU(negative_slope=0.01),
#             nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, stride=2, padding=2),
#             nn.LeakyReLU(negative_slope=0.01),
#             nn.Dropout(p=0.5),
#             nn.MaxPool1d(2,2),
#
#             nn.Conv1d(in_channels=64, out_channels=128, kernel_size=16, stride=2, padding=2),
#             nn.LeakyReLU(negative_slope=0.01),
#
#             nn.Flatten(),
#             # nn.Linear(128*24, 512), #7 or 31
#             # nn.LeakyReLU(negative_slope=0.01),
#             # nn.Linear(512, 128),
#             # nn.LeakyReLU(negative_slope=0.01)
#         )
#
#     def forward(self, x):
#         noutputs = self.ncnn(x)
#         woutputs = self.wcnn(x)
#         cnnout = torch.cat((noutputs, woutputs), dim=1)
#         return cnnout
#
# model = LSTMEncoder()
# print(model(data).size())

data = np.load(r"F:\35\check\1700622645562_0_0.npy")
print(data.shape)