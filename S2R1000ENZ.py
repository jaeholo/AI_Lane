import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from dataset import VAEDataset1000ENZ1
from tqdm import tqdm
import sys
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt # plotting library
import os
# from torchsummary import summary


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
#
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
#         self.fclayers = nn.Sequential(
#             nn.Dropout(0.6),
#             nn.Linear(3584, 1024), #234
#             nn.LeakyReLU(negative_slope=0.01),
#             nn.Dropout(0.5),
#             nn.Linear(1024, 128),
#             nn.LeakyReLU(negative_slope=0.01),
#             nn.Dropout(0.4),
#         )
#
#     def forward(self, x):
#         noutputs = self.ncnn(x)
#         woutputs = self.wcnn(x)
#         cnnout = torch.cat((noutputs, woutputs), dim=1)
#         fcout = self.fclayers(cnnout)
#
#         mu = self.linear1(fcout)
#         sigma = torch.exp(self.linear2(fcout))
#         z = mu + sigma*self.N.sample(mu.shape)
#         self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
#         return z
#
#
# class Decoder(nn.Module):
#     def __init__(self, latent_dims):
#         super().__init__()
#
#         self.decoder_lin = nn.Sequential(
#             nn.Linear(latent_dims, 128),
#             nn.ReLU(True),
#             nn.Linear(128, 8 * 8 * 32),
#             nn.ReLU(True)
#         )
#
#         self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 8, 8))
#
#         self.decoder_conv = nn.Sequential(
#             nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm2d(8),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
#         )
#
#     def forward(self, x):
#         x = self.decoder_lin(x)
#         x = self.unflatten(x)
#         x = self.decoder_conv(x)
#         x = torch.sigmoid(x)
#         return x
#
#
# class VariationalAutoencoder(nn.Module):
#     def __init__(self, latent_dims):
#         super(VariationalAutoencoder, self).__init__()
#         self.encoder = LSTMEncoder(input_dim=4096, hidden_dim=256, num_layers=4, latent_dims=latent_dims)
#         self.decoder = Decoder(latent_dims)
#
#     def forward(self, x):
#         # x = x.to(device)
#         z = self.encoder(x)
#         return self.decoder(z)


class Encoder(nn.Module):
    def __init__(self, latent_dims=8):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(128, latent_dims)
        self.linear2 = nn.Linear(128, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

        self.ncnn = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=8, kernel_size=8, stride=1, padding=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=8, stride=1, padding=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.6),
            nn.MaxPool1d(2,2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8, stride=1, padding=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, stride=1, padding=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(2,2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=1, padding=2),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Flatten(),
        )

        self.wcnn = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=8, kernel_size=32, stride=1, padding=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=32, stride=1, padding=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.6),
            nn.MaxPool1d(2,2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=32, stride=1, padding=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=32, stride=1, padding=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(2,2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=32, stride=1, padding=2),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Flatten(),
        )

        self.fclayers = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(128*436, 1024), #1972
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.4),
        )

    def forward(self, x):
        noutputs = self.ncnn(x)
        woutputs = self.wcnn(x)
        cnnout = torch.cat((noutputs, woutputs), dim=1)
        fcout = self.fclayers(cnnout)

        mu = self.linear1(fcout)
        sigma = torch.exp(self.linear2(fcout))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.ReLU(True),
            nn.Linear(128, 8 * 8 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 8, 8))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)










#

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# # loss_func = nn.MSELoss()
d = 8
vae = VariationalAutoencoder(latent_dims=d)
#
#
# ###
# ### ABOVE IS THE MODEL
# ###
#
vae = torch.load(r"F:\convert\models\034\4\model\430_39.947670515106275_57.68844268459291_49.45934567966088.pth") #6hr
vae.to(device)
#
# # batch_size = 1 # just a random number
# # x = torch.randn(batch_size, 2, 4096).to(device) # 4x1024 (4s of 1000hz)
#
# # print(torch_out.shape)
# # print("Supported ONNX opset versions in PyTorch:", torch.onnx.export.available_opset_version())
#
# # Export the model
# # torch.onnx.export(vae, # model being run
# #                   x, # model input (or a tuple for multiple inputs)
# #                   "1000nz.onnx", # where to save the model (can be a file or file-like object)
# #                   export_params=True, # store the trained parameter weights inside the model file
# #                   opset_version=18, # the ONNX version to export the model to
# #                   do_constant_folding=True, # whether to execute constant folding for optimization
# #                   # verbose=True,
# #                   input_names=['input'], # the model's input names
# #                   output_names=['output'], # the model's output names
# #                   dynamic_axes={'input': {0: 'batch_size'}, # variable length axes
# #                   'output': {0: 'batch_size'}})
# # exit(1)
#
#
# # summary(vae, (2, 4096*10))
# # exit()
# # vae.eval()
#
# ###
# ### CHANGE THIS EVERY TIME YOU RUN
# ###
# # basep: directory to save generate models and test images
# basep = "models/035/2"
# # basep/models： saves every other model
# # basep/n_label 194
# # basep/n_model 194
# # basep/m_label 228
# # basep/m_model 228
#
#
#
#
# lr = 1e-6
#
# # optim = torch.optim.SGD(vae.parameters(), lr=lr, weight_decay=1e-3, momentum=0.9)
# optim = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-3, eps=1e-8)
#
#
# num_epochs = 500
#
# # milestones = [2,32,64]
# # scheduler = MultiStepLR(optim, milestones=milestones, gamma=0.1)
#
# ### Training function
def train_epoch(vae, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    t = 0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for x, y in dataloader:
        t += 1
        # Move tensor to the proper device
        x = x.to(device)
        y_hat = vae(x)
        # if t % 100 == 0:
        #     print(y_hat)
        #     print(y)
        # print(y_hat.shape)
        # print(y.shape)
        # Evaluate loss
        loss = ((y - y_hat)**2).sum() + vae.encoder.kl
        # print(loss)
        # loss = loss_func(y_hat, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.item()))
        train_loss+=loss.item()

        # print(type(loss.item()))
        # print(type(train_loss))

        # max_float_value = sys.float_info.max
        # print(max_float_value)

    # scheduler.step()
    return train_loss / len(dataloader.dataset)
#
#
def test_epoch(vae, device, dataloader):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0
    with torch.no_grad(): # No need to track the gradients
        for x, y in dataloader:
            # Move tensor to the proper device
            x = x.to(device)
            # Encode data
            encoded_data = vae.encoder(x)
            # Decode data
            y_hat = vae(x)
            loss = ((y - y_hat)**2).sum() + vae.encoder.kl
            # loss = loss_func(y_hat, y)
            val_loss += loss.item()

    return val_loss / len(dataloader.dataset)
#
# ###
# ###
# ###
# #input directory
# # sacdir = "F:/35/check"  # pick some data for prediction, manually
# #
# # #output directory
# # resdir = "F:/35/result" # the corresponding label of the picked data, take manually
# #
# # ### if you want to visualize you can use this
# # inspectdir = "F:/35/inspect"
# #
# # os.makedirs(resdir,exist_ok=True)
# # os.makedirs(os.path.join(inspectdir,"model"),exist_ok=True)
# # os.makedirs(os.path.join(inspectdir,"label"),exist_ok=True)
# # os.makedirs(os.path.join(inspectdir,"graph"),exist_ok=True)
# #
# # i = 0
# # vae.eval()
#
def testing(sacdir, resdir, model, device):
    for narr in tqdm(os.listdir(sacdir)):
        # i += 1
        test_data = os.path.join(sacdir, narr)
        tsdata = np.load(test_data)
        sac = np.vstack((tsdata[0:1024], tsdata[1024:2048]))  # nz
        # sac = tsdata[:598]
        # x = torch.tensor(sac).float().unsqueeze(dim=0).unsqueeze(dim=0).to(device)  # nz
        x = torch.tensor(sac).float().unsqueeze(dim=0).to(device)  # nz
        data = model(x).cpu().detach().numpy().reshape(64, 64)
        np.save(os.path.join(resdir, narr), data)

# def npytesting(npypath, resdir, labeldir, model, device):
#     data = np.load(npypath)
#     length = data.shape[0]
#     for i in range(length):
#         sac = data[i,:598]
#         label = data[i, 598:].reshape(64,64)
#         x = torch.tensor(sac).float().unsqueeze(dim=0).unsqueeze(dim=0).to(device)
#         res = model(x).cpu().detach().numpy().reshape(64, 64)
#         np.save(os.path.join(resdir, f"{i}.npy"), res)
#         np.save(os.path.join(labeldir, f"{i}.npy"), label)


        # test_data = os.path.join(npydir, narr)
        # tsdata = np.load(test_data)
        # # sac = np.vstack((tsdata[0:1024], tsdata[1024:2048]))  # nz
        # sac = tsdata[:598]
        # x = torch.tensor(sac).float().unsqueeze(dim=0).unsqueeze(dim=0).to(device)  # nz
        # data = model(x).cpu().detach().numpy().reshape(64, 64)
        # np.save(os.path.join(resdir, narr), data)
#
# # for narr in tqdm(os.listdir(sacdir)):
# #     i += 1
# #     # if i < 11500:
# #     #     continue
# #         # exit()
# #     test_data = os.path.join(sacdir, narr)
# #     # test_label = os.path.join(birddir, narr)
# #     tsdata = np.load(test_data)
# #     sac = np.vstack((tsdata[0:1024], tsdata[1024:2048])) #nz
# #     # sac = tsdata[4096:8192] #z
# #     img = tsdata[2048:].reshape(64,64)
# #
# #     # sac = np.vstack((tsdata[0:1024], tsdata[1024:2048])) #nz
# #     # img = tsdata[2048:].reshape(64,64)
# #     # sac = tsdata[1024:2048] #z
# #     # img = tsdata[2048:].reshape(64,64)
# #
# #     x = torch.tensor(sac).float().unsqueeze(dim=0).to(device) #nz
# #     # x = torch.tensor(sac).float().unsqueeze(dim=0).unsqueeze(dim=0).to(device) #z
# #
# #     data = vae(x).cpu().detach().numpy().reshape(64,64)
# #     # print(data)
# #     np.save(os.path.join(resdir,narr), data)
# #
# #     # inspection
# #     # if i < 3000:
# #     #     plt.figure()
# #     #     plt.imshow(img)
# #     #     plt.savefig(f'{inspectdir}/label/{i}.png')
# #     #     plt.close()
# #     #     plt.figure()
# #     #     plt.imshow(data, cmap='viridis', vmin=0, vmax=0.7)
# #     #     plt.savefig(f'{inspectdir}/model/{i}.png')
# #     #     plt.close()
# #     #     # plt.figure()
# #     #     # plt.ylim(-8,8)
# #     #     # plt.plot(graph)
# #     #     # plt.savefig(f'{inspectdir}/graph/{i}.png')
# #     #     # plt.close()
# #
# # exit()
#
# ###
# ###
# ###
#
#
# train_dataset = VAEDataset1000ENZ1("F:/35/250nz64npz/train")
# dev_dataset = VAEDataset1000ENZ1("F:/35/250nz64npz/dev")
# test_dataset = VAEDataset1000ENZ1("F:/35/250nz64npz/test")
# train_loader = DataLoader(train_dataset, 128, True, num_workers=0)
# dev_loader = DataLoader(dev_dataset, 128, True, num_workers=0)
# test_loader = DataLoader(test_dataset, 128, True, num_workers=0)
# # train_dataset = VAEDataset1000ENZ1("F:/013/250nz_npz/check")
# # dev_dataset = train_dataset
# # test_dataset = train_dataset
# # train_loader = DataLoader(train_dataset, 128, True, num_workers=0)
# # dev_loader = train_loader
# # test_loader = train_loader
#
# # pred_idx + 36(34.5,35,37,38,41.5) = label_idx
#
# # obs_in = ["D:/Bafang/data/hope/18/1000nz/1/1676686127353_1_0.npy", # 1
# #           "D:/Bafang/data/hope/18/1000nz/1/1676685610804_3_0.npy", # 2
# #           "D:/Bafang/data/hope/18/1000nz/1/1676686164824_0_1.npy", # 2
# #           "D:/Bafang/data/hope/18/1000nz/1/1676685633693_1_0.npy", # 3
# #           "D:/Bafang/data/hope/18/1000nz/1/1676685532626_0_1.npy", # 3
# #           "D:/Bafang/data/hope/18/1000nz/1/1676686204297_2_0.npy", # 12
# #           "D:/Bafang/data/hope/18/1000nz/1/1676685630524_2_0.npy", # 12
# #           "D:/Bafang/data/hope/18/1000nz/1/1676685648074_2_1.npy", # 23
# #           "D:/Bafang/data/hope/18/1000nz/1/1676685733693_2_0.npy"] # 13
# #
# # obs_in_np = []
# # obs_out_np = []
# #
# # for i in range(9):
# #     v = np.load(obs_in[i])
# #     # sac = np.vstack((v[0:1024], v[1024:2048], v[2048:3072]))
# #     # sac = np.vstack((v[4096:8192],v[8192:12288]))
# #     # sac = v[1024:2048]
# #     # img = v[12288:]
# #
# #     sac = np.vstack((v[0:4096],v[4096:8192]))
# #     img = v[8192:]
# #
# #     sac_t = torch.tensor(sac).float().unsqueeze(dim=0).to(device)
# #
# #     obs_in_np.append(sac_t)
# #     obs_out_np.append(img)
# #
# # n_in = [
# #         "D:/Bafang/data/hopen/17/1000nz/1/1676604784560_1_0.npy",
# #         "D:/Bafang/data/hopen/17/1000nz/1/1676604775217_1_0.npy",
# #         "D:/Bafang/data/hopen/17/1000nz/1/1676604820729_2_0.npy",
# #         "D:/Bafang/data/hopen/17/1000nz/1/1676604988964_0_1.npy",
# #         # "D:/Bafang/data/hopen/17/enz/1/1676604682458_1_0.npy",
# #         # "D:/Bafang/data/hopen/17/enz/1/1676604736712_2_0.npy",
# #         ]
# #
# # n_in_np = []
# # n_out_np = []
# # for i in range(4):
# #     v = np.load(n_in[i])
# #
# #     # sac = np.vstack((v[0:1024], v[1024:2048], v[2048:3072]))
# #     # sac = np.vstack((v[4096:8192],v[8192:12288]))
# #     # sac = v[1024:2048]
# #     # img = v[12288:]
# #
# #     sac = np.vstack((v[0:4096],v[4096:8192]))
# #     img = v[8192:]
# #
# #     sac_t = torch.tensor(sac).float().unsqueeze(dim=0).to(device)
# #
# #     n_in_np.append(sac_t)
# #     n_out_np.append(img)
#
def training(basedir, model, device, train_loader, dev_loader, test_loader, optim, num_epochs=500):
    os.makedirs(os.path.join(basedir, "model"), exist_ok=True)
    os.makedirs(os.path.join(basedir, "m_label"), exist_ok=True)
    os.makedirs(os.path.join(basedir, "m_pred"), exist_ok=True)
    os.makedirs(os.path.join(basedir, "n_label"), exist_ok=True)
    os.makedirs(os.path.join(basedir, "n_pred"), exist_ok=True)
    shutil.copyfile("S2R1000ENZ.py", os.path.join(basedir, "code.txt"))
    for epoch in tqdm(range(num_epochs)):
        train_loss = train_epoch(model, device, train_loader, optim)
        val_loss = test_epoch(model, device, dev_loader)
        test_loss = test_epoch(model, device, test_loader)
        print('\n EPOCH {}/{} \t train loss {:.6f} \t val loss {:.6f} \t test loss {:.6f}'.format(epoch + 1, num_epochs,
                                                                                                  train_loss, val_loss,
                                                                                                  test_loss))
        if epoch % 2 == 0:
            torch.save(model, f'{os.path.join(basedir, "model")}/{epoch}_{train_loss}_{val_loss}_{test_loss}.pth')
#
#
#
# os.makedirs(os.path.join(basep,"model"), exist_ok=True)
# os.makedirs(os.path.join(basep,"m_label"), exist_ok=True)
# os.makedirs(os.path.join(basep,"m_pred"), exist_ok=True)
# os.makedirs(os.path.join(basep,"n_label"), exist_ok=True)
# os.makedirs(os.path.join(basep,"n_pred"), exist_ok=True)
# shutil.copyfile("S2R1000ENZ.py", os.path.join(basep, "code.txt"))
#
# for epoch in tqdm(range(num_epochs)):
#     # if epoch == 200:
#     #     optim.param_groups[0]['weight_decay'] = 1e-3
#     # if epoch == 1000:
#     #     optim.param_groups[0]['weight_decay'] = 1e-4
#     # if epoch == 2000:
#     #     optim.param_groups[0]['weight_decay'] = 1e-5
#     # if epoch == 10:
#     #     lr = 1e-6
#     #     optim = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-4, eps=1e-8)
#     # if epoch == 100:
#     #     lr = 1e-7
#     #     optim = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-4, eps=1e-8)
#     train_loss = train_epoch(vae,device,train_loader,optim)
#     val_loss = test_epoch(vae, device, dev_loader)
#     test_loss = test_epoch(vae, device, test_loader)
#
#     # val_loss = 0
#     # if epoch % 10 == 0:
#     #     val_loss = test_epoch(vae, device, dev_loader)
#     print('\n EPOCH {}/{} \t train loss {:.6f} \t val loss {:.6f} \t test loss {:.6f}'.format(epoch + 1, num_epochs,
#                                                                                               train_loss, val_loss,
#                                                                                               test_loss))
#     if epoch % 2 == 0:
#         torch.save(vae, f'{os.path.join(basep,"model")}/{epoch}_{train_loss}_{val_loss}_{test_loss}.pth')
#         # for i in range(9): #station 228 "m"
#         #     xin = obs_in_np[i]
#         #     yout = obs_out_np[i]
#         #     data = vae(xin).cpu().detach().numpy().reshape(64, 64)
#         #     if epoch == 0:
#         #         y = yout.reshape(64,64)
#         #         plt.figure()
#         #         plt.imshow(y)
#         #         plt.savefig(f'{os.path.join(basep,"m_label")}/{epoch}_{i}.png')
#         #         plt.close()
#         #     plt.figure()
#         #     plt.imshow(data)
#         #     plt.savefig(f'{os.path.join(basep,"m_pred")}/{epoch}_{i}.png')
#         #     plt.close()
#         #
#         # for i in range(4): #station 194 "n"
#         #     # if i < 3:
#         #     #     x = np.load(n_in[i])
#         #     # else:
#         #     #     x = np.load(n_in[i])*0.754
#         #     # xin = torch.tensor(x[28672:32768]).float().unsqueeze(dim=0).unsqueeze(dim=0).to(device)
#         #     xin = n_in_np[i]
#         #     yout = n_out_np[i]
#         #     data = vae(xin).cpu().detach().numpy().reshape(64, 64)
#         #     if epoch == 0:
#         #         y = yout.reshape(64,64)
#         #         plt.figure()
#         #         plt.imshow(y)
#         #         plt.savefig(f'{os.path.join(basep,"n_label")}/{epoch}_{i}.png')
#         #         plt.close()
#         #     plt.figure()
#         #     plt.imshow(data)
#         #     plt.savefig(f'{os.path.join(basep,"n_pred")}/{epoch}_{i}.png')
#         #     plt.close()
#
