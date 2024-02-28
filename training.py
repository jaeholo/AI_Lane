from S2R1000ENZ import VariationalAutoencoder
from S2R1000ENZ import training
# from S2R1000ENZ import npytesting
import torch
from VAEDataset1000ENZ import VAEDataset1000ENZ1
from torch.utils.data import DataLoader


if __name__ == '__main__':

    '''
    hyper-parameter
    d: latent dimensions for VAE
    lr: learning rate
    optim: optimizer
    dataset: the train,test and dev dataset which are created manually in the last step in preparedata.py
    basedir
    '''
    d = 8
    vae = VariationalAutoencoder(latent_dims=d)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    vae.to(device)
    lr = 1e-6
    optim = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-3, eps=1e-8)
    #
    train_dataset = VAEDataset1000ENZ1("F:/34/250nz64npz/train")
    dev_dataset = VAEDataset1000ENZ1("F:/34/250nz64npz/dev")
    test_dataset = VAEDataset1000ENZ1("F:/34/250nz64npz/test")
    # train_dataset, dev_dataset, test_dataset = torch.utils.data.random_split(VAEDataset1000ENZ1(r"C:\Users\hp\Desktop\pca_data.npy"), (0.8, 0.1, 0.1))
    train_loader = DataLoader(train_dataset, 128, True, num_workers=0)
    dev_loader = DataLoader(dev_dataset, 128, True, num_workers=0)
    test_loader = DataLoader(test_dataset, 128, True, num_workers=0)

    '''
    training function use the dataset created before to train the model
    the model will be saved to the based dir
    035 means the 35th station
    1 means the 1st model
    '''
    basedir = "models/034/4"
    training(basedir=basedir, model=vae, device=device, train_loader=train_loader, dev_loader=dev_loader,
             test_loader=test_loader, optim=optim, num_epochs=500)

    '''
    the following phase is the testing phase, put the unseen sac data into the model
    sac_dir : where to get the unseen sac data, pick some data for prediction, manually
    resdir: the output directory, the corresponding label of the picked data, take manually
    model: the model we trained before  
    '''
    # npypath = r"C:\Users\hp\Desktop\check_data.npy"
    # resdir = "F:/35/result1"
    # labeldir = r"F:\35\label1"
    # model = torch.load("models/035/2/model/244_79.30639851937664_76.58439362632848_76.09489795707218.pth")
    # model.to(device)
    # npytesting(npypath=npypath, resdir=resdir, labeldir=labeldir, model=model, device=device)
    # print("Testing done!")
