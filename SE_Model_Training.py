import argparse
import json
import torch
from create_data import CreateDataForSE
from torch.utils.data import DataLoader
import models
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


def _main():

    epochs = train_config['train_epoch']
    tr_loss = np.zeros(epochs)
    perplexity_t = np.zeros(epochs)
    perplexity_b = np.zeros(epochs)

    # Training process
    for i in range(epochs):
        print(f"Epoch {i + 1}\n-------------------------------")
        tr_loss[i], perplexity_t[i], perplexity_b[i] = train(**train_config)
        print("Epoch:", i, "Training Loss: ", tr_loss[i])

        epoch_nb = np.arange(0, i, 1)
        plt.figure(figsize=(5, 4))
        plt.plot(epoch_nb, tr_loss[:i])
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss (Linear)')
        plt.savefig('fig/Loss2/Training_loss_lin_' + str(i) + '.png')

        plt.figure(figsize=(5, 4))
        plt.plot(epoch_nb, perplexity_t[:i])
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.savefig('fig/Per2/Perplexity Top.png_' + str(i) + '.png')

        plt.figure(figsize=(5, 4))
        plt.plot(epoch_nb, perplexity_b[:i])
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.savefig('fig/Per2/Perplexity bot.png_' + str(i) + '.png')
    print('Training done!')


def train(model_name, batch_size, learning_rate=1e-3, clean_dir_train='', noisy_dir_train='',
          sample_rate=16000, data_duration=4.0, train_epoch=0):
    """Training the Speech Enhancement using gammatone analysis & synthesis models"""

    # Create data for training
    datatrain = CreateDataForSE(clean_dir_train, noisy_dir_train, samplerate=sample_rate, duration=data_duration)
    dataloader = DataLoader(datatrain, batch_size=batch_size, shuffle=True)

    model = getattr(models, model_name)(**model_config)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    model.train()
    train_loss = []
    perp_t = []
    perp_b = []
    for batch, (y, x) in enumerate(dataloader):

        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss_components = model(y, x, train_clean=False)
        loss = loss_components["loss"]
        loss.backward()
        optimizer.step()

        perplex_top = loss_components["plx_top"]
        perplex_bot = loss_components["plx_bot"]
        train_loss.append(loss.item())
        perp_t.append(perplex_top.item())
        perp_b.append(perplex_bot.item())

    torch.save(model.state_dict(), 'SEModel_VQ2.pt')

    return np.mean(train_loss), np.mean(perp_t), np.mean(perp_b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='JSON file for configuration')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    global model_config
    model_config = config['model_configs']
    global train_config
    train_config = config['training_configs']
    _main()
