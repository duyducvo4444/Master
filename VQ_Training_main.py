import argparse
import json
import torch
from create_data import CreateDataForSE
from torch.utils.data import DataLoader
import models
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


def train(model_name, batch_size=None, learning_rate=1e-3, clean_dir_train='', noise_dir_train='',
          sample_rate=16000, data_duration=4.0, train_epoch=0, checkpoint="", pretrain_pt=''):
    """Training the Speech Enhancement using gammatone analysis & synthesis models"""

    epochs = train_epoch
    tr_loss = []
    t_loss = []
    perplexity_t = []
    perplexity_b = []

    # Create data for training
    datatrain = CreateDataForSE(clean_url=clean_dir_train, noise_url=noise_dir_train,
                                samplerate=sample_rate, augmentation=True, duration=data_duration)
    dataloader = DataLoader(datatrain, batch_size=batch_size, shuffle=True)

    model = getattr(models, model_name)(**model_config)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    start = 0
    validator = Validation(**test_config)

    # Load pretrain parameters
    model.copy_state_dict(torch.load(pretrain_pt)['state_dict'])

    if checkpoint != "":
        chkpt = torch.load(checkpoint)
        model.copy_state_dict(chkpt['state_dict'])
        start = chkpt['start'] + 1
        tr_loss = chkpt['loss']
        t_loss = chkpt['test_loss']
        perplexity_b = chkpt['perplex_bot']
        perplexity_t = chkpt['perplex_top']

    for i in range(start, epochs):
        print('Epoch ' + str(i) + ' running')
        model.train()
        train_loss = []
        perp_t = []
        perp_b = []
        for batch, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            model.zero_grad()
            loss_components = model(x, y, train_clean=False)
            loss = loss_components["loss"]
            loss.backward()
            optimizer.step()

            perplex_top = loss_components["plx_top"]
            perplex_bot = loss_components["plx_bot"]
            if checkNan(loss.item()):
                torch.save(loss_components, 'Trained/NaN_' + str(i) + '.pt')
                raise ValueError('NaN in step' + str(batch))
            train_loss.append(loss.item())
            perp_t.append(perplex_top.item())
            perp_b.append(perplex_bot.item())
            if batch > 3:  # edit based on $batch_size
                break

        tr_loss.append(np.mean(train_loss))
        perplexity_t.append(np.mean(perp_t))
        perplexity_b.append(np.mean(perp_b))
        t_loss.append(validator(model))
        print("Epoch:", i, "Training Loss: ", tr_loss[i], "Testing loss: ", t_loss[i])
        if i % 5 == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "start": i,
                "loss": tr_loss,
                "test_loss": t_loss,
                "perplex_top": perplexity_t,
                "perplex_bot": perplexity_b
            }
            torch.save(checkpoint, 'Trained/SEModel_VQ2_' + str(i) + '.pt')

    print('Training done!')

    return 0


def checkNan(num):
    return num != num


class Validation(object):
    def __init__(self, clean_dir, noise_dir, samplerate, data_duration=None):
        self.datatest = CreateDataForSE(clean_url=clean_dir, noise_url=noise_dir,
                                        samplerate=samplerate, augmentation=True, duration=data_duration)
        self.dataloader = DataLoader(self.datatest)

    def __call__(self, model):
        model.eval()
        test_loss = []
        for batch, (x, y) in enumerate(self.dataloader):
            with torch.no_grad():
                x = x.to(device)
                y = y.to(device)
                loss_components = model(x, y, train_clean=False)
                loss = loss_components["loss"]
                test_loss.append(loss.item())
            if batch > 100:
                break
        return np.mean(test_loss)


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
    global test_config
    test_config = config['validation_configs']
    train(**train_config)
