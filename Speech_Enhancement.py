import librosa
import torch
from models.Torch_Conv1d_Gamatone import ConvGamma
from models.Torch_Conv1d_Gamatone_Synthesis import InvConvGamma
from gtfblib import gtfb
import soundfile as sf

# Declaring parameters
noisy_file = 'wav/male_airCon_10.wav'

""" DO NOT CHANGE THE FOLLOWING PARAMETERS """
N = 1024
sr = 16000
low_freq = 20
high_freq = sr // 2
n_gamma = 32
gamma_filt_len = N + 1
""" DO NOT CHANGE THE UPPER PARAMETERS """

batch_size = 10
data_duration = 1.0  # in seconds
epochs = 200

initialize_ana_kernels = True

device = "cpu"
print("Using {} device".format(device))


def _main():
    # Prepare the models
    #SEModel = SENetwork(input_size=n_gamma, hidden_layer_size=100, num_layers=2, output_size=n_gamma,
                        #batch_size=batch_size)
    cfs = gtfb.ERBspacing_given_N(low_freq, high_freq, n_gamma)
    ana_model = ConvGamma(N=N, sr=sr, nb_gamma_filters=n_gamma, gamma_filt_len=gamma_filt_len, cfs=cfs,
                          initialize_ana_kernels=initialize_ana_kernels).to(device)
    syn_model = InvConvGamma(N=N, sr=sr, nb_gamma_filters=n_gamma, gamma_filt_len=gamma_filt_len, trained=True)
    syn_model.to(device)
    #SEModel.load_state_dict(torch.load('SEModel.pt'))
    #SEModel.to(device)

    ana_model.eval()
    #SEModel.eval()
    syn_model.eval()

    speech, _ = librosa.load(noisy_file, sr=sr)
    input_x = torch.FloatTensor(speech)
    input_x = input_x.unsqueeze(0)
    xr, xi = ana_model(input_x)
    mag = torch.sqrt(torch.square(xr) + torch.square(xi))
    phase = torch.atan2(xi, xr)
    # yr, yi = SEModel(xr, xi)
    output = syn_model(mag, phase)
    output = output.squeeze()
    #
    sf.write('female_syn.wav', output.detach().numpy(), samplerate=sr)


if __name__ == '__main__':
    _main()
