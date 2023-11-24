import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from audio_processing import window_sumsquare


class AdaptiveAudioNormalization(nn.Module):
    def __init__(self, sigma=20, truncate=4.0):
        super(AdaptiveAudioNormalization, self).__init__()

        filter_ = self.get_gaus_filter1d(sigma, truncate)
        self.register_buffer("filter_", filter_)
        self.reflect = torch.nn.ReflectionPad1d(sigma * int(truncate))

    def forward(self, spect):
        spect = torch.log1p(spect * 1048576)
        mean = spect.mean(dim=1, keepdim=True)
        mean = self.reflect(mean)
        mean = F.conv1d(mean, self.filter_)
        mean_mean = mean.mean(dim=-1, keepdim=True)
        spect = spect.add(-mean_mean)
        return spect

    @staticmethod
    def get_gaus_filter1d(sigma, truncate=4.0):
        sd = float(sigma)
        lw = int(truncate * sd + 0.5)
        sigma2 = sigma * sigma
        x = np.arange(-lw, lw + 1)
        phi_x = np.exp(-0.5 / sigma2 * x**2)
        phi_x = phi_x / phi_x.sum()
        return torch.FloatTensor(phi_x.reshape(1, 1, -1))


class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(
        self, filter_length=800, hop_length=200, win_length=800, window="hann"
    ):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )

        if window is not None:
            assert filter_length >= win_length
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        )
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length,
                dtype=np.float32,
            )
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0]
            )
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False
            )
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[
                approx_nonzero_indices
            ]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2) :]
        inverse_transform = inverse_transform[:, :, : -int(self.filter_length / 2) :]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ):
        super().__init__()

        self.dw_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                groups=in_channels,
            ),
            nn.Identity(),
            nn.ReLU(),
        )

        self.pw_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
            nn.Identity(),
        )

        self.proj = nn.Conv1d(
            in_channels=out_channels,
            out_channels=in_channels,
            kernel_size=1,
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        x = self.proj(x)
        x = self.activation(x)
        return x


class SileroVADModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = STFT()

        self.first_layer = nn.Sequential(
            ConvBlock(258, 16, kernel_size=5),
            nn.Dropout(),
        )

        self.encoder = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            ConvBlock(16, 32, kernel_size=5),
            nn.Dropout(0.5),
            nn.Conv1d(32, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            ConvBlock(32, 32, kernel_size=5),
            nn.Dropout(0.5),
            nn.Conv1d(32, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            ConvBlock(32, 64, kernel_size=5),
            nn.Dropout(0.5),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.LSTM(64, 64, 2),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_layer(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    model = STFT(filter_length=256, win_length=256)
    print(model.forward_basis.shape)
    # model = AdaptiveAudioNormalization()
    # print(model.filter_.shape)
    # x = torch.rand(1, 10000)
    # y = torch.stft(x, n_fft=512, onesided=True, return_complex=True)
    # print(y.shape)
    # self.feature_extractor = torch.stft(onesided=True)
    # model = SileroVADModel()
    # model.eval()
    # # script_model = torch.jit.script(model)
    # # script_model.save(model.jit)
    # print(sum([p.numel() for p in model.parameters()]))
