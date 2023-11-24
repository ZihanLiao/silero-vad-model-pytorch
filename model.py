import torch
import torch.nn as nn


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
    x = torch.rand(1, 10000)
    y = torch.stft(x, n_fft=512, onesided=True, return_complex=True)
    print(y.shape)
    # self.feature_extractor = torch.stft(onesided=True)
    # model = SileroVADModel()
    # model.eval()
    # # script_model = torch.jit.script(model)
    # # script_model.save(model.jit)
    # print(sum([p.numel() for p in model.parameters()]))
