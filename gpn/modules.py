from flash_attn.models.bert import BertEncoder
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
)


class TransposeLayer(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        return x


class ConvLayer(nn.Module):
    def __init__(
        self,
        hidden_size=None,
        intermediate_size=None,
        hidden_dropout_prob=None,
        bias=None,
        **kwargs,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.LayerNorm(hidden_size, bias=bias),
            TransposeLayer(),
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                padding="same",
                bias=bias,
                **kwargs,
            ),
            TransposeLayer(),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=bias),
            nn.Dropout(hidden_dropout_prob),
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_size, bias=bias),
            nn.Linear(hidden_size, intermediate_size, bias=bias),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size, bias=bias),
            nn.Dropout(hidden_dropout_prob),
        )

    def forward(self, x):
        x = x + self.conv(x)
        x = x + self.ffn(x)
        return x


class ByteNetLayer(nn.Module):
    def __init__(
        self,
        hidden_size=None,
        slim=False,
        bias=None,
        **kwargs,
    ):
        super().__init__()
        intermediate_size = hidden_size // 2 if slim else hidden_size
        self.layer = nn.Sequential(
            nn.LayerNorm(hidden_size, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_size, intermediate_size, bias=bias),
            nn.LayerNorm(intermediate_size, bias=bias),
            nn.GELU(),
            TransposeLayer(),
            nn.Conv1d(
                in_channels=intermediate_size,
                out_channels=intermediate_size,
                padding="same",
                bias=bias,
                **kwargs,
            ),
            TransposeLayer(),
            nn.LayerNorm(intermediate_size, bias=bias),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size, bias=bias),
        )

    def forward(self, x):
        x = x + self.layer(x)
        return x


class OneHotEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x):
        return F.one_hot(x, num_classes=self.hidden_size).float()


def get_dilation_schedule(config):
    return [
        min(
            config.dilation_max,
            config.dilation_base
            ** ((i % config.dilation_cycle) // config.dilation_double_every),
        )
        for i in range(config.num_hidden_layers)
    ]


class ConvNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dilation_schedule = get_dilation_schedule(config)
        print(f"{dilation_schedule=}")
        self.layer = nn.Sequential(
            *[
                ConvLayer(
                    hidden_size=config.hidden_size,
                    kernel_size=config.first_kernel_size if i == 0 else config.rest_kernel_size,
                    dilation=dilation_schedule[i],
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    bias=config.bias,
                    intermediate_size=config.intermediate_size,
                    groups=1 if (not config.depthwise or i == 0) else config.hidden_size,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    def forward(self, hidden_states):
        hidden_states = self.layer(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


class ByteNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dilation_schedule = get_dilation_schedule(config)
        print(f"{dilation_schedule=}")
        self.layer = nn.Sequential(
            *[
                ByteNetLayer(
                    hidden_size=config.hidden_size,
                    kernel_size=config.first_kernel_size if i == 0 else config.rest_kernel_size,
                    dilation=dilation_schedule[i],
                    bias=config.bias,
                    groups=1 if (not config.depthwise or i == 0) else config.hidden_size,
                    slim=config.slim,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    def forward(self, hidden_states):
        hidden_states = self.layer(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LayerNorm(input_size, bias=False),
            nn.Linear(input_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, output_size, bias=False),
        )
        if input_size != output_size:
            self.shortcut = nn.Linear(input_size, output_size, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.shortcut(x) + self.layer(x)


class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size=None, **kwargs):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LayerNorm(input_size, bias=False),
            TransposeLayer(),
            nn.Conv1d(input_size, hidden_size, kernel_size, bias=False, padding="same", **kwargs),
            TransposeLayer(),
            nn.GELU(),
            nn.Linear(hidden_size, output_size, bias=False),
        )
        if input_size != output_size:
            self.shortcut = nn.Linear(input_size, output_size, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.shortcut(x) + self.layer(x)


# source: https://github.com/lucidrains/enformer-pytorch
def exponential_linspace_int(start, end, num, divisible_by = 1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]


class DownsampleLayer(nn.Module):
    def __init__(
        self,
        in_channels=None,
        out_channels=None,
        bias=None,
        **kwargs,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.LayerNorm(in_channels, bias=bias),
            TransposeLayer(),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                bias=bias,
                **kwargs,
            ),
            TransposeLayer(),
            nn.LayerNorm(out_channels, bias=bias),
            nn.GELU(),
            nn.Linear(out_channels, out_channels, bias=bias),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsampleLayer(nn.Module):
    def __init__(
        self,
        in_channels=None,
        out_channels=None,
        bias=None,
        **kwargs,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.LayerNorm(in_channels, bias=bias),
            TransposeLayer(),
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                bias=bias,
                **kwargs,
            ),
            TransposeLayer(),
            nn.LayerNorm(out_channels, bias=bias),
            nn.GELU(),
            nn.Linear(out_channels, out_channels, bias=bias),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DownPyramid(nn.Module):
    def __init__(self, dims=None, kernel_sizes=None, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_channels, out_channels, kernel_size in zip(
            dims[:-1], dims[1:], kernel_sizes
        ):
            self.layers.append(DownsampleLayer(
                in_channels, out_channels, kernel_size=kernel_size, **kwargs
            ))

    def forward(self, x):
        old_xs = []
        for layer in self.layers:
            x, old_x = layer(x)
            old_xs.append(old_x)
        return x, old_xs


class UpPyramid(nn.Module):
    def __init__(self, dims=None, kernel_sizes=None, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_channels, out_channels, kernel_size in zip(
            dims[:-1], dims[1:], kernel_sizes
        ):
            self.layers.append(UpsampleLayer(
                in_channels, out_channels, kernel_size=kernel_size, **kwargs
            ))

    def forward(self, x, old_xs):
        for layer, old_x in zip(self.layers, old_xs):
            x = layer(x, old_x)
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        config.head_dim = config.hidden_size // config.num_attention_heads
        config.rotary_emb_dim = config.head_dim
        config.intermediate_size = config.hidden_size * 4
        self.model = BertEncoder(config)
    
    def forward(self, x):
        return self.model(x)


class UNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        n_downsampling = math.log(config.bin_size, 2)
        assert n_downsampling.is_integer()
        n_downsampling = int(n_downsampling)
        self.config.dims = exponential_linspace_int(
            config.base_hidden_size, config.hidden_size, 1+n_downsampling
        )
        kernel_sizes = (
            [config.first_kernel_size] +
            [config.rest_kernel_size] * (n_downsampling)
        )
        self.down_pyramid = DownPyramid(
            dims=self.config.dims, kernel_sizes=kernel_sizes, bias=config.bias,
        )
        self.transformer = Transformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, x):
        x, old_xs = self.down_pyramid(x)
        x = self.transformer(x)
        return BaseModelOutput(last_hidden_state=x, hidden_states=old_xs)
