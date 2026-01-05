import torch
from torch import nn

class EncoderProjectorQFormer(nn.Module):
    def __init__(self, encoder_dim=768, llm_dim=2048, k=8):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        from transformers import Blip2QFormerConfig, Blip2QFormerModel
        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.encoder_dim
        # configuration.num_hidden_layers = config.qformer_layers
        configuration.num_hidden_layers = k ##make it config

        # self.query_len = int(config.get("query_len", 64))
        self.query_len = 64
        self.query = nn.Parameter(torch.zeros(1, self.query_len, configuration.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)

        self.linear = nn.Linear(configuration.hidden_size, self.llm_dim)
        self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)

    def forward(self, x, atts=None):
        query = self.query.expand(x.shape[0], -1, -1)
        
        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        
        query_proj = self.norm(self.linear(query_output.last_hidden_state))
        
        return query_proj


def get_connector(name, audio_enc_dim, llm_dim, k):
    if name == 'linear-pool':
        return LinearPoolConnector(audio_enc_dim, llm_dim, k)
    elif name == 'linear':
        return LinearConnector(audio_enc_dim, llm_dim, k)
    elif name == 'cnn':
        return CNNConnector(audio_enc_dim, llm_dim, k)
    elif name == 'qformer':
        return EncoderProjectorQFormer(audio_enc_dim, llm_dim, k)
    else:
        raise NotImplementedError

class LinearConnector(nn.Module):
    def __init__(self, in_dim, out_dim, k):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.pool = nn.AvgPool1d(kernel_size=k, stride=k)

    def forward(self, x):
        x = self.layer(x)
        x = x.transpose(1, 2) 
        x = self.pool(x)  
        x = x.transpose(1, 2)
        return x


class LinearPoolConnector(nn.Module):
    def __init__(self, input_dim, output_dim, k):
        super(LinearPoolConnector, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU())
        self.pool = nn.AvgPool1d(kernel_size=k, stride=k)
        self.linear2 = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim))

    def forward(self, x):
        # x: [B, T, d]
        x = self.linear1(x)  # x: [B, T, D]
        x = x.transpose(1, 2)  # x: [B, D, T]
        x = self.pool(x)  # x: [B, D, T']
        x = x.transpose(1, 2)  # x: [B, T', D]
        x = self.linear2(x)
        return x

class CNNConnector(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels//2, kernel_size=5,
                      stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(out_channels//2, out_channels, kernel_size=5,
                      stride=k, padding=0),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5,
                      stride=1, padding=0),
        )

    def forward(self, x):
        return self.layer(x.transpose(1,2)).transpose(1,2)



if __name__ == "__main__":
    model = CNNConnector(128, 256, 2)
    x = torch.randn(4, 50, 128)
    z = model(x)
    print(z.shape)
    for name, module in model.named_modules():
        if not isinstance(module, nn.Sequential) and not isinstance(module, model): # Exclude container modules
            output = module(output)
            print(f"Layer: {name}, Output Shape: {output.shape}")