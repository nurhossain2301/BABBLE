import torch.nn as nn
import torch

class QFormer(nn.Module):
    def __init__(self, num_queries=32, hidden_dim=768, num_layers=6, nhead=8, llm_dim=3072):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim))  # [1, Q, D]
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(hidden_dim, llm_dim)

    def forward(self, audio_embeds):
        """
        audio_embeds: [B, T, D] from audio encoder
        """
        B = audio_embeds.size(0)
        queries = self.queries.expand(B, -1, -1)  # [B, Q, D]
        # Attend audio embeddings with queries
        combined = torch.cat([queries, audio_embeds], dim=1)  # [B, Q+T, D]
        output = self.transformer(combined)[:, :queries.size(1), :]  # [B, Q, D]
        output = self.linear(output)
        return output