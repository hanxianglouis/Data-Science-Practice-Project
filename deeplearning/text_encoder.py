import torch
import torch.nn as nn

# Transformer-based Text Encoder
class TextEncoderTransformer(nn.Module):
    """
    基于 Transformer Encoder 的文本编码器，输出论文中的 F_txt
    输入:  [B, max_len]，每个元素是 token id
    输出:  [B, max_len, emb_dim]

    说明：
    - token id 先经过 embedding，得到 [B, max_len, emb_dim]。
    - 加入可学习的位置编码，保留 token 的顺序信息。
    - Transformer Encoder 对每个 token 进行上下文建模。
    - 不做 pooling，不取最后一个 hidden state，因此输出仍然是 token 序列。
    - 最终输出 F_txt，形状为 [B, max_len, emb_dim]。
    """
    def __init__(self, vocab_size, emb_dim=64, max_len=10, device='cuda', num_heads=8, num_layers=4, dropout=0.1):
        super(TextEncoderTransformer, self).__init__()
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.device = device

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.emb_dim
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=self.max_len,
            embedding_dim=self.emb_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=num_heads,
            dim_feedforward=self.emb_dim * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        self.to(self.device)

    def forward(self, x):
        """
        x: [B, max_len]，每个元素是 token id
        return: [B, max_len, emb_dim]
        """
        x = x.to(self.device)  # [B, max_len]
        batch_size, seq_len = x.shape

        token_emb = self.token_embedding(x)  # [B, max_len, emb_dim]

        positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.position_embedding(positions)  # [B, max_len, emb_dim]

        x = token_emb + pos_emb  # [B, max_len, emb_dim]
        text_features = self.transformer_encoder(x)  # [B, max_len, emb_dim]

        return text_features