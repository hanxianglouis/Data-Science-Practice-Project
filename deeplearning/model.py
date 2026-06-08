import torch
import torch.nn as nn
import torch.nn.functional as F

from image_encoder import ImageEncoderResnNet50
from text_encoder import TextEncoderTransformer

class CrossModalAttentionFusion(nn.Module):
    """
    论文中的 cross-modal attention fusion。
    输入:
        image_features: [B, m, emb_dim]
        text_features:  [B, n, emb_dim]
    输出:
        fusion_embedding: [B, emb_dim]
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, device='cuda'):
        super(CrossModalAttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        self.device = device

        self.image_to_text_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.text_to_image_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

        self.to(self.device)

    def forward(self, image_features, text_features):
        """
        image_features: [B, m, emb_dim]
        text_features:  [B, n, emb_dim]
        return: [B, emb_dim]
        """
        image_features = image_features.to(self.device)
        text_features = text_features.to(self.device)

        # image query attends to text key/value: [B, m, emb_dim]
        image_attended, _ = self.image_to_text_attention(
            query=image_features,
            key=text_features,
            value=text_features
        )

        # text query attends to image key/value: [B, n, emb_dim]
        text_attended, _ = self.text_to_image_attention(
            query=text_features,
            key=image_features,
            value=image_features
        )

        # 分类前需要把 token 序列压成整体 embedding
        image_pooled = image_attended.mean(dim=1)  # [B, emb_dim]
        text_pooled = text_attended.mean(dim=1)    # [B, emb_dim]

        fusion_input = torch.cat([image_pooled, text_pooled], dim=-1)  # [B, 2 * emb_dim]
        fusion_embedding = self.fusion_mlp(fusion_input)               # [B, emb_dim]

        return fusion_embedding
    

class CNNTransformer(nn.Module):
    """
    CNN + Transformer 多模态分类模型。
    输入形式与 MultiviewModel 类似，但默认同时使用 image 和 text 两个模态。

    imgs:  [B, 3, 244, 244]
    texts: [B, max_len]

    image_encoder 输出 F_img: [B, m, emb_dim]
    text_encoder 输出 F_txt:  [B, n, emb_dim]
    cross-modal attention fusion 后得到 [B, emb_dim]
    final_linear 输出分类 logit: [B, cat_number]
    """
    def __init__(self, cat_number, vocab_size, max_len=10, emb_dim=64, eps=0.1, device='cuda', num_heads=8, num_layers=4, dropout=0.1, CL=True, freeze_image_backbone=True, has_image_encoder=True, has_text_encoder=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.device = device
        self.vocab_size = vocab_size
        self.cat_number = cat_number
        self.max_len = max_len
        self.eps = eps
        self.CL = CL

        self.has_image_encoder = has_image_encoder
        self.has_text_encoder = has_text_encoder

        self.CL_loss_weight = 0.05
        self.CL_loss_image_weight = 1
        self.CL_loss_text_weight = 1

        if self.vocab_size is None:
            raise ValueError("To use CNNTransformer, vocab_size must be provided")
        
        if (not self.has_image_encoder) and (not self.has_text_encoder) :
            raise ValueError("At least one modality should be utilized.")

        if self.has_image_encoder :
            print(">>> Initializing ResNet50 Image Encoder")
            self.image_encoder = ImageEncoderResnNet50(
                emb_dim=self.emb_dim,
                device=self.device,
                freeze_backbone=freeze_image_backbone
            )

        if self.has_text_encoder :
            print(">>> Initializing Transformer Text Encoder")
            self.text_encoder = TextEncoderTransformer(
                vocab_size=self.vocab_size,
                emb_dim=self.emb_dim,
                max_len=self.max_len,
                device=self.device,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout
            )

        if self.has_image_encoder and self.has_text_encoder :
            print(">>> Initializing Cross-Modal Attention Fusion")
            self.fusion_layer = CrossModalAttentionFusion(
                embed_dim=self.emb_dim,
                num_heads=num_heads,
                dropout=dropout,
                device=self.device
            )

        self.final_linear = nn.Linear(self.emb_dim, self.cat_number, bias=True, device=self.device)
        self.to(self.device)

    def forward(self, imgs, texts, pertubed=True):

        # ------------- Encoding --------------
        if self.has_image_encoder :
            image_features = self.image_encoder(imgs)  # [B, m, emb_dim]
            if pertubed :
                noise_image = torch.randn_like(image_features) * self.eps
                image_features_pertubed = image_features + noise_image
        
        if self.has_text_encoder :
            text_features = self.text_encoder(texts)   # [B, n, emb_dim]
            if pertubed :
                noise_text = torch.randn_like(text_features) * self.eps
                text_features_pertubed = text_features + noise_text

        # ------------- Fusion --------------
        if self.has_image_encoder and self.has_text_encoder :
            final_embedding = self.fusion_layer(image_features, text_features)  # [B, emb_dim]
        elif self.has_image_encoder and (not self.has_text_encoder) :
            final_embedding = image_features.mean(dim=1)
        elif self.has_text_encoder and (not self.has_image_encoder) :
            final_embedding = text_features.mean(dim=1) 

        logit = self.final_linear(final_embedding)                          # [B, cat_number]

        # ------------- Contrastive Learning --------------
        CL_loss = torch.tensor(0.0, device=self.device)
        if pertubed:
            if self.has_image_encoder :
                image_embedding = image_features.mean(dim=1)                    # [B, emb_dim]
                image_embedding_pertubed = image_features_pertubed.mean(dim=1)  # [B, emb_dim]
                CL_loss_image = self.info_nce_loss(image_embedding, image_embedding_pertubed)

                CL_loss += self.CL_loss_image_weight * CL_loss_image
            
            if self.has_text_encoder :
                text_embedding = text_features.mean(dim=1)                      # [B, emb_dim]
                text_embedding_pertubed = text_features_pertubed.mean(dim=1)    # [B, emb_dim]
                CL_loss_text = self.info_nce_loss(text_embedding, text_embedding_pertubed)

                CL_loss += self.CL_loss_text_weight * CL_loss_text

        return logit, CL_loss

    @staticmethod
    def info_nce_loss(e1, e2, temperature=0.2):
        """
        e1, e2: [N, D]
        """
        N = e1.size(0)

        pos_sim = F.cosine_similarity(e1, e2, dim=1)  # [N]

        neg_idx = torch.randint(0, N - 1, (N,), device=e1.device)
        neg_idx = neg_idx + (neg_idx >= torch.arange(N, device=e1.device)).long()

        neg_sim = F.cosine_similarity(e1, e2[neg_idx], dim=1)  # [N]

        logits = torch.stack([pos_sim, neg_sim], dim=1) / temperature  # [N, 2]
        labels = torch.zeros(N, dtype=torch.long, device=e1.device)
        loss = F.cross_entropy(logits, labels)

        return loss