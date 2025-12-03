import torch
from torch import nn


# szkielet sieci
class KWSNet(nn.Module):

    def __init__(self, num_of_classes, num_of_speakers, speaker_emb_dim=16):
        super().__init__()
        self.speaker_embedding = nn.Embedding(num_of_speakers, speaker_emb_dim) # czy może kalsę do tego? to potem
        self.backbone = BCResNet1(in_channels=1) # zdefiniować backbone
        self.classifier = nn.Linear(self.backbone.out_dim + speaker_emb_dim,
                                    num_of_classes)

    def forward(self, x, speaker_id):
        speaker_vector = self.speaker_embedding(speaker_id)  # [B, emb_dim]
        feat = self.backbone(x)  # [B, out_dim]
        fused = torch.cat([feat, speaker_vector], dim=1)
        logits = self.classifier(fused) # logit, to funkcja mapująca prawdopodobieństwo z wartości -inf do +inf
        return logits