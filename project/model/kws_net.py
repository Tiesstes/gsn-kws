import torch
from torch import nn

from project.model.arch.bcresnet1 import BCResNet1

# TODO: finish model creation, embedding and fusion

# szkielet sieci
class KWSNet(nn.Module):

    def __init__(self, num_of_classes, num_of_speakers, speaker_emb_dim=32):
        """
        Klasa łącząca w sobie BC-ResNet-1 oraz embedding mówców

        :param num_of_classes: liczba klas
        :param num_of_speakers: liczba mówców w datasecie
        :param speaker_emb_dim: wielkość wetkora embedding'u dla mówców
        """
        super().__init__()

        self.backbone = BCResNet1(in_channels=1)

        assert self.backbone.output_channels == speaker_emb_dim # jak nie to AssertionError

        self.speaker_embedding = nn.Embedding(num_of_speakers, speaker_emb_dim)  # czy może kalsę do tego? to potem
        # kanałów 32, które wyszły po avg pooling
        self.classifier = nn.Linear(self.backbone.output_channels,
                                    num_of_classes)

    def forward(self, x, speaker_id):
        speaker_vector = self.speaker_embedding(speaker_id)  # [B, emb_dim]
        backbone_features = self.backbone(x)  # [B, output_channels_dim]

        # TODO: fusion with the speaker vector
        fused = 0

        logits = self.classifier(fused) # logit, to funkcja mapująca prawdopodobieństwo
                                        # z wartości -inf do +inf i ona prawdę nam powie
        return logits

