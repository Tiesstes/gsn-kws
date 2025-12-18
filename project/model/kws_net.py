import torch
from torch import nn

from project.model.arch.bcresnet1 import BCResNet1


# szkielet sieci
class KWSNet(nn.Module):

    def __init__(self, num_of_classes, num_of_speakers):
        """
        Klasa łącząca w sobie BC-ResNet-1 oraz embedding mówców

        :param num_of_classes: liczba klas w datasecie
        :param num_of_speakers: liczba mówców w datasecie -> ile ma być wektorów embeddingu
        Atrybut:
        *speaker_emb_dim* to wielkość wektora embedding'u dla mówców -> wynosi 32, bo 32 mapy cech BC-ResNet-1
        """
        super().__init__()

        self.speaker_emb_dim = 32
        self.backbone = BCResNet1(in_channels=1)

        assert self.backbone.output_channels == self.speaker_emb_dim # jak nie to AssertionError

        self.speaker_embedding = nn.Embedding(num_of_speakers, self.speaker_emb_dim)

        # kanałów 32, które wyszły po avg pooling (w sumie to po ostatniej konwolucji)
        self.classifier = nn.Linear(self.backbone.output_channels,
                                    num_of_classes)

    def forward(self, x, speaker_id) -> torch.Tensor:
        speaker_vector = self.speaker_embedding(speaker_id)  # [B, emb_dim]
        backbone_features = self.backbone(x)  # [B, output_channels_dim]

        fused = backbone_features + speaker_vector # można też z tym expand_as

        logits = self.classifier(fused) # logit, to funkcja mapująca prawdopodobieństwo [z 32 map cech robimy 12 klas]
                                        # z wartości -inf do +inf i ona prawdę nam powie
        return logits


