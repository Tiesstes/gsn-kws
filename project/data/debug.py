import time

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS


from dataset import SpeechCommandsKWS
from project.data.dataset import SplitBuilder
from project.model.kws_net import KWSNet

import warnings
# bo głupi torchaudio krzyczy
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torchaudio"
)

"""
Tutaj jest skrypcik kontrolny. W zasadzie to samo co preprocess.py, tylko ma mniej dataset'ów
"""

# konfiguracja
BASE_PATH = Path(__file__).resolve().parent.parent # katalog /project
GSC_PATH = Path(__file__).resolve().parent # katalog /project/data
NOISE_PATH = Path(GSC_PATH) / "SpeechCommands" / "speech_commands_v0.02" / "_background_noise_"
TRAINED_WEIGHTS_PATH = Path(BASE_PATH) / "model" / "KWSNet_weights.pt"

# do treningu
EPOCHS = 22
BATCH_SIZE = 128
WORKERS = 0 # bo na moim komputerze Windows tego nie ogarnia (i ja też nie)
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0 # opcjonalnie
SCHEDULER_STEP_SIZE = 4 # zmiana lr pomaga z tego co sprawdzałam
SCHEDULER_GAMMA = 0.5


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA DEVICE:", torch.cuda.get_device_name(0))
    print("Torch version:", torch.__version__)
    print("")

    # podzbiory od razu:
    base_data = SPEECHCOMMANDS(root=GSC_PATH, subset=None, download=True)

    dataset_indexer = SplitBuilder(base_data)

    pretrainer_splits = dataset_indexer.build_pretrain_splits()

    train_ds = SpeechCommandsKWS(
        dataset=base_data,
        split_indices=pretrainer_splits["train"],
        allowed_speakers=pretrainer_splits["allowed_speakers"],
        speaker_id_map=dataset_indexer.speaker_id_map,  # <-- POPRAWKA
        noise_dir=NOISE_PATH,  # <-- POPRAWKA (masz już Path)
        duration=1.0,
        sample_rate=16000,
        number_of_mel_bands=40,
        silence_per_target=1.0,
        unknown_to_target_ratio=1.0,
        seed=1234,
    )


    def kind_counts(ds):
        counts = {"target": 0, "unknown": 0, "silence": 0}
        for kind, _ in ds.final_indices:
            counts[kind] += 1
        return counts


    def count_unique_speakers_no_silence_fast(ds):
        speakers = set()
        for kind, index in ds.final_indices:
            if kind == "silence":
                continue
            _, _, _, speaker, _ = ds.base_dataset[index]
            speakers.add(speaker)
        return len(speakers), speakers


    def count_unique_speaker_ids_no_silence_slow(ds):
        silence_label_id = ds.label_map["silence"]
        ids = set()
        for i in range(len(ds)):
            sample = ds[i]
            if int(sample["label"]) == silence_label_id:
                continue
            ids.add(int(sample["speaker_id"]))
        return len(ids)


    print("\n--- DATASET CHECKS ---")
    print("len(train_ds):", len(train_ds))

    counts = kind_counts(train_ds)
    print("target_count (split_indices):", len(train_ds.indices))
    print("silence_per_target:", 1.0)  # albo wypisz zmienną, którą przekazujesz
    print("computed silence_count:", train_ds.silence_count)
    print("Counts:", counts)

    if train_ds.silence_count == 0:
        print(
            "WARN: silence_count == 0 -> brak silence w final_indices (to może być OK, jeśli split_indices jest małe/puste).")

    # 2) speakerzy bez silence (FAST)
    n_spk_fast, speakers_fast = count_unique_speakers_no_silence_fast(train_ds)
    print("Unique speakers (no silence) FAST:", n_spk_fast)

    # 3) weryfikacja przez SLOW (po __getitem__)
    n_spk_slow = count_unique_speaker_ids_no_silence_slow(train_ds)
    print("Unique speaker_id (no silence) SLOW:", n_spk_slow)

    assert n_spk_fast == n_spk_slow, (
        f"Mismatch: FAST speakers={n_spk_fast}, SLOW speaker_ids={n_spk_slow}. "
        "To sugeruje problem z speaker_id_map (np. kolizje lub zły mapping)."
    )

    # 4) sanity: 'none' nie powinno wejść jeśli pomijamy silence
    assert "none" not in speakers_fast, "Znaleziono 'none' mimo pomijania silence."

    print("OK: speaker counting works\n")

    print("len(pretrainer_splits['train']):", len(pretrainer_splits["train"]))
    print("len(pretrainer_splits['val']):", len(pretrainer_splits["val"]))

    all_base_speakers = set()
    for i in range(len(base_data)):
        _, _, _, speaker, _ = base_data[i]
        all_base_speakers.add(speaker)

    print("SPEECHCOMMANDS unique speakers:", len(all_base_speakers))
    print("Example speakers from SPEECHCOMMANDS:", list(sorted(all_base_speakers))[:10])

    print("\n--- SplitBuilder speakers ---")
    print("all_speakers:", len(dataset_indexer.all_speakers))
    print("speaker_poor:", len(dataset_indexer.speaker_poor))
    print("speaker_rich:", len(dataset_indexer.speaker_rich))

    # sanity: zbiory powinny pokrywać all_speakers
    print("poor ∩ rich =", len(dataset_indexer.speaker_poor & dataset_indexer.speaker_rich))
    print("poor ∪ rich =", len(dataset_indexer.speaker_poor | dataset_indexer.speaker_rich))
    print("equals all_speakers =",
          (dataset_indexer.speaker_poor | dataset_indexer.speaker_rich) == dataset_indexer.all_speakers)

    # ile rich jest w ogóle w speaker_stats (powinno być True dla wszystkich rich)
    print("rich ⊆ speaker_stats.keys() =",
          dataset_indexer.speaker_rich.issubset(set(dataset_indexer.speaker_stats.keys())))

    missing_in_builder = all_base_speakers - dataset_indexer.all_speakers
    extra_in_builder = dataset_indexer.all_speakers - all_base_speakers

    print("\n--- Diff vs SPEECHCOMMANDS ---")
    print("speakers in base but not in builder:", len(missing_in_builder))
    print("speakers in builder but not in base:", len(extra_in_builder))

    # podgląd (zwykle oba powinny być 0)
    print("missing example:", list(sorted(missing_in_builder))[:10])
    print("extra example:", list(sorted(extra_in_builder))[:10])

    pre = dataset_indexer.build_pretrain_splits()
    pretrain_speaker_map = dataset_indexer.build_speaker_id_map_from_speakers(pre["allowed_speakers"])

    train_ds = SpeechCommandsKWS(
        dataset=base_data,
        split_indices=pre["train"],
        allowed_speakers=pre["allowed_speakers"],
        speaker_id_map=pretrain_speaker_map,
        noise_dir=NOISE_PATH,
        seed=1234,
    )

    model = KWSNet(num_of_classes=len(train_ds.label_map),
                   num_of_speakers=len(pretrain_speaker_map)).to(DEVICE)


