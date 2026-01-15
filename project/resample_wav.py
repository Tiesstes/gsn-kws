import os
from pathlib import Path
import librosa
import soundfile as sf

# Path to the directory w OG data
CUSTOM_GSCv2_PATH = Path('.') / 'project'/ 'custom_speech_commands_v2'

# Path to the output directory
OUTPUT_BASE_PATH = Path('.') / 'project' / 'custom_speech_commands_v2_resampled'
OUTPUT_BASE_PATH.mkdir(parents=True, exist_ok=True)

# Target sample rate
TARGET_SAMPLE_RATE = 16000

print(f"Processing audio files from: {CUSTOM_GSCv2_PATH}")
print(f"Saving processed files to: {OUTPUT_BASE_PATH}")

# For each subfolder in CUSTOM_GSCv2_PATH
for subfolder in CUSTOM_GSCv2_PATH.iterdir():
    # Check if subfolder is the directory
    if subfolder.is_dir():
        print(f"\nProcessing subfolder: {subfolder.name}")

        # Create corresponding output subfolder
        output_subfolder_path = OUTPUT_BASE_PATH / subfolder.name
        output_subfolder_path.mkdir(parents=True, exist_ok=True)

        # For each .wav file in the current subfolder
        for wav_file in subfolder.glob('*.wav'):
            try:
                # Load the audio file
                audio, sr = librosa.load(wav_file, sr=None) # sr=None, aby wczytać audio z OG sr

                # Resample to target sample rate if necessary
                if sr != TARGET_SAMPLE_RATE:
                    print(f"  Resampling {wav_file.name} from {sr} Hz to {TARGET_SAMPLE_RATE} Hz...")
                    audio_resampled = librosa.resample(y=audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
                else:
                    audio_resampled = audio
                    print(f"  {wav_file.name} already at {TARGET_SAMPLE_RATE} Hz, no resampling needed.")

                # Define output file path
                output_wav_path = output_subfolder_path / wav_file.name

                # Save the resampled audio
                sf.write(output_wav_path, audio_resampled, TARGET_SAMPLE_RATE)
                print(f"  Successfully processed and saved: {output_wav_path.name}")

            except Exception as e:
                print(f"  Error processing {wav_file.name}: {e}")

print("\nAudio processing complete.")