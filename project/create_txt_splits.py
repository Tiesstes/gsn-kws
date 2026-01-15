from typing import Iterable
from pathlib import Path

SUBFOLDERS = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'unknown', 'up', 'yes']

SPEAKERS = ['Bedoes', 'charlie', 'ckm', 'cowy', 'Hala', 'KS', 'morrison', 'psp', 'SeattleSeahawks', 'soundjul', 'Weronika']   # TODO: append w others

OUTPUT_DIR = Path('.') / 'project'

TRAIN_FILE = OUTPUT_DIR / 'train_list.txt'
VAL_FILE = OUTPUT_DIR / 'val_list.txt'
TEST_FILE = OUTPUT_DIR / 'test_list.txt'

def main():
    # Make sure the output directory exists. Otherwise, make it
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Create a train_file list
    print("Creating train list...")
    write_lines(SUBFOLDERS, SPEAKERS, [0,1,2,3], TRAIN_FILE)
    print("Train list created and saved successfully!")
    # Create a val_file list
    print("Creating validation list...")
    write_lines(SUBFOLDERS, SPEAKERS, [4], VAL_FILE)
    print("Validation list created and saved successfully!")
    # Create a test_file list
    print("Creating test list...")
    write_lines(SUBFOLDERS, SPEAKERS, [5], TEST_FILE)
    print("Test list created and saved successfully!")



def create_lines(subfolders: list, speakers: list, indxs: list) -> Iterable[str]:
    for subfolder in subfolders:
        for speaker in speakers:
            for idx in indxs:
                yield f'{subfolder}/{speaker}_{idx}.wav\n'

def write_lines(subfolders: list, speakers: list, indxs: list, file_name: Path) -> None:
    lines = create_lines(subfolders, speakers, indxs)
    with open(file_name, 'w') as writer:
        #for line in lines:
        writer.writelines(lines)


if __name__ == "__main__":
    main()