import re
import glob
import pandas as pd
from typing import List


def get_songs(data_path: str):
    with open(data_path, encoding='utf-8') as f:
        author_text = f.read().splitlines()
        songs_starts = [s_idx for s_idx, song in enumerate(author_text) if 'Lyrics' in song]
        songs_starts.append(len(author_text))

        s_idx = songs_starts[0]
        songs = []
        songs_names = []
        for end_idx in songs_starts[1:]:
            songs_names.append(get_song_name(author_text[s_idx]))
            song = author_text[s_idx + 1: end_idx]
            songs.append('\n'.join(song))

            s_idx = end_idx
    return songs, songs_names


def get_song_name(song_name_line):
    return re.sub("[\(\[].*?[\)\]]", "", song_name_line).strip()


def create_dataset_gpt(data_paths: List) -> List:
    data = []
    for d_path in data_paths:
        author = d_path.split('/')[-1].split('_')[0]
        songs, songs_names = get_songs(d_path)

        for song, song_name in zip(songs, songs_names):
            data.append((author, song_name, song))
    return data


def get_custom_tokenize(sample, tokenizer):
    author_name = sample['artist']
    song_name = sample['song_name']
    lyrics = sample['lyrics']

    encoded = tokenizer(
        f"[s:artist] {author_name} [e:artist] [s:song_name] {song_name} [e:song_name] [s:lyrics] {lyrics} [e:lyrics]",
        max_length=512, truncation=True, padding='max_length', return_token_type_ids=True)
    encoded["labels"] = encoded['input_ids'].copy()
    return encoded


def create_dataframe():
    data_paths = glob.glob("/kaggle/input/russian-lyrics/new/*.txt")
    raw_data = create_dataset_gpt(data_paths)

    dataframe = pd.DataFrame({
        'artist': [sample[0] for sample in raw_data],
        'song_name': [sample[1] for sample in raw_data],
        'lyrics': [sample[2] for sample in raw_data],
    })
    return dataframe
