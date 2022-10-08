from utils import load_config

from typing import List
import os
import glob

# convert midi to notes
from music21 import converter, note, chord
import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, config_path: str, config={}):
        # if configuration manually changed as a dictionary
        if config:
            self.config = config
        else:
            self.config = load_config(config_path)

        self.inputs, self.targets = self.convert_all_files()
        self.notes2num = self.get_notes_mapping(self.inputs, self.targets)

    def __getitem__(self, idx):
        input_notes = [self.notes2num[note] for note in self.inputs[idx]]
        target_note = self.notes2num[self.targets[idx]]
        return torch.tensor(input_notes, dtype=torch.long), torch.tensor([target_note])

    def __len__(self):
        return len(self.inputs)

    def convert_midi_to_notes(self, midi_path: str) -> List[str]:
        """
        Converts midi to pitches and chords.
        """
        assert os.path.isfile(midi_path), 'The wrong path to the file.'

        notes = []
        midi = converter.parse(midi_path)

        for el in midi.flat.notes:
            if isinstance(el, note.Note):
                notes.append(str(el.pitch))
            elif isinstance(el, chord.Chord):
                notes.append('.'.join(map(str, el.normalOrder)))
        return notes

    def get_notes_mapping(self, notes_in, notes_out):
        notes = sum(notes_in, [])
        notes.extend(notes_out)
        notes_mapping = dict((note, number) for number, note in enumerate(set(notes)))
        return notes_mapping

    def convert_all_files(self) -> (List[List], List[List]):
        """
        Convert all midi files to notes in datasets.
        """
        inputs, targets = [], []
        for data_dir in self.config['datasets_paths']:
            for midi_path in glob.glob(data_dir + '/*.mid'):
                midi_notes = self.convert_midi_to_notes(midi_path)
                samples_in, samples_out = self.get_simple_slicing(midi_notes)
                inputs.append(samples_in)
                targets.append(samples_out)
        inputs = sum(inputs, [])
        targets = sum(inputs, [])
        return inputs, targets

    def get_simple_slicing(self, notes_sample: list):
        """
        Sample notes with pairs while target is the last note.
        :param notes_sample: one
        """
        samples_in, sample_out = [], []
        seq_length = self.config['sequence_length']
        sample_len = len(notes_sample)
        for i in range(0, sample_len - seq_length - 1):
            samples_in.append(notes_sample[i:i + seq_length])
            sample_out.extend(notes_sample[i + seq_length])
        return samples_in, sample_out


def collate_fn(batch):
    input_notes, targets = zip(*batch)
    input_notes = torch.stack(input_notes, dim=0)
    targets = torch.Tensor(targets)
    return {'input_notes': input_notes, 'targets': targets}
