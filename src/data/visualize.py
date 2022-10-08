from visual_midi import Plotter
from visual_midi import Preset
from pretty_midi import PrettyMIDI


def plot_midi_file(midi_path: str) -> None:
  pm = PrettyMIDI(midi_path)
  plotter = Plotter()
  plotter.show(pm, midi_path.replace('mid', 'html'))