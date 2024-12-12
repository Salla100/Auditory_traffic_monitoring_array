import numpy as np
import h5py
import soundfile as sf

def flac_to_h5_and_wav(flac_file, h5_file, wav_file):
  """
  Converts a FLAC file to both an HDF5 file and a WAV file, preserving channel information.

  Args:
    flac_file: Path to the input FLAC file.
    h5_file: Path to the output HDF5 file.
    wav_file: Path to the output WAV file.
  """

  # Load the FLAC file
  data, samplerate = sf.read(flac_file)

  # Handle multi-channel audio
  if data.ndim == 1:
    num_channels = 1
  else:
    num_channels = data.shape[1]

  # Print the number of channels
  print(f"Number of channels: {num_channels}")

  # Create the HDF5 file
  with h5py.File(h5_file, 'w') as f:
    # Create a dataset to store the audio data
    dset = f.create_dataset('audio', data=data, dtype=data.dtype)
    # Store the sampling rate as an attribute
    dset.attrs['samplerate'] = samplerate
    # Store the number of channels as an attribute
    dset.attrs['num_channels'] = num_channels

  # Create the WAV file
  sf.write(wav_file, data, samplerate)

if __name__ == "__main__":
  flac_file = r"C:\Users\sven1\Downloads\simulation\simulation\loc1\car\left\event-0002_speed-057.flac"
  h5_file = "event-0002_speed-057.h5"
  wav_file = "event-0002_speed-057.wav"
  flac_to_h5_and_wav(flac_file, h5_file, wav_file)
