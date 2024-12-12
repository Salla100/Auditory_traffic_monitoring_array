from scipy.io import wavfile
import tables
import os

# Hardcode the path to the WAV file
wav_file_path = r"C:\Users\sven1\Downloads\engine-sounds\engine-sounds\car\18.wav"

# Read data from the WAV file
fs, data = wavfile.read(wav_file_path)

# Generate output folder and name dynamically
folder, file_name = os.path.split(wav_file_path)
name = os.path.splitext(file_name)[0] + ".h5"
output_path = os.path.join(folder, name)

# Save to Acoular H5 format
acoularh5 = tables.open_file(output_path, mode="w", title=name)
acoularh5.create_earray(
    '/',
    'time_data',
    atom=None,
    title='',
    filters=None,
    expectedrows=100000,
    byteorder=None,
    createparents=False,
    obj=data
)
acoularh5.set_node_attr('/time_data', 'sample_freq', fs)
acoularh5.close()

print(f"HDF5 file created at: {output_path}")
