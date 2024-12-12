import os
import numpy as np
import acoular
import soundfile as sf
from scipy.signal import correlate, resample
import time

def load_audio_from_wav(wav_file):
    """
    Loads audio data from a WAV file using Acoular.

    Args:
        wav_file: Path to the input WAV file.

    Returns:
        acoular.TimeSamples: Acoular TimeSamples object containing the audio data.
    """
    data, samplerate = sf.read(wav_file)

    # Ensure data is 2D even for single-channel audio
    if data.ndim == 1:
        data = data[:, np.newaxis]  # Add a channel dimension

    ts = acoular.TimeSamples(data=data, sample_freq=samplerate)
    return ts

def resample_acoular_ts(ts, target_fs):
    """
    Resamples an Acoular TimeSamples object to a target sampling rate.

    Args:
        ts: Acoular TimeSamples object.
        target_fs: Target sampling rate in Hz.

    Returns:
        acoular.TimeSamples: Resampled Acoular TimeSamples object.
    """
    import scipy.signal  # Import scipy for resampling

    data = ts.data
    fs = ts.sample_freq

    if fs != target_fs:
        num_samples = round(len(data) * float(target_fs) / fs)
        data_resampled = scipy.signal.resample(data, num_samples)
        ts_resampled = acoular.TimeSamples(data=data_resampled, sample_freq=target_fs)
        return ts_resampled
    else:
        return ts

def match_car_sound(reference_ts, test_ts, threshold=0.7):
    """
    Matches car sounds by correlating their time-domain signals.

    Args:
        reference_ts: Acoular TimeSamples object for the reference car sound.
        test_ts: Acoular TimeSamples object for the test car sound.
        threshold: Correlation threshold for a match.

    Returns:
        bool: True if the correlation exceeds the threshold, False otherwise.
    """
    ref_data = reference_ts.data
    test_data = test_ts.data

    min_length = min(len(ref_data), len(test_data))
    ref_data = ref_data[:min_length]
    test_data = test_data[:min_length]

    correlation = correlate(test_data, ref_data, mode='valid')
    correlation /= np.max(correlation)

    return np.max(correlation) > threshold

def beamform_and_detect_direction(test_ts, mic_array_geometry, grid, ref_freq=8000):
    """
    Performs beamforming and detects the direction of the sound source.

    Args:
        test_ts: Acoular TimeSamples object for the test sound.
        mic_array_geometry: Path to the microphone array geometry file.
        grid: Acoular Grid object defining the search space for the beamformer.
        ref_freq: Reference frequency for beamforming.

    Returns:
        str: Direction of the sound source ('left-to-right' or 'right-to-left').
    """
    ps = acoular.PowerSpectra(source=test_ts, block_size=128, window='Hanning')
    mg = acoular.MicGeom(from_file=mic_array_geometry)
    st = acoular.SteeringVector(grid=grid, mics=mg)
    bb = acoular.BeamformerBase(freq_data=ps, steer=st)
    pm = bb.synthetic(ref_freq, 3)
    Lm = acoular.L_p(pm)
    max_indices = np.unravel_index(np.argmax(Lm, axis=None), Lm.shape)
    x_direction = max_indices[1] - Lm.shape[1] // 2
    return 'left-to-right' if x_direction > 0 else 'right-to-left'

def main(stationary_car_wav_file, input_wav_file, mic_array_geometry):
    """
    Main function to load audio, match car sounds, and perform beamforming.

    Args:
        stationary_car_wav_file: Path to the stationary car WAV file.
        input_wav_file: Path to the input WAV file.
        mic_array_geometry: Path to the microphone array geometry file.
    """
    cars_detected = 0
    detection_times = []

    reference_ts = load_audio_from_wav(stationary_car_wav_file)
    test_ts = load_audio_from_wav(input_wav_file)

    # Ensure same sampling rate
    test_ts = resample_acoular_ts(test_ts, reference_ts.sample_freq)

    if match_car_sound(reference_ts, test_ts):
        rg = acoular.RectGrid(x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, z=0.3, increment=0.05)
        direction = beamform_and_detect_direction(test_ts, mic_array_geometry, rg)
        print(f"Car detected in {os.path.basename(input_wav_file)}, moving {direction}.")
        cars_detected += 1
        detection_times.append(time.time()) 

    print(f"Total cars detected: {cars_detected}")
    print(f"Detection times: {detection_times}")

# Example Execution
stationary_car_wav_file = r"C:\Users\sven1\Downloads\engine-sounds\engine-sounds\car\18.wav" 
input_wav_file = r"C:\Users\sven1\Downloads\acoular_test\event-0002_speed-057.wav" 
mic_array_geometry = "4_linear.xml"
main(stationary_car_wav_file, input_wav_file, mic_array_geometry)