import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

# Constants for ZX Spectrum WAV format
SAMPLE_RATE = 44100  # Sample rate in Hz
FREQ_0 = 1200  # Frequency for bit 0 (1200 Hz)
FREQ_1 = 2400  # Frequency for bit 1 (2400 Hz)
PILOT_FREQ = 2400  # Frequency for pilot tone (2400 Hz)
PILOT_LENGTH = 8063  # Number of pulses in the pilot tone for BASIC files
SYNC_1_LENGTH = 667  # Length of the first sync pulse in microseconds
SYNC_2_LENGTH = 735  # Length of the second sync pulse in microseconds
BIT_0_LENGTH = 2168  # Length of bit 0 in microseconds
BIT_1_LENGTH = 667   # Length of bit 1 in microseconds
THRESHOLD_FREQ = 1800  # Threshold for frequency to detect bits

# ----- Writing Process -----

def generate_square_wave(freq, duration, sample_rate):
    """
    Generate a square wave of given frequency and duration.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = 0.5 * (1 + np.sign(np.sin(2 * np.pi * freq * t)))
    return wave

def generate_pilot_tone(length, sample_rate):
    """
    Generate the pilot tone, a sequence of 1s at 2400 Hz.
    """
    tone_duration = length * (SYNC_2_LENGTH / 1e6)  # Pilot tone total duration
    return generate_square_wave(PILOT_FREQ, tone_duration, sample_rate)

def generate_sync_pulses(sample_rate):
    """
    Generate the sync pulses following the pilot tone.
    """
    sync_pulse_1 = generate_square_wave(PILOT_FREQ, SYNC_1_LENGTH / 1e6, sample_rate)
    sync_pulse_2 = generate_square_wave(PILOT_FREQ, SYNC_2_LENGTH / 1e6, sample_rate)
    return np.concatenate((sync_pulse_1, sync_pulse_2))

def byte_to_bits(byte):
    """
    Convert a byte to 8 bits.
    """
    return [(byte >> i) & 1 for i in range(7, -1, -1)]

def generate_bit_wave(bit, sample_rate):
    """
    Generate the waveform for a single bit (0 or 1).
    """
    if bit == 0:
        return generate_square_wave(FREQ_0, BIT_0_LENGTH / 1e6, sample_rate)
    else:
        return generate_square_wave(FREQ_1, BIT_1_LENGTH / 1e6, sample_rate)

def generate_data_block(program_bytes, sample_rate):
    """
    Generate the waveform for the entire data block (BASIC program).
    """
    waveform = np.array([])

    # Convert each byte to bits and generate waveforms
    for byte in program_bytes:
        bits = byte_to_bits(byte)
        for bit in bits:
            bit_wave = generate_bit_wave(bit, sample_rate)
            waveform = np.concatenate((waveform, bit_wave))

    return waveform

def save_basic_to_wav(filename, basic_program):
    """
    Save a BASIC program as a ZX Spectrum WAV file.
    """
    # Convert the BASIC program to bytes
    program_bytes = basic_program.encode('ascii')

    # Log encoded bytes for comparison
    print(f"Encoded Bytes (written to WAV): {list(program_bytes)}")

    # Generate the pilot tone (sequence of 1s)
    pilot_tone = generate_pilot_tone(PILOT_LENGTH, SAMPLE_RATE)

    # Generate the sync pulses after the pilot tone
    sync_pulses = generate_sync_pulses(SAMPLE_RATE)

    # Generate the data block (actual BASIC program)
    data_block = generate_data_block(program_bytes, SAMPLE_RATE)

    # Combine the pilot tone, sync pulses, and data block into a single waveform
    waveform = np.concatenate((pilot_tone, sync_pulses, data_block))

    # Adjust signal amplitude (range [-1, 1]) to [-0.5, 0.5] to reduce distortion
    waveform = waveform * 0.5

    # Save the waveform as a WAV file
    wavfile.write(filename, SAMPLE_RATE, waveform.astype(np.float32))
    print(f'BASIC program saved to {filename}')

# ----- Reading Process -----

def load_wav_file(filename):
    """
    Load a WAV file and return the sample rate and audio data.
    """
    sample_rate, data = wavfile.read(filename)
    
    # If stereo, take only one channel
    if len(data.shape) == 2:
        data = data[:, 0]
    
    return sample_rate, data

def bandpass_filter(data, lowcut=1000, highcut=3000, sample_rate=SAMPLE_RATE, order=5):
    """
    Apply a stricter bandpass filter to the data to isolate frequencies between 1000 Hz and 3000 Hz.
    This isolates the expected 1200 Hz (bit 0) and 2400 Hz (bit 1).
    """
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def zero_crossing_detection(data):
    """
    Detect zero crossings in the waveform for frequency detection, with added debugging.
    """
    zero_crossings = np.where(np.diff(np.sign(data)))[0]
    crossing_intervals = np.diff(zero_crossings)

    # Log zero-crossing intervals and expected intervals
    expected_interval_0 = SAMPLE_RATE // FREQ_0  # Expected interval for 1200 Hz
    expected_interval_1 = SAMPLE_RATE // FREQ_1  # Expected interval for 2400 Hz
    print(f"Expected interval for 1200 Hz: {expected_interval_0}")
    print(f"Expected interval for 2400 Hz: {expected_interval_1}")
    print(f"Crossing intervals (first 20): {crossing_intervals[:20]}")

    # Calculate frequencies based on zero-crossing intervals
    frequencies = SAMPLE_RATE / crossing_intervals

    # Filter out extremely high frequencies (due to noise)
    frequencies = frequencies[frequencies < 5000]

    return frequencies

def smooth_frequencies(frequencies):
    """
    Apply simple smoothing to frequencies to reduce noise.
    """
    smoothed = np.convolve(frequencies, np.ones(5) / 5, mode='same')  # Moving average
    return smoothed

def detect_pilot_tone(bits, threshold=50):
    """
    Detect a long sequence of 1 bits as the pilot tone. 
    After the pilot tone ends, the actual data starts.
    """
    count_ones = 0
    for i, bit in enumerate(bits):
        if bit == 1:
            count_ones += 1
        else:
            count_ones = 0
        if count_ones > threshold:
            return i  # Return the index where the pilot tone ends
    return 0  # If no pilot tone is detected, start from the beginning

def decode_waveform(data, sample_rate):
    """
    Decode the waveform by detecting frequencies corresponding to 0 and 1 bits.
    """
    # Apply a bandpass filter to focus on the frequency range of interest
    filtered_data = bandpass_filter(data, 1000, 3000, sample_rate)

    # Visualize the filtered waveform for debugging
    plt.figure(figsize=(10, 4))
    plt.plot(filtered_data[:5000])  # Plot a portion of the waveform
    plt.title("Filtered Signal")
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

    # Find zero crossings and calculate the frequencies
    frequencies = zero_crossing_detection(filtered_data)

    # Apply smoothing to reduce noise and variability
    smoothed_frequencies = smooth_frequencies(frequencies)

    # Debug: Visualize the first 100 detected frequencies
    plt.figure(figsize=(10, 4))
    plt.plot(smoothed_frequencies[:100], 'o-')
    plt.title("Smoothed Frequencies (First 100)")
    plt.xlabel("Index")
    plt.ylabel("Frequency (Hz)")
    plt.grid(True)
    plt.show()

    bits = []
    
    # Decode frequencies into bits based on the threshold
    for freq in smoothed_frequencies:
        if freq < THRESHOLD_FREQ:
            bits.append(0)  # 1200 Hz for bit 0
        else:
            bits.append(1)  # 2400 Hz for bit 1

    # Detect and skip the pilot tone (long sequence of 1s)
    start_index = detect_pilot_tone(bits)
    return bits[start_index:]  # Return bits after the pilot tone

def bits_to_bytes(bits):
    """
    Convert a list of bits to a list of bytes.
    """
    byte_list = []
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits):
                byte |= (bits[i + j] << (7 - j))
        byte_list.append(byte)
    return byte_list

def decode_bytes_to_program(byte_list):
    """
    Convert a list of bytes back into an ASCII-encoded BASIC program.
    """
    try:
        return ''.join(chr(byte) for byte in byte_list if 32 <= byte < 128)  # Printable ASCII
    except:
        return "Error decoding bytes."

def load_basic_from_wav(filename):
    """
    Load and decode a BASIC program from a WAV file.
    """
    # Load the WAV file
    sample_rate, data = load_wav_file(filename)

    # Decode the waveform to extract the bits
    bits = decode_waveform(data, sample_rate)

    # Debugging: Print first 100 detected bits
    print(f'Detected Bits (first 100): {bits[:100]}')

    # Convert the bits to bytes
    byte_data = bits_to_bytes(bits)

    # Log the decoded bytes for comparison
    print(f"Decoded Bytes (from WAV): {byte_data}")

    # Convert the bytes to the original BASIC program
    basic_program = decode_bytes_to_program(byte_data)

    return basic_program

# ----- Main Script -----

# Example BASIC program (simple)
basic_program = """10 PRINT "HELLO WORLD"
20 GOTO 10"""

# Step 1: Save the BASIC program to a WAV file
wav_filename = 'basic_program.wav'
save_basic_to_wav(wav_filename, basic_program)

# Step 2: Reload the BASIC program from the WAV file and display it
reloaded_program = load_basic_from_wav(wav_filename)
print("Reloaded BASIC Program:")
print(reloaded_program)