import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Constants for Manchester encoding
SAMPLE_RATE = 44100  # Sample rate in Hz
BIT_RATE = 1200  # Number of bits per second for ZX Spectrum
FREQ_HIGH = 2400  # Frequency representing a transition from low to high
FREQ_LOW = 1200   # Frequency representing a transition from high to low
PILOT_TONE_DURATION = 0.2  # Reduced pilot tone duration in seconds
ZOOM_SAMPLES = 1000  # Number of samples to zoom into for graph visualization

# ----- Writing Process -----

def generate_pilot_tone(sample_rate, duration):
    """
    Generate a pilot tone to be played before the data, used for synchronization.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    pilot_tone = np.sin(2 * np.pi * FREQ_HIGH * t)
    return pilot_tone

def manchester_encode_bit(bit, sample_rate, bit_duration):
    """
    Encode a single bit using Manchester encoding.
    Bit 0 -> High-to-Low transition
    Bit 1 -> Low-to-High transition
    """
    t = np.linspace(0, bit_duration, int(sample_rate * bit_duration), endpoint=False)
    
    if bit == 0:
        # High-to-Low transition
        return np.concatenate((np.ones(int(len(t)/2)), -np.ones(int(len(t)/2))))
    else:
        # Low-to-High transition
        return np.concatenate((-np.ones(int(len(t)/2)), np.ones(int(len(t)/2))))

def manchester_encode_byte(byte, sample_rate, bit_duration):
    """
    Encode a single byte using Manchester encoding.
    """
    bits = [(byte >> i) & 1 for i in range(7, -1, -1)]
    encoded_bits = np.concatenate([manchester_encode_bit(bit, sample_rate, bit_duration) for bit in bits])
    return encoded_bits

def save_file_to_wav(filename, input_filename):
    """
    Read a binary file, encode its content using Manchester encoding, and save as a WAV file.
    """
    # Read binary file
    with open(input_filename, 'rb') as f:
        file_bytes = f.read()

    # Generate pilot tone
    pilot_tone = generate_pilot_tone(SAMPLE_RATE, PILOT_TONE_DURATION)

    # Generate waveform using Manchester encoding for each byte in the binary file
    waveform = np.concatenate([manchester_encode_byte(byte, SAMPLE_RATE, 1.0 / BIT_RATE) for byte in file_bytes])

    # Normalize waveform to the range [-1, 1]
    waveform = waveform / np.max(np.abs(waveform))

    # Combine pilot tone with encoded waveform
    final_waveform = np.concatenate((pilot_tone, waveform))

    # Save the waveform as a WAV file
    wavfile.write(filename, SAMPLE_RATE, final_waveform.astype(np.float32))
    print(f'File {input_filename} saved to {filename} as a WAV')

    # Plot the pilot tone (Zoomed-in portion)
    plt.figure(figsize=(10, 4))
    plt.plot(final_waveform[:ZOOM_SAMPLES])  # Show only a small portion of the pilot tone
    plt.title(f"Pilot Tone (First {ZOOM_SAMPLES} Samples)")
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

    # Plot the data waveform (Zoomed-in portion, starting after pilot tone)
    plt.figure(figsize=(10, 4))
    plt.plot(final_waveform[int(PILOT_TONE_DURATION * SAMPLE_RATE):int(PILOT_TONE_DURATION * SAMPLE_RATE) + ZOOM_SAMPLES])
    plt.title(f"Encoded Data Waveform (First {ZOOM_SAMPLES} Samples After Pilot Tone)")
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

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

def manchester_decode(data, sample_rate, bit_duration):
    """
    Decode a waveform using Manchester encoding.
    """
    samples_per_bit = int(sample_rate * bit_duration)
    decoded_bits = []

    for i in range(0, len(data), samples_per_bit):
        bit_chunk = data[i:i+samples_per_bit]
        if len(bit_chunk) < samples_per_bit:
            continue  # Ignore incomplete chunks at the end

        # Check if we have a Low-to-High transition (bit 1) or High-to-Low transition (bit 0)
        if bit_chunk[:samples_per_bit//2].mean() > 0 and bit_chunk[samples_per_bit//2:].mean() < 0:
            decoded_bits.append(0)
        elif bit_chunk[:samples_per_bit//2].mean() < 0 and bit_chunk[samples_per_bit//2:].mean() > 0:
            decoded_bits.append(1)

    return decoded_bits

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

def load_wav_to_file(wav_filename, output_filename):
    """
    Load and decode a WAV file back into its original binary format and save as a new file.
    """
    # Load the WAV file
    sample_rate, data = load_wav_file(wav_filename)

    # Decode the waveform to extract the bits
    bit_duration = 1.0 / BIT_RATE
    bits = manchester_decode(data[int(PILOT_TONE_DURATION * sample_rate):], sample_rate, bit_duration)

    # Convert the bits to bytes
    file_bytes = bits_to_bytes(bits)

    # Write the decoded bytes to a new file
    with open(output_filename, 'wb') as f:
        f.write(bytearray(file_bytes))

    print(f'Decoded WAV file saved as {output_filename}')

    # Plot the loaded waveform (Zoomed-in portion for debugging)
    plt.figure(figsize=(10, 4))
    plt.plot(data[:ZOOM_SAMPLES])  # Show only a small portion of the loaded waveform
    plt.title(f"Loaded WAV Signal (First {ZOOM_SAMPLES} Samples)")
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

    # Plot the decoded data waveform (Zoomed-in portion for debugging)
    plt.figure(figsize=(10, 4))
    plt.plot(data[int(PILOT_TONE_DURATION * SAMPLE_RATE):int(PILOT_TONE_DURATION * SAMPLE_RATE) + ZOOM_SAMPLES])
    plt.title(f"Decoded Data Waveform (First {ZOOM_SAMPLES} Samples After Pilot Tone)")
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

# ----- Main Script -----

# File paths
input_filename = 'bess.jpg'
wav_filename = 'bess_test_manchester.wav'
output_filename = 'bess_out.jpg'

# Step 1: Convert the input binary file (test.jpg) to a WAV file
save_file_to_wav(wav_filename, input_filename)

# Step 2: Convert the WAV file back to a binary file (output.jpg)
load_wav_to_file(wav_filename, output_filename)