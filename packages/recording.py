from packages.common_packages import pyaudio, os, wave
from packages.common_packages import SAMPLE_RATE

CHANNEL = 1
DURATION = 3
CHUNK = 1024


def main():
    record_audio_and_save()
    # Record audio for 3 seconds
    audio_file_path = record_audio(duration=3)


def record_audio_and_save(filename='output.wav',
                          duration=DURATION,
                          sample_rate=SAMPLE_RATE, channels=CHANNEL):
    chunk = CHUNK  
    sample_format = pyaudio.paInt16
    p = pyaudio.PyAudio()

    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=sample_rate,
                    frames_per_buffer=chunk,
                    input=True)
    # Store data in chunks for duration seconds
    frames = []
    for i in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    print('Finished recording')

    # # Save the recorded data as a WAV file
    # wf = wave.open(filename, 'wb')
    # wf.setnchannels(channels)
    # wf.setsampwidth(p.get_sample_size(sample_format))
    # wf.setframerate(sample_rate)
    # wf.writeframes(b''.join(frames))
    # wf.close()

    # # Print the audio properties
    # print_audio_properties(filename)


# Part 11
def record_audio(duration, fs=SAMPLE_RATE, channels=CHANNEL, format=pyaudio.paInt16):
    audio = pyaudio.PyAudio()

    stream = audio.open(format=format, channels=channels, rate=fs, input=True, frames_per_buffer=CHUNK)

    print("Recording started...")
    frames = []
    for i in range(0, int(fs / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Find the highest existing file number
    file_number = 0
    while os.path.exists("recording/my_voice%d.wav" % file_number):
        file_number += 1

    # Save the recorded audio to a new WAV file with an incremented name
    WAVE_OUTPUT_FILENAME = "recording/my_voice%d.wav" % file_number
    wave_file = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wave_file.setnchannels(channels)
    wave_file.setsampwidth(audio.get_sample_size(format))
    wave_file.setframerate(fs)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()

    return WAVE_OUTPUT_FILENAME


if __name__ == '__main__':
    main()
