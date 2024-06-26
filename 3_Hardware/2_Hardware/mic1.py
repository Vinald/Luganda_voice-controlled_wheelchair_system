import pyaudio
import wave
import os

def record_audio(filename='recorded_audio.wav', duration=3, fs=16000, channels=1, format=pyaudio.paInt16):
    CHUNK = 1024
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

    # Save the audio file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    return filename


# Test the function
record_audio()
