import pyaudio
import wave


def record_audio(filename,duration, sample_rate = 16000, channels=1, format=pyaudio.paInt16):
    p = pyaudio.PyAudio()

    stream = p.open(format=format, channels=channels, rate=sample_rate,input= True, frames_per_buffer=1024)

    print("Recording...")

    frames = []
    for i in range(int(sample_rate/1024*duration)):
        data = stream.read(1024)
        frames.append(data)

    print("Recording fininshed")

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

if __name__ == "__main__":
    output_filename = "recorded_audio.wav"
    recording_duration = 5

    record_audio(output_filename, recording_duration)
    

