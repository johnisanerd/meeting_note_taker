#  brew install portaudio
# pip install pyaudio
# pip install pydub
# pip install wave

import pyaudio
import wave
from pydub import AudioSegment


chunk = 1024  # Buffer size
sample_format = pyaudio.paInt16  # 16-bit resolution
channels = 1  # Stereo
fs = 44100  # Sample rate
seconds = 5  # Duration of recording

p = pyaudio.PyAudio()

stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)

print("Recording...")

frames = []

for i in range(int(fs / chunk * seconds)):
    data = stream.read(chunk)
    frames.append(data)

print("Finished recording.")

stream.stop_stream()
stream.close()
p.terminate()

wave_filename = "output.wav"

wf = wave.open(wave_filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()

sound = AudioSegment.from_file("output.wav")
sound.export("output_file.mp3", format="mp3", bitrate="128k")
