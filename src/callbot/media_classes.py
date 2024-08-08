from utils import *
from aiortc import MediaStreamTrack
import asyncio
from aiortc.contrib.media import MediaRecorder
import av
import numpy as np
import wave
import av
import time
from fractions import Fraction

class BytesIOAudioStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, audio_buffer):
        super().__init__()  # Initialize the base MediaStreamTrack class
        self.audio_buffer = audio_buffer
        self.audio_buffer.seek(0)
        self.wave_file = wave.open(self.audio_buffer, 'rb')
        self.sample_rate = self.wave_file.getframerate()
        self.channels = self.wave_file.getnchannels()
        self.samples_per_frame = 960  # This depends on your desired frame duration (e.g., 20ms for 48kHz audio)
        self.frame_duration = self.samples_per_frame / self.sample_rate
        self.pts = 0  # Initialize PTS counter

    async def recv(self):
        frames = self.wave_file.readframes(self.samples_per_frame)
        data=np.frombuffer(frames, dtype=np.int16)
        # print(data)
        if len(data) == 0:
            print('empty')

        frame = av.AudioFrame.from_ndarray(
            data.reshape(1,-1),
            format="s16",
            layout="stereo" if self.channels == 2 else "mono"
        )
        frame.sample_rate = self.sample_rate
        frame.time_base = Fraction(1, self.sample_rate)  # Set the time base for accurate timestamping
        frame.pts = self.pts  # Set the PTS for the frame
        self.pts += self.samples_per_frame  # Increment PTS counter by the number of samples in the frame

        return frame

# Create a child class to MediaRecorder class to record audio data to a buffer
class BufferMediaRecorder(MediaRecorder):
    """
    A subclass of MediaRecorder that supports using BytesIO buffer as output.
    
    :param buffer: The buffer containing audio data as a BytesIO object.
    """
    def __init__(self, buffer, format="wav"):
        self.__container = av.open(buffer, format=format, mode="w") 
        self.__tracks = {} 
        super().__init__(buffer, format=format) 

class NumpyAudioStreamTrack(MediaStreamTrack):
    """
    Custom class that inherits from MediaStreamTrack and reads
    from a NumPy array, converting it to a MediaStreamTrack object.
    """

    kind = "audio"

    def __init__(self, audio_array, add_silence=False, sample_rate=48000, channels=1, samples_per_frame=960):
        super().__init__()  # Initialize the MediaStreamTrack base class
        self.audio_array = audio_array
        self.add_silence = add_silence
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = 2  # Assuming 16-bit samples (2 bytes per sample)
        self.samples_per_frame = samples_per_frame
        self.pts = 0  # Initialize PTS counter
        self._start = None  # Track the start time
        self._timestamp = 0  # Track the elapsed samples
        self.frame_index = 0  # Track the current frame index
        # self.remaining_audio_array_length_to_stream = len(audio_array)  # Initialize remaining length of the array

        self.logs={} # Speech and timestamp

    async def recv(self):

        if self.add_silence:
            self.audio_array=np.append(self.audio_array,np.zeros(self.sample_rate))

        if self._start is None:
            self._start = time.time()

        # Calculate the number of samples to read for this frame
        start_index = self.frame_index * self.samples_per_frame * self.channels
        end_index = start_index + self.samples_per_frame * self.channels

        if end_index > len(self.audio_array):
            raise EOFError("End of audio stream")

        # Get the audio data for the current frame
        data = self.audio_array[start_index:end_index]

        # Decrease the remaining length
        # self.remaining_audio_array_length_to_stream -= len(data)

        # Update the frame index for the next read
        self.frame_index += 1

        # Reshape the array to be 2D: (number of frames, number of channels)
        # data = data.reshape(-1, self.channels)
        data = data.reshape(-1, 1)

        # Convert float32 data to int16
        if data.dtype == np.float64:
            # Normalize and convert to int16
            data = np.clip(data, -1.0, 1.0)  # Ensure values are in the range [-1.0, 1.0]
            data = (data * 32767).astype(np.int16)

        # print(data.shape)

        frame = av.AudioFrame.from_ndarray(
            data.T,
            format="s16",
            layout="stereo" if self.channels == 2 else "mono"
        )

        frame.sample_rate = self.sample_rate
        frame.time_base = Fraction(1, self.sample_rate)
        frame.pts = self.pts
        self.pts += self.samples_per_frame

        # Calculate wait time to synchronize the audio
        self._timestamp += self.samples_per_frame
        wait = self._start + (self._timestamp / self.sample_rate) - time.time()

        if wait > 0.001:
            await asyncio.sleep(wait)

        return frame
