from fastapi import FastAPI, HTTPException, Form
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, MediaStreamTrack, AudioStreamTrack
import json
import asyncio
from aiortc.contrib.media import MediaRecorder, MediaPlayer
import av
import io
from starlette.responses import StreamingResponse
import numpy as np
import wave
import requests
import torch
import torchaudio
import shutil
import tempfile
import uuid
import av
import time
from fractions import Fraction
import gc
# import logging

app = FastAPI() # Initialize the FastAPI 
pcs = set() # set of peer connections
clients={} # set of client instances that can be cleanedup once the client disconnects

# loading model for vad
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=False)

# Global constants related to the audio input format and chosen chunk values.
# Edit as appropriate for the input stream.
SAMPLE_RATE = 16000
ORIG_SAMPLE = 48000
SILENCE_TIME = 2 # seconds
CHUNK_SAMPLES = 512
CHANNELS = 2
BIT_DEPTH = 2
CHUNK_SIZE = int(CHUNK_SAMPLES * CHANNELS * BIT_DEPTH * (ORIG_SAMPLE/ SAMPLE_RATE)) # amt of bytes per chunk
SILENCE_SAMPLES = SAMPLE_RATE * SILENCE_TIME

resample = torchaudio.transforms.Resample(orig_freq = ORIG_SAMPLE, new_freq = SAMPLE_RATE)
resample_back= torchaudio.transforms.Resample(orig_freq = SAMPLE_RATE, new_freq = ORIG_SAMPLE)

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

class NumpyAudioStreamTrack(MediaStreamTrack):
    """
    Custom class that inherits from MediaStreamTrack and reads
    from a NumPy array, converting it to a MediaStreamTrack object.
    """

    kind = "audio"

    def __init__(self, audio_array, add_silence=False, sample_rate=48000, channels=2, samples_per_frame=960):
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

        # Update the frame index for the next read
        self.frame_index += 1

        # Reshape the array to be 2D: (number of frames, number of channels)
        data = data.reshape(-1, self.channels)

        # Convert float64 data to int16
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


class Client:
    '''
        For Managing Co-routines and Readability, we'll use different co-routines as member functions of Client

    '''
    def __init__(self, client_id):
        self.client_id = client_id # uuid - client_id for different clients
        self.buffer_lock=asyncio.Lock() # asyncio lock to prevent Race Condition for audio_buffer (data member)
        self.audio_buffer = io.BytesIO() # streamed audio is recorded into BytesIO objects by start_recorder co-routine
        self.speech_audio=torch.tensor([]) # tensor that holds speech inside VAD co-routine
        self.silence_audio=torch.tensor([]) # tensor that holds silence inside VAD co-routine
        self.speech_threshold=0.0 # Initial speech_threshold that will be modified by Adaptive Thresholding inside VAD co-routine
        self.prob_data = [] # List of probability of speeches that is used inside VAD co-routine
        self.silence_found = False # Flag to indicate that 2 seconds of silence is found inside VAD co-routine
        self.n = 0 
        self.voiced_chunk_count = 0 # Number of voiced_chunks to count to implement interrupts
        self.interruption = False # Flag to indicate that interruption has occurred inside VAD co-routine i.e, CLient is speaking during streaming
        self.m = 0 # Counter of Interruption
        self.audio_sender_queue = asyncio.Queue() # A Queue where TTS co-routine will push audio_array
        self.replace_track=True # Flag to indicate that track has to be replaced inside send_audio_back co-routine
        self.audio_array=np.array([]) # an array that holds audio that has to be streamed back to the Client

        self.chunk_append=b'' # an array that holds speech instead of using torch 

    # A Co-routine that starts recording audio_butes into audio_buffer
    async def start_recorder(self, recorder): 
        async with self.buffer_lock:
            await recorder.start() # recorder is instance of BufferMediaRecorder Class that records to audio_buffer datamember
    
    # A co-routine that reads chunks from the audio_buffer where server is writing audio_bytes continuously into
    async def read_buffer_chunks(self):
        while True:
            await asyncio.sleep(0.01)  # adjust the sleep time based on your requirements
            async with self.buffer_lock:
                
                self.audio_buffer.seek(0, io.SEEK_END) # seek to end of audio
                size = self.audio_buffer.tell() # size of audio
                
                if size>=CHUNK_SIZE:
                    self.audio_buffer.seek(0, io.SEEK_SET)
                    chunk = self.audio_buffer.read(CHUNK_SIZE)

                    # Implement VAD in this chunk
                    asyncio.ensure_future(self.VAD(chunk))

                    self.audio_buffer.seek(0)
                    self.audio_buffer.truncate()

    # VAD function using Silero-VAD model, https://github.com/snakers4/silero-vad,
    # Receives chunk of audio in bytes and converts to PyTorch Tensor. If the chunk
    # has voice in it, the function adds it to a tensor 'speech_audio' and clears 
    # the tensor 'silence_audio', and if it does not, it adds it to 'silence_audio'. 
    # When 'silence_audio' is SILENCE_TIME long (2 seconds), it will pass the speech 
    # to 'client_speech', and pop from 'client_audio'.
    async def VAD(self, chunk, threshold_weight = 0.9):
        # To convert from BytesAudio to PyTorch tensor, first convert
        # from BytesAudio to np_chunk and normalize to [-1,1] range.
        # Then mean from the number of CHANNELS of audio to single
        # channel audio, convert to PyTorch tensor, and resample from
        # 44100 Hz to 16000 Hz audio

        np_chunk_orig = np.frombuffer(chunk, dtype = np.int16)
        np_chunk = np_chunk_orig.astype(np.float32) / 32768.0
        np_chunk = np_chunk.reshape(-1, CHANNELS).mean(axis = 1)
        # print(np_chunk.shape)
        chunk_audio = torch.from_numpy(np_chunk)
        chunk_audio = resample(chunk_audio)

        # Find prob of speech for using silero-vad model
        self.speech_prob = model(chunk_audio, SAMPLE_RATE).item()
        self.prob_data.append(self.speech_prob)
        
        if self.interruption:
            self.m += 1
        
        if not self.silence_found:
            if self.speech_prob >= self.speech_threshold:
                # Add chunk to the speech tensor and clear the silence tensor
                self.chunk_append += chunk 
                self.speech_audio = torch.cat((self.speech_audio, chunk_audio), dim=0)
                self.silence_audio = torch.empty(0)
                self.voiced_chunk_count += 1

            else:
                self.chunk_append += chunk 
                # Add chunk to both silence tensor and speech tensor
                self.silence_audio = torch.cat((self.silence_audio, chunk_audio), dim=0)
                self.speech_audio = torch.cat((self.speech_audio, chunk_audio), dim=0)
                # If the silence is longer than the SILENCE_TIME (2 sec)
                # save outputSpeech and add path to
                # client_audiosender_buffer where send_audio_back will
                # use the path to play it.

                if self.silence_audio.shape[0] >= SILENCE_SAMPLES:
                    # silence_found=True
                    # TEMPORARY: saving the speech into outputSpeech.wav
                    # speech_unsq = torch.unsqueeze(self.speech_audio, dim=0)

                    # speech_resampled = resample_back(speech_unsq)
                
                    # # Convert back to stereo by duplicating the mono channel
                    # stereo_audio = torch.cat([speech_resampled, speech_resampled], dim=0)

                    # torchaudio.save(f"outputSpeech_{self.client_id}{self.n}.wav", stereo_audio, ORIG_SAMPLE)

                    # torchaudio.save("outputSpeech_"+self.client_id+str(self.n)+".wav", resample_back(speech_unsq), ORIG_SAMPLE)
                    # np.savetxt('speech_unsq'+str(n)+'.txt', speech_unsq.numpy())
                    # print(f" Saved at outputSpeech_"+client_id+str(n)+".wav")

                    # print(speech_unsq.numpy())
                    numpy_audio=np.frombuffer(self.chunk_append, dtype=np.int16)
                    with wave.open('output_audio.wav', 'wb') as wf:
                        # Set the parameters for the WAV file
                        wf.setnchannels(2)  # Number of audio channels
                        wf.setsampwidth(2)  # Sample width in bytes
                        wf.setframerate(48000)  # Sample rate
                        wf.writeframes(self.chunk_append)  # Write the audio data

                    await self.audio_sender_queue.put(np.frombuffer(self.chunk_append, dtype=np.int16)) # push to client_audiosender_buffer which will be read continuously by audio_sender couroutine
                    
                    # client_audiosender_buffer[client_id].append("outputSpeech_ead0cdc4-b0a2-49fb-b532-8bf4e464fc550.wav")
                    # print("voiced chunk count", voiced_chunk_count)

                    # save speech data into client_speech
                    # client_speech[client_id] = speech_unsq.numpy()
                    self.chunk_append = b''
                    self.speech_audio = torch.empty(0)
                    self.silence_audio = torch.empty(0)
                    self.voiced_chunk_count = 0
                    self.silence_found = True
                    self.n+=1
        else:
            if self.speech_prob >= self.speech_threshold:

                # Reset silence_found to False and start accumulating new speech
                self.silence_found = False
                self.speech_audio = torch.cat((self.speech_audio, chunk_audio), dim=0)
                self.silence_audio = torch.empty(0)
                self.voiced_chunk_count += 1

        # Adaptive thresholding which should allow for silence at the beginning
        # of audio and adapt to differing confidence levels of the VAD model.
        # Equation acquired from link:
        # https://vocal.com/voice-quality-enhancement/voice-activity-detection-with-adaptive-thresholding/
        self.speech_threshold = threshold_weight * max([i**2 for i in self.prob_data]) + (1 - threshold_weight) * min([i**2 for i in self.prob_data])

    # async def asr():

    #     while True:
    #         await asyncio.sleep(0.01)

    #         if asr_queue and asr_queue[client_id]:
    #             print('text extracted from speech')


    # async def llm():
    #     while True:
    #         await asyncio.sleep(0.01)

    #         if llm_queue and llm_queue[client_id]:
    #             print('got response from llm')

    # async def tts():
    #     while True:
    #         await asyncio.sleep(0.01)

    #         if tts_queue and tts_queue[client_id]:
    #             print('text converted to audio')
    #             print('push audio array to audio_sender_queue')

    # Audio_sender co-routine that should handle all the audio_streaming part
    # This should include if audio_path is available in the client_audiosender_buffer
    # # This should also include if client speaks while streaming, it should interrupt  
    async def send_audio_back(self, audio_sender):

        numpy_track=NumpyAudioStreamTrack(np.zeros(48000), add_silence=True)

        while True:
            await asyncio.sleep(0.01)
            if self.audio_sender_queue:
                # print('queue size:', self.audio_sender_queue.qsize())

                if self.audio_sender_queue.empty():  # Check if the queue is empty
                    await asyncio.sleep(1)
                    # print('queue empty and audio_array length',len(audio_array))
                    # numpy_track.add_silence=True
                    print('queue empty --> append silence at the end of array and audio_array length =',numpy_track.audio_array.shape)
                    # audio_array = np.append(audio_array, np.zeros(16000))
                    # print('appended silence: ', np.zeros(44100))

                else:
                    
                    self.audio_array = await self.audio_sender_queue.get()

                    # print(audio_array, audio_array.shape)
                    numpy_track=NumpyAudioStreamTrack(self.audio_array, add_silence=True)
                        
                    # numpy_track.audio_array=audio_array
                    # numpy_track.add_silence=False
                    self.replace_track=True
                    # audio_array = np.append(audio_array, audio_data)
                    # print(audio_data)
                    self.audio_sender_queue.task_done()  # Indicate that the item has been consumed from the queue
                    # print('queue size:', self.audio_sender_queue.qsize())
                    
                if self.replace_track:
                    # print(audio_array)
                    # numpy_track=NumpyAudioStreamTrack(audio_array)
                    audio_sender.replaceTrack(numpy_track)
                    self.replace_track=False

# endpoint to accept offer from webrtc client for handshaking
@app.post("/offer")
async def offer_endpoint(sdp: str = Form(...), type: str = Form(...), client_id: str = Form(...)):

    # logging.info(f"Received SDP: {sdp}, type: {type}")
    config = RTCConfiguration(iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")]) # make use of google's stun server
    pc = RTCPeerConnection(configuration=config) # pass the config to configuration to make use of stun server
    pcs.add(client_id) # add peer connection to set of peer connections

    client=Client(client_id)
    clients[client_id] = client  # Add the client to the clients dictionary, which can be cleaned up once the client disconnects

    recorder = BufferMediaRecorder(client.audio_buffer)

    # event handler for data channel
    @pc.on("datachannel")
    def on_datachannel(channel):
        # client_datachannels[client_id]=channel # to make datachannel accessible outside of this scope
        channel.send(f"Hello I'm server")

        @channel.on("message")
        async def on_message(message):
            print(message)
            # logging.info(f"Message received: {message}")


    # event handler for tracks (audio/video)
    @pc.on("track")
    def on_track(track: MediaStreamTrack):
        print(f"Track {track.kind} received. Make sure to use .start() to start recording to buffer")
        if track.kind == "audio":
            recorder.addTrack(track)
            # audio_sender=pc.addTrack(MediaPlayer('./serverToClient.wav').audio)
            audio_sender=pc.addTrack(AudioStreamTrack())
            # asyncio.ensure_future(recorder.start())
            asyncio.ensure_future(client.start_recorder(recorder))
            asyncio.ensure_future(client.read_buffer_chunks())

            # Start coroutine to handle interrupts
            # for example: if audio is streaming back to client and client speaks in the middle, replaceTrack(AudioStreamTrack()) with silence
            asyncio.ensure_future(client.send_audio_back(audio_sender))

            # pc.addTrack(AudioStreamTrack())
            
        @track.on("ended")
        async def on_ended():
            print(f"Track {track.kind} ended")
            await recorder.stop()
            # asyncio.ensure_future(save_audio())
            # await pc.close()

    # Clean-up function for disconnection of clients
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        # print(pc.connectionState)
        if pc.connectionState in ["failed", "closed", "disconnected"]:
            print(f"Connection state is {pc.connectionState}, cleaning up")
            print("Deleting client_id:", client_id)

            await recorder.stop()

            clients.pop(client_id, None)  # Remove the client from the clients dictionary
            pcs.discard(pc)  # Remove the pc from the set of peer connections

            await pc.close()

            # Manually trigger garbage collection
            collected = gc.collect()

            # Verify memory release
            print(f"Garbage collector collected {collected} objects.")

    # Handshake with the clients to make WebRTC Connections
    try:
        offer_desc = RTCSessionDescription(sdp=sdp, type=type)
        await pc.setRemoteDescription(offer_desc)

        answer_desc = await pc.createAnswer()
        await pc.setLocalDescription(answer_desc)

        response = {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }
        
        print(response)
        # logging.info(f"Sending SDP answer: {response}")
        return response # respond with the sdp information of the server
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def getClients():

    return {
        "clients": list(pcs),
        "active": list(clients.keys())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8080) # Increase the number of workers as needed and limit_max_requests


# class NumpyAudioStreamTrack(MediaStreamTrack):
#     """
#     Custom class that inherits from MediaStreamTrack and reads
#     from a NumPy array, converting it to a MediaStreamTrack object.
#     """

#     kind = "audio"

#     def __init__(self, audio_array, sample_rate=16000, channels=1, samples_per_frame=960):
#         super().__init__()  # Initialize the MediaStreamTrack base class
#         self.audio_array = audio_array
#         self.sample_rate = sample_rate
#         self.channels = channels
#         self.sample_width = 4  # Assuming 16-bit samples (2 bytes per sample)
#         self.samples_per_frame = samples_per_frame
#         self.pts = 0  # Initialize PTS counter
#         self._start = None  # Track the start time
#         self._timestamp = 0  # Track the elapsed samples
#         self.frame_index = 0  # Track the current frame index

#     async def recv(self):

#         if self._start is None:
#             self._start = time.time()

#         # Calculate the number of samples to read for this frame
#         start_index = self.frame_index * self.samples_per_frame * self.channels
#         end_index = start_index + self.samples_per_frame * self.channels

#         if end_index > len(self.audio_array):
#             raise EOFError("End of audio stream")

#         # Get the audio data for the current frame
#         data = self.audio_array[start_index:end_index]

#         # Update the frame index for the next read
#         self.frame_index += 1

#         # Reshape the array to be 2D: (number of frames, number of channels)
#         data = data.reshape(-1, self.channels)

#         # Convert float64 data to int16
#         if data.dtype == np.float64:
#             # Normalize and convert to int16
#             data = np.clip(data, -1.0, 1.0)  # Ensure values are in the range [-1.0, 1.0]
#             data = (data * 32768).astype(np.int16)

#         frame = av.AudioFrame.from_ndarray(
#             data.T,
#             format="s16",
#             layout="stereo" if self.channels == 2 else "mono"
#         )

#         frame.sample_rate = self.sample_rate
#         frame.time_base = Fraction(1, self.sample_rate)
#         frame.pts = self.pts
#         self.pts += self.samples_per_frame

#         # Calculate wait time to synchronize the audio
#         self._timestamp += self.samples_per_frame
#         wait = self._start + (self._timestamp / self.sample_rate) - time.time()
#         if wait > 0:
#             await asyncio.sleep(wait)

#         return frame