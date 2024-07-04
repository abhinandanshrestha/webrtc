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
# import logging

app = FastAPI() # Initialize the FastAPI 

pcs = set() # set of peer connections
client_buffer={} # {'c1':'io.BytesIO()', 'c2':'io.BytesIO()', ...} client buffer mapping dictionary to store streaming audio data into buffer
client_chunks={} # {'c1':[], 'c2':[], ...} client - list mapping dictionary to read from the buffer and check if audio is available in the chunks
client_datachannels={} # {'c1': channelC1, 'c2':channelC2, ...} client - datachannels mapping dictionary to make channel accessible outside of the event handler
# logging.basicConfig(level=logging.DEBUG)
client_audio = {} # {'c1':[[]], 'c2': [[]], ...} client - np.array mapping dictionary for all of client's audio as one np.array, popped when VAD detects silence
client_speech = {} # {'c1':[[]], 'c2': [[]], ...} client - np.array mapping dictionary for output speech from VAD

client_info = {} # {'c1' : {'speech_tensor':'torch.tensor', 'silence_tensor':'torch.tensor', 'speech_threshold':float 'prob_data': [], 'silence_found':bool}} 
# client - dictionary mappign dictionary that stores data for the VAD logic, such as the PyTorch tensors for speech and silence, the adaptive thresholding value,
# and the list of the probabilities for use in the adaptive thresholding logic

buffer_lock = {}  # buffer_lock to avoid race condition
client_audiosender_buffer={} # {'c1':['<path to audio which will be used by audio_sender>'], 'c2':[]}

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
n=0
# VAD function using Silero-VAD model, https://github.com/snakers4/silero-vad,
# Receives chunk of audio in bytes and converts to PyTorch Tensor. If the chunk
# has voice in it, the function adds it to a tensor 'speech_audio' and clears 
# the tensor 'silence_audio', and if it does not, it adds it to 'silence_audio'. 
# When 'silence_audio' is SILENCE_TIME long (2 seconds), it will pass the speech 
# to 'client_speech', and pop from 'client_audio'.
async def VAD(chunk, client_id, threshold_weight = 0.9):
    global n
    # Pull information from client_info dictionary and save the
    # appropriate values for use in VAD, editing as needed within
    # VAD function.
    info = client_info[client_id]
    speech_threshold = info['speech_threshold']
    speech_audio = info['speech_audio']
    silence_audio = info['silence_audio']
    prob_data = info['prob_data']
    silence_found = info['silence_found']

    # To convert from BytesAudio to PyTorch tensor, first convert
    # from BytesAudio to np_chunk and normalize to [-1,1] range.
    # Then mean from the number of CHANNELS of audio to single
    # channel audio, convert to PyTorch tensor, and resample from
    # 44100 Hz to 16000 Hz audio
    np_chunk = np.frombuffer(chunk, dtype = np.int16)
    np_chunk = np_chunk.astype(np.float32) / 32768.0
    np_chunk = np_chunk.reshape(-1, CHANNELS).mean(axis = 1)
    chunk_audio = torch.from_numpy(np_chunk)
    chunk_audio = resample(chunk_audio)

    # Save all chunks to client_audio, collecting for VAD later
    # or allowing audio_sender to see interruptions.
    client_audio[client_id] = np.append(client_audio[client_id], chunk_audio.numpy())

    # Find prob of speech for using silero-vad model
    speech_prob = model(chunk_audio, SAMPLE_RATE).item()
    prob_data.append(speech_prob)

    if not silence_found:
        if speech_prob >= speech_threshold:
            # Add chunk to the speech tensor and clear the silence tensor
            speech_audio = torch.cat((speech_audio, chunk_audio), dim=0)
            silence_audio = torch.empty(0)
        else:
            # Add chunk to both silence tensor and speech tensor
            silence_audio = torch.cat((silence_audio, chunk_audio), dim=0)
            speech_audio = torch.cat((speech_audio, chunk_audio), dim=0)
            # If the silence is longer than the SILENCE_TIME (2 sec)
            # pop client_audio and save to client_speech, which LLM
            # will use
            if silence_audio.shape[0] >= SILENCE_SAMPLES:
                silence_found=True
                # TEMPORARY: saving the speech into outputSpeech.wav
                speech_unsq = torch.unsqueeze(speech_audio, dim=0)
                torchaudio.save("outputSpeech_"+client_id+str(n)+".wav", speech_unsq, SAMPLE_RATE)
                torchaudio.save("testSpeech_"+client_id+str(uuid.uuid4())+".wav", speech_unsq, SAMPLE_RATE)
                print(f"Speech data saved at outputSpeech_{client_id}.wav")

                if client_id not in client_audiosender_buffer:
                    client_audiosender_buffer[client_id]=[]

                client_audiosender_buffer[client_id].append("outputSpeech_"+client_id+str(n)+".wav") # push to client_audiosender_buffer which will be read continuously by audio_sender couroutine
                n+=1
                # Save the speech into a temporary file
                # with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                #     speech_unsq = torch.unsqueeze(speech_audio, dim=0)
                #     torchaudio.save(temp_wav.name, speech_unsq, SAMPLE_RATE)
                #     temp_path = temp_wav.name
                #     print(f"Speech data saved at {temp_path}")

                # pop from client_audio and save into client_speech
                speech = client_audio[client_id]
                client_audio[client_id] = np.empty(0)
                client_info[client_id]['speech_audio'] = torch.empty(0)
                client_speech[client_id] = speech
    
    # Adaptive thresholding which should allow for silence at the beginning
    # of audio and adapt to differing confidence levels of the VAD model.
    # Equation acquired from link:
    # https://vocal.com/voice-quality-enhancement/voice-activity-detection-with-adaptive-thresholding/
    speech_threshold = threshold_weight * max([i**2 for i in prob_data]) + (1 - threshold_weight) * min([i**2 for i in prob_data])

    # Save data back into client_info with updated values
    client_info[client_id] = {'speech_audio':speech_audio, 'silence_audio':silence_audio, 'speech_threshold':speech_threshold, 'prob_data': prob_data, 'silence_found':silence_found, 'streamAudio':False}


    # audio_sender._track_id and track._MediaStreamTrack__ended

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


# endpoint to accept offer from webrtc client for handshaking
@app.post("/offer")
async def offer_endpoint(sdp: str = Form(...), type: str = Form(...), client_id: str = Form(...)):

    # logging.info(f"Received SDP: {sdp}, type: {type}")
    config = RTCConfiguration(iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")]) # make use of google's stun server
    pc = RTCPeerConnection(configuration=config) # pass the config to configuration to make use of stun server
    pcs.add(client_id) # add peer connection to set of peer connections

    # Separate out buffer for each peer connection
    audio_buffer = io.BytesIO()
    client_buffer[client_id]=audio_buffer
    client_chunks[client_id]=[]
    client_audio[client_id]=np.empty(0)
    client_speech[client_id]=np.empty(0)
    client_info[client_id]={'speech_audio':torch.empty(0), 'silence_audio': torch.empty(0), 'speech_threshold':0.0, 'prob_data':[],'silence_found':False}
    buffer_lock[client_id]=asyncio.Lock()
    # By default, records the received audio to a file
    # example: recorder = MediaRecorder('received_audio.wav')
    # To record the received audio to the buffer, Implement a child class to the main MediaRecorder class
    recorder = BufferMediaRecorder(audio_buffer)

    # event handler for data channel
    @pc.on("datachannel")
    def on_datachannel(channel):
        client_datachannels[client_id]=channel # to make datachannel accessible outside of this scope
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
            asyncio.ensure_future(start_recorder(recorder))
            asyncio.ensure_future(read_buffer_chunks(client_id))

            # Start coroutine to handle interrupts
            # for example: if audio is streaming back to client and client speaks in the middle, replaceTrack(AudioStreamTrack()) with silence
            asyncio.ensure_future(send_audio_back(audio_sender, client_id))

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
            pcs.discard(client_id)
            del client_buffer[client_id]
            del client_chunks[client_id]
            del client_datachannels[client_id]
            del client_info[client_id]
            del client_audio[client_id]  
            del client_speech[client_id] 

    # start writing to buffer with buffer_lock
    async def start_recorder(recorder): 
        async with buffer_lock[client_id]:
            await recorder.start()
    
    async def read_buffer_chunks(client_id):
        while True:
            await asyncio.sleep(0.01)  # adjust the sleep time based on your requirements
            async with buffer_lock[client_id]:
                audio_buffer = client_buffer[client_id]
                audio_buffer.seek(0, io.SEEK_END)
                size = audio_buffer.tell()
                if size>=CHUNK_SIZE:
                    audio_buffer.seek(0, io.SEEK_SET)
                    chunk = audio_buffer.read(CHUNK_SIZE)

                    # Implement VAD in this chunk
                    asyncio.ensure_future(VAD(chunk, client_id))

                    audio_buffer.seek(0)
                    audio_buffer.truncate()

                # get the client's datachannel 
                # dc=client_datachannels[client_id]
                # dc.send("Iteration inside While Loop")

    # Audio_sender co-routine that should handle all the audio_streaming part
    # This should include if audio_path is available in the client_audiosender_buffer
    # # This should also include if client speaks while streaming, it should interrupt  
    async def send_audio_back(audio_sender, client_id):
        # print(audio_sender, client_audio)
        # Change interrupt threshold appropriately, likely lower than this, but should be tested
        INTERRUPT_THRESHOLD = 0.8

        while True:
            await asyncio.sleep(0.1)

            # Check if audio_path is written to client_audiosender_buffer from which audio has to be streamed to the client
            if client_audiosender_buffer and client_audiosender_buffer[client_id]:
                print(client_audiosender_buffer,client_audiosender_buffer[client_id])

                audio_path=client_audiosender_buffer[client_id].pop(0) # Pop audio_path from audio_sender_buffer of client that has to be sent
                print(audio_path)

                # Use replaceTrack flag to replace the track with the saved outputSpeech just once instead of replacing in a loop
                if not client_info[client_id]['streamAudio']:

                    client_info[client_id]['streamAudio'] = True

                    player=MediaPlayer(audio_path) # Create a MediaPlayer object
                    print(audio_path)
                    track=player.audio # add track to player
                    audio_sender.replaceTrack(track)
                    
                    print("Audio is being streamed to the client(samples)", client_speech[client_id].shape[0])
                    print("client_audio size", client_audio[client_id].shape[0])

                    # track._MediaStreamTrack__ended returns True if entire audio has been recorded by the client
                    while not track._MediaStreamTrack__ended:
                        await asyncio.sleep(0.1)
                        # print(track._MediaStreamTrack__ended)
                        # this is for incoming audio when audio is being streamed to client, it progressively checks client_audio for interrupts
                        # if client_info[client_id]['silence_found'] and client_speech[client_id].size:
                        if(client_audio[client_id].shape[0]):
                            np_interrupt = client_audio[client_id][:CHUNK_SAMPLES] # Collect samples from client_audio after speech has been popped from it and saved to path which is pushed to client_audiosender_buffer
                            print("np_interrupt size", np_interrupt.shape[0])     
                            if np_interrupt.shape[0]>=CHUNK_SAMPLES:   
                                interrupt_audio = torch.from_numpy(np_interrupt).float()
                                interrupt_prob = model(interrupt_audio, SAMPLE_RATE).item()
                                if interrupt_prob >= INTERRUPT_THRESHOLD: # Check if audio is found while streaming and if found stream silence to break from the loop
                                    # pass
                                    print('streaming silence to client')
                                    audio_sender.replaceTrack(AudioStreamTrack())
                                    client_info[client_id]['silence_found'] = False
                                    client_info[client_id]['streamAudio'] = False
                                    break
                                    # change track to silence
                                else:
                                    # pass and remove the first chunk from the buffer for the client
                                    client_audio[client_id] = client_audio[client_id][CHUNK_SAMPLES:]
                                    print('continue streaming audio to client')
                        print(track._MediaStreamTrack__ended)
                        
                    await asyncio.sleep(1) # Introduce slight delay before streamAudio is set False
                    client_info[client_id]['streamAudio'] = False
                    audio_sender.replaceTrack(AudioStreamTrack()) # Once all the audio has been streamed to the client, stream Silence again



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


# end point to send audio back to the client (for now just streaming the audio back to the client)
# endpoint: /audio?client_id=<client_id>
@app.get("/audio")
async def get_audio(client_id: int):
    print(client_buffer)
    audio_buffer = client_buffer[client_id]
    audio_buffer.seek(0) # seek the audio buffer to the start of the audio
    # # Read the entire content of the buffer
    # audio_data = audio_buffer.read()
    
    # # Print the audio data on the console
    # print(audio_data)
    return StreamingResponse(audio_buffer, media_type="audio/wav") 


# test endpoint to break data into chunks
# comment @app.get("/read-audio") and return statement
# and uncomment asyncio.ensure_future(save_audio()) to run the co-routine asynchronously
## @app.get("/read-audio")
## async def save_audio(client_id: int):
    # audio_buffer = client_buffer[client_id]
    # chunk_size=4096
    # audio_buffer.seek(0)
    # data = []
    # while True:
    #     # Read a chunk of bytes from the buffer
    #     chunk = audio_buffer.read(chunk_size)  # Adjust chunk size as needed
    #     # print(chunk)
    #     # Break the loop if no more data is available
    #     if not chunk:
    #         break

    #     # Append the chunk to the popped data
    #     data.append(chunk) 

    # audio_data_bytes = b''.join(data)
    # # audio_array = np.frombuffer(audio_data_bytes, dtype=np.int16)
    # with wave.open('abhi.wav', 'wb') as wf:
    #     wf.setnchannels(2)
    #     wf.setsampwidth(2)
    #     wf.setframerate(44100)
    #     wf.writeframes(audio_data_bytes)
    ## chunks=client_chunks[client_id]
    # print(chunks)
    ## return {"index":len(chunks)}
    # return {"data":audio_buffer.read(),
    #         "index":audio_buffer.tell()}

@app.get("/")
def getClients():

    return {
        "clients": list(pcs),  # Convert set to list for JSON serialization
        "client_buffer": list(client_buffer.keys()),  # Get all client IDs in the buffer
        "client_chunks": {client_id: len(chunks) for client_id, chunks in client_chunks.items()} , # Get all client chunks and their sizes
        "client_indi_chunk": {client_id: len(chunks[0]) for client_id, chunks in client_chunks.items()},
        "client_sum_chunk": {client_id: sum(len(chunk) for chunk in chunks) for client_id, chunks in client_chunks.items()}
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8080) # Increase the number of workers as needed and limit_max_requests