import asyncio
import json
import uuid
from aiortc import AudioStreamTrack
import io
import numpy as np
import torch
import torchaudio
import math, wave
import requests
from datetime import datetime
from media_classes import NumpyAudioStreamTrack
from utils import (pcs, 
                    clients, 
                    welcome_array, 
                    warning_array,
                    state1_array,
                    state2_yes_array,
                    state2_no_array,
                    state3_yes_array, 
                    state3_no_array,
                    state4_yes_array,
                    state4_no_array,
                    state5_array) 
from llm_handler import (get_embedding,
                         find_similarity,
                            positive_embeds,
                            negative_embeds,
                            repeat_embeds)

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
REMINDER_TIME = 20 # seconds
DISCONNECT_TIME = 4 # seconds
CHUNK_SAMPLES = 512
CHANNELS = 2
BIT_DEPTH = 2
CHUNK_SIZE = int(CHUNK_SAMPLES * CHANNELS * BIT_DEPTH * (ORIG_SAMPLE/ SAMPLE_RATE)) # amt of bytes per chunk

# SILENCE_SAMPLES = SAMPLE_RATE * SILENCE_TIME
SILENCE_SAMPLES = ORIG_SAMPLE * SILENCE_TIME
SILENCE_CHUNKS = math.ceil(SILENCE_SAMPLES/(CHUNK_SAMPLES*BIT_DEPTH*CHANNELS))

REMINDER_SAMPLES = ORIG_SAMPLE * REMINDER_TIME 
REMINDER_CHUNKS=math.ceil(REMINDER_SAMPLES/(CHUNK_SAMPLES*BIT_DEPTH*CHANNELS))

DISCONNECT_SAMPLES = ORIG_SAMPLE * DISCONNECT_TIME 
DISCONNECT_CHUNKS=math.ceil(REMINDER_SAMPLES/(CHUNK_SAMPLES*BIT_DEPTH*CHANNELS))

resample = torchaudio.transforms.Resample(orig_freq = ORIG_SAMPLE, new_freq = SAMPLE_RATE)
resample_back= torchaudio.transforms.Resample(orig_freq = SAMPLE_RATE, new_freq = ORIG_SAMPLE)


class Client:
    """
        To manage Co-routines for each client, we'll use different co-routines as member functions of Client,
        and shared data of different co-routines will be our data members.
    """
    def __init__(self, client_id):
        self.client_id = client_id # uuid - client_id for different clients
        self.buffer_lock=asyncio.Lock() # asyncio lock to prevent Race Condition for audio_buffer (data member)
        self.audio_buffer = io.BytesIO() # streamed audio is recorded into BytesIO objects by start_recorder co-routine

        self.vad_dictionary = {} # a dictionary containing {'chunk': isSpeech (1/0) }
        # self.speech_audio=torch.tensor([]) # tensor that holds speech inside VAD co-routine
        # self.silence_audio=torch.tensor([]) # tensor that holds silence inside VAD co-routine
        self.speech_threshold=0.0 # Initial speech_threshold that will be modified by Adaptive Thresholding inside VAD co-routine
        self.prob_data = [] # List of probability of speeches that is used inside VAD co-routine
        self.silence_found = False # Flag to indicate that 2 seconds of silence is found inside VAD co-routine

        # self.n = 0 
        self.voiced_chunk_count = 0 # Number of voiced_chunks to count to implement interrupts
        self.silence_count=0 # Counter that counts number of silence chunks 
        # self.interruption = False # Flag to indicate that interruption has occurred inside VAD co-routine i.e, CLient is speaking during streaming
        self.m = 0 # Counter of Interruption

        self.asr_queue=asyncio.Queue()
        self.llm_queue=asyncio.Queue()
        self.tts_queue=asyncio.Queue()
        self.audio_sender_queue = asyncio.Queue() # A Queue where TTS co-routine will push audio_array

        self.state=1
        self.out_state=4

        self.replace_track=False # Flag to indicate that track has to be replaced inside send_audio_back co-routine
        self.audio_array=np.array([]) # an array that holds audio that has to be streamed back to the Client
        self.out_stream_status=False # Flag to indicate that server is streaming 
        
        self.played_reminder=False # Flag to indicate that reminder-audio has to be played
        self.logs={}

    # A Co-routine that starts recording audio_butes into audio_buffer
    async def start_recorder(self, recorder): 
        async with self.buffer_lock:
            await recorder.start() # recorder is instance of BufferMediaRecorder Class that records to audio_buffer datamember
    
    # A co-routine that reads chunks from the audio_buffer where server is writing audio_bytes continuously into
    async def read_buffer_chunks(self):
        while True:
            await asyncio.sleep(0.0001)  # adjust the sleep time based on your requirements
            async with self.buffer_lock:
                
                self.audio_buffer.seek(0, io.SEEK_END) # seek to end of audio
                size = self.audio_buffer.tell() # size of audio
                
                if size>=CHUNK_SIZE:
                    self.audio_buffer.seek(0, io.SEEK_SET)
                    chunk = self.audio_buffer.read(CHUNK_SIZE)
                    # print(len(chunk))
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
        # print(self.vad_dictionary)
        # To convert from BytesAudio to PyTorch tensor, first convert
        # from BytesAudio to np_chunk and normalize to [-1,1] range.
        # Then mean from the number of CHANNELS of audio to single
        # channel audio, convert to PyTorch tensor, and resample from
        # 44100 Hz to 16000 Hz audio
        np_chunk = np.frombuffer(chunk, dtype = np.int16)
        np_chunk = np_chunk.astype(np.float32) / 32768.0
        np_chunk = np_chunk.reshape(-1, CHANNELS).mean(axis = 1)
        # print(np_chunk.shape)
        chunk_audio = torch.from_numpy(np_chunk)
        # print(chunk_audio.shape)
        chunk_audio = resample(chunk_audio)
        # print(chunk_audio.shape)
        # Find prob of speech for using silero-vad model

        self.speech_prob = model(chunk_audio, SAMPLE_RATE).item()
        self.prob_data.append(self.speech_prob)
        
        # if self.interruption:
        #     self.m += 1
        
        # if not self.silence_found:
        if self.speech_prob >= self.speech_threshold:
            # Add chunk to the speech tensor and clear the silence tensor
            # self.speech_audio = torch.cat((self.speech_audio, chunk_audio), dim=0)
            # self.silence_audio = torch.empty(0)
            self.voiced_chunk_count += 1
            self.silence_count=0
            self.vad_dictionary[chunk]=1

            # print('added voice chunk')
        else:
            self.vad_dictionary[chunk]=0
            self.silence_count+=1
            # print('detected silence chunk')
                # Add chunk to both silence tensor and speech tensor
                # self.silence_audio = torch.cat((self.silence_audio, chunk_audio), dim=0)
                # self.speech_audio = torch.cat((self.speech_audio, chunk_audio), dim=0)
                # If the silence is longer than the SILENCE_TIME (2 sec)
                # save outputSpeech and add path to
                # client_audiosender_buffer where send_audio_back will
                # use the path to play it.

                # if self.silence_audio.shape[0] >= SILENCE_SAMPLES:
                    # # silence_found=True
                    # # TEMPORARY: saving the speech into outputSpeech.wav
                    # speech_unsq = torch.unsqueeze(self.speech_audio, dim=0)
                    # # torchaudio.save("outputSpeech_"+client_id+str(n)+".wav", speech_unsq, SAMPLE_RATE)
                    # # np.savetxt('speech_unsq'+str(n)+'.txt', speech_unsq.numpy())
                    # # print(f" Saved at outputSpeech_"+client_id+str(n)+".wav")

                    # # print(speech_unsq.numpy())
                    # await self.audio_sender_queue.put(resample_back(speech_unsq).numpy()) # push to client_audiosender_buffer which will be read continuously by audio_sender couroutine
                    
                    # # client_audiosender_buffer[client_id].append("outputSpeech_ead0cdc4-b0a2-49fb-b532-8bf4e464fc550.wav")
                    # # print("voiced chunk count", voiced_chunk_count)

                    # # save speech data into client_speech
                    # # client_speech[client_id] = speech_unsq.numpy()
                    # self.speech_audio = torch.empty(0)
                    # self.silence_audio = torch.empty(0)
                    # self.voiced_chunk_count = 0
                    # self.silence_found = True
                    # self.n+=1
        # else:
        #     if self.speech_prob >= self.speech_threshold:
                
        #         # Reset silence_found to False and start accumulating new speech
        #         self.silence_found = False
        #         self.speech_audio = torch.cat((self.speech_audio, chunk_audio), dim=0)
        #         self.silence_audio = torch.empty(0)
        #         self.voiced_chunk_count += 1

        # Adaptive thresholding which should allow for silence at the beginning
        # of audio and adapt to differing confidence levels of the VAD model.
        # Equation acquired from link:
        # https://vocal.com/voice-quality-enhancement/voice-activity-detection-with-adaptive-thresholding/
        self.speech_threshold = threshold_weight * max([i**2 for i in self.prob_data]) + (1 - threshold_weight) * min([i**2 for i in self.prob_data])

    async def read_vad_dictionary(self):
        
        c=0 # counter for saving audio chunks as file temporarily

        while True:
            await asyncio.sleep(0.0001)  # adjust the sleep time based on your requirements
            # print('in read_vad_dictionary',self.vad_dictionary)
            # check if there's number of chunks in vad_dictionary with all consecutive silence chunks then pop the chunks samples
            if not self.silence_found:
                if self.vad_dictionary and all(i==0 for i in list(self.vad_dictionary.values())[- SILENCE_CHUNKS:]):
                    
                    popped_chunks=list(self.vad_dictionary.keys())[:]

                    # # Filter out silence chunks before joining chunks to get only chunks that has audio
                    # audio_chunks = [k for k, v in self.vad_dictionary.items() if v != 0]
                    
                    print('popped chunks',len(popped_chunks))
                    audio_bytes=b''.join(popped_chunks)

                    for i in popped_chunks:
                        del self.vad_dictionary[i]
                    
                    # Save the audio bytes as a .wav file to test if VAD is working correctly
                    # with wave.open('myaudio'+str(c)+'.wav', 'wb') as wf:
                    #     wf.setnchannels(2)  # Assuming mono audio
                    #     wf.setsampwidth(2)  # Assuming 16-bit audio
                    #     wf.setframerate(48000)  # Assuming a sample rate of 16kHz
                    #     wf.writeframes(audio_bytes)
                    
                    self.logs['VADoutput'+str(uuid.uuid4())]=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    self.voiced_chunk_count=0
                    await self.asr_queue.put(audio_bytes)

                    c+=1
                    self.silence_found=True

            else:
                if self.vad_dictionary and list(self.vad_dictionary.values())[-1]==1:
                    print("Detected Voice")
                    self.silence_found=False

                # break
    async def asr(self):

        asr_base_url='http://192.168.88.10:8028/transcribe_abhi'

        while True:
            await asyncio.sleep(0.0001)

            if self.asr_queue:
                if not self.asr_queue.empty():  # Check if the queue is empty
                    print('audiobytes added to asr queue')

                    audio_bytes = await self.asr_queue.get() # Get data from the queue
                    self.asr_queue.task_done()
                    
                    # Prepare for POST request
                    files = {
                        "audio_file": ("audio.wav", audio_bytes, "audio/wav")
                    }

                    # # Send the POST request
                    # files = {
                    #     "audio": ("audio.wav", audio_bytes, "audio/wav")
                    # }

                    response = requests.post(asr_base_url, files=files)

                    asr_output_text=response.json()
                    print(asr_output_text)

                    self.logs['ASROutput: '+asr_output_text]=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    await self.llm_queue.put(asr_output_text)


    async def llm(self):
        # llm_base_url="http://192.168.88.40:8026/embed"

        while True:
            await asyncio.sleep(0.0001)

            if self.llm_queue and not self.llm_queue.empty():
                
                text = await self.llm_queue.get() # Get data from the queue
                self.llm_queue.task_done()

                try:
                    text_embedding =  get_embedding(text)
                    print(text_embedding)
                except:
                    print('£'*100)
                    print("Could not get text embedding")

                # out_state = self.state+1 # Out state increase by 1

                print("&"*100)
                print(np.array(text_embedding).shape)
                print(np.array(positive_embeds[0]).shape)
                print("&"*100)

                if self.state == 1:
                    similarity = find_similarity(text_embedding, positive_embeds[self.state-1], negative_embeds[self.state-1],repeat_embeds[0])
                    if similarity == 0:
                        response = 'yes'
                    elif similarity == 1:
                        response = 'repeat'
                    else :
                        response = 'no'
                
                elif self.state == 2:
                    similarity =   find_similarity(text_embedding, positive_embeds[self.state-1], negative_embeds[self.state-1], repeat_embeds[0])
                    if similarity == 0:
                        response = 'yes'
                    elif similarity == 1:
                        response = 'repeat'
                    else :
                        response = 'no'
                        
                elif self.state==3:
                    similarity =   find_similarity(text_embedding, positive_embeds[self.state-1], negative_embeds[self.state-1],repeat_embeds[0])
                    if similarity == 0:
                        response = 'yes'
                    elif similarity == 1:
                        response = 'repeat'
                    else :
                        response = 'no'

                elif self.state==4:
                    similarity =   find_similarity(text_embedding, positive_embeds[self.state-1], negative_embeds[self.state-1],repeat_embeds[0])
                    if similarity == 0:
                        response = 'yes'
                    elif similarity == 1:
                        response = 'repeat'
                    else :
                        response = 'no'
                else:
                    out_state = 4
                    response = 'no'

                print(response,self.state)
                # # response = requests.post(llm_base_url, data={'text':text}) # get response from the endpoint
                # # llm_output=response.json()
                # llm_output='send to tts'
                # print('output of llm:',llm_output)

                self.logs['LLMOutput '+str(uuid.uuid4())+':'+response]=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                await self.tts_queue.put(response)
                # print('got response from llm')

    async def tts(self):
        # tts_base_url =''

        while True:
            await asyncio.sleep(0.0001)

            if self.tts_queue and not self.tts_queue.empty():
                print("tts queue has new item")
                llm_output = await self.tts_queue.get() # Get data from the queue
                self.tts_queue.task_done()
                
                # # response = requests.post(tts_base_url, data={'tts_input':tts_input}) # get response from the
                # # audio_array=response
                # with wave.open("./audios/tts-output-1.wav", 'rb') as wav_file:
                #     num_frames = wav_file.getnframes()
                #     audio_data = wav_file.readframes(num_frames)

                # np_array=np.frombuffer(audio_data,dtype=np.int16)
                # np_array=np_array.astype(np.float32) /32768.0
                # audio_tensor = torch.from_numpy(np_array)
                # audio_data = torchaudio.functional.resample(audio_tensor, 22050, 48000)
                # # Step 4: Convert mono tensor to stereo by duplicating the channel
                # stereo_tensor = torch.stack([audio_data, audio_data], dim=0)

                # audio_array=stereo_tensor.numpy()

                if self.state==1:
                    if llm_output=='yes':
                        audio_array=state2_yes_array
                        self.state+=1
                    elif llm_output=='no':
                        audio_array=state2_no_array
                        self.state=self.out_state
                    elif llm_output=='repeat':
                        audio_array=state1_array
                elif self.state==2:
                    if llm_output=='yes':
                        audio_array=state3_yes_array
                        self.state+=1
                    elif llm_output=='no':
                        audio_array=state3_no_array
                        self.state=self.out_state
                    elif llm_output=='repeat':
                        audio_array=state2_yes_array
                elif self.state==3:
                    if llm_output=='yes':
                        audio_array=state4_yes_array
                        self.state+=1
                    elif llm_output=='no':
                        audio_array=state4_no_array
                        self.state=self.out_state
                    elif llm_output=='repeat':
                        audio_array=state3_yes_array
                elif self.state==4:
                    if llm_output=='yes':
                        audio_array=state5_array
                print('text converted to audio',self.state)
                # print()

                self.logs['TTSOutput'+str(uuid.uuid4())+'.wav']=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                await self.audio_sender_queue.put(audio_array)
                print('pushed audio array to audio_sender_queue')

    # Audio_sender co-routine that should handle all the audio_streaming part
    # This should include if audio_path is available in the client_audiosender_buffer
    # # This should also include if client speaks while streaming, it should interrupt  
    async def send_audio_back(self, audio_sender, client_id,pc):

        numpy_track=NumpyAudioStreamTrack(np.zeros(48000), add_silence=True)
        # numpy_track=NumpyAudioStreamTrack(state1_array, add_silence=True)

        while True:
            await asyncio.sleep(0.0001)
            if self.audio_sender_queue:
                # print(self.audio_sender_queue.qsize())
                # print('queue size:', self.audio_sender_queue.qsize())

                if not self.audio_sender_queue.empty():  # Check if the queue is empty
                #     await asyncio.sleep(1)
                #     # print('queue empty and audio_array length',len(audio_array))
                #     # numpy_track.add_silence=True
                #     print('queue empty --> append silence at the end of array and audio_array length =',numpy_track.audio_array.shape)
                #     # audio_array = np.append(audio_array, np.zeros(16000))
                #     # print('appended silence: ', np.zeros(44100))

                # else:
                    print('audio_sender_queue has new item')
                    self.audio_array = await self.audio_sender_queue.get()
                    self.out_stream_status=True
                    # print(audio_array, audio_array.shape)
                    numpy_track=NumpyAudioStreamTrack(self.audio_array, add_silence=True)
                    print('ReplacedTrack')    
                    # numpy_track.audio_array=audio_array
                    # numpy_track.add_silence=False
                    self.replace_track=True
                    # audio_array = np.append(audio_array, audio_data)
                    # print(audio_data)
                    self.audio_sender_queue.task_done()  # Indicate that the item has been consumed from the queue
                    # print('queue size:', self.audio_sender_queue.qsize())
                    
                    # Save the audio bytes as a .wav file
                    # with wave.open('original_audio.wav', 'wb') as wf:
                    #     wf.setnchannels(2)  # Assuming mono audio
                    #     wf.setsampwidth(2)  # Assuming 16-bit audio
                    #     wf.setframerate(48000)  # Assuming a sample rate of 16kHz
                    #     wf.writeframes(self.audio_buffer.getvalue())

                if self.replace_track:
                    # print(audio_array)
                    # numpy_track=NumpyAudioStreamTrack(audio_array)
                    audio_sender.replaceTrack(numpy_track)

                    self.logs['Streaming speech'+str(uuid.uuid4())]=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    self.replace_track=False

            # To Handle Interrupt
            if self.voiced_chunk_count >= 5 and self.out_stream_status==True:
                print(self.voiced_chunk_count)
                print("Detected interruption --> Stream Silence")
                audio_sender.replaceTrack(AudioStreamTrack())

                self.logs['Interrupt detected Streaming silence'+str(uuid.uuid4())]=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                self.out_stream_status=False
            
            # To Handle Reminder that Client hasn't spoken for 6 seconds before we disconnect client 
            if not self.played_reminder and self.silence_count>=REMINDER_CHUNKS:
                print("Play Reminder audio")
                
                numpy_track=NumpyAudioStreamTrack(warning_array, add_silence=True)
                # reminder_track=BytesIOAudioStreamTrack(io.BytesIO(audio_data))
                audio_sender.replaceTrack(numpy_track)
                self.logs['Play Reminder Audio'+str(uuid.uuid4())]=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                self.played_reminder=True

            if self.played_reminder:
                if self.silence_count>=(REMINDER_CHUNKS+DISCONNECT_CHUNKS):
                    print("Disconnect Client")

                    # Save to a JSON file
                    with open('logs/'+self.client_id+'.json', 'w') as json_file:
                        json.dump(self.logs, json_file)

                    clients.pop(client_id, None)  # Remove the client from the clients dictionary
                    pcs.discard(pc)  # Remove the pc from the set of peer connections

                    
                    await pc.close()

                    self.logs['Disconnected'+str(uuid.uuid4())]=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    break

                if self.voiced_chunk_count>0:
                    self.played_reminder=False
            
            if self.state==self.out_state:
                await asyncio.sleep(7)
                await pc.close()
                pcs.discard(pc)
                clients.pop(client_id, None)

            



