import json
import time
from pyVoIP.VoIP import CallState
import wave
import uuid
import io
import numpy as np
import torch
import math
import threading
import requests
import queue
from datetime import datetime
import scipy.io.wavfile as wav
from scipy.signal import resample
import sys, os
from llm_handler_tv import *

# Add the project root directory to sys.path and then import llm_handler from callbot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dbconn import models
from dbconn.database import SessionLocal, engine
import requests

# # Define the URL and parameters
# calllogs_endpoint_url = 'http://localhost:8001/call_logs/'

# # Loading model for VAD
model, utils = torch.hub.load(repo_or_dir='/home/oem/webrtc/silero-vad/',
                              source='local',
                              model='silero_vad',
                              force_reload=True,
                              onnx=False)


# Global constants related to the audio input format and chosen chunk values.
SAMPLE_RATE = 8000
SILENCE_TIME = 2  # seconds
CHUNK_SAMPLES = 256
CHANNELS = 1
BIT_DEPTH = 2
CHUNK_SIZE = int(CHUNK_SAMPLES * CHANNELS * BIT_DEPTH)  # amt of bytes per chunk

SILENCE_SAMPLES = SAMPLE_RATE * SILENCE_TIME
SILENCE_CHUNKS = math.ceil(SILENCE_SAMPLES / (CHUNK_SAMPLES * BIT_DEPTH * CHANNELS))

def convert_8bit_to_16bit(audio_data):
    audio_data = np.frombuffer(audio_data, dtype=np.uint8).astype(np.int16)
    audio_data = (audio_data - 128) * 256  # Scale from 8-bit to 16-bit
    return audio_data.tobytes()

def split_audio_bytes(audio_bytes, sample_rate=8000, sample_width=1):
    bytes_per_second = sample_rate * sample_width
    chunks = [audio_bytes[i:i + bytes_per_second] for i in range(0, len(audio_bytes), bytes_per_second)]
    return chunks

def get_audio_bytes(input_path):
    # Load the audio file
    sample_rate, data = wav.read(input_path)

    # Check if the audio is in 16-bit PCM format
    if data.dtype != np.int16:
        raise ValueError("Input audio is not in 16-bit PCM format")

    # Resample the audio to 8000 Hz
    target_sample_rate = 8000
    num_samples = int(len(data) * target_sample_rate / sample_rate)
    resampled_data = resample(data, num_samples)

    # Convert to 8-bit PCM
    min_val, max_val = np.min(resampled_data), np.max(resampled_data)
    # Scale the data to 8-bit range
    resampled_data = np.clip((resampled_data - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)
    
    return resampled_data.tobytes()
    
class CallHandler:
    def __init__(self) -> None:
        self.caller_id=uuid.uuid4()
        self.chunks_to_skip = SILENCE_CHUNKS + 50 # Number of initial chunks to skip
        self.skipped_chunks = 0
        self.skipped_bytes = 0
        self.speech_threshold = 0.0
        self.voiced_chunk_count = 0
        self.silence_count = 0
        self.prob_data = []
        self.vad_dictionary = {}
        self.buffer_lock = threading.Lock()
        self.audio_buffer = io.BytesIO()  # BytesIO object that will be used to store the audio bytes that the user speaks
        self.silence_found=False
        self.last_position=0
        self.asr_queue=queue.Queue()
        self.llm_queue=queue.Queue()
        self.tts_queue=queue.Queue()
        self.audio_sender_queue=queue.Queue()
        self.state=1
        self.out_state=4

        self.replace_track=False # Flag to indicate that track has to be replaced inside send_audio_back co-routine
        self.audio_array=b'' # an array that holds audio that has to be streamed back to the Client
        self.out_stream_status=False # Flag to indicate that server is streaming 
        self.stream_chunks=[]

        self.played_reminder=False # Flag to indicate that reminder-audio has to be played
        self.logs={}
        self.caller=False

        self.call_logs_url='http://localhost:8001/call_logs/'

    def call_hangup(self, call):
        while True:
            time.sleep(0.0001)
            if self.state==4:
                time.sleep(8)
                self.logs['Hangup']=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                requests.post(self.call_logs_url, params={'caller_id': self.caller_id,'event_type': 'Call Hangup','event_detail': ''}, headers={'accept': 'application/json'})   
                call.hangup()
                with open('logs/'+datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'.json', 'w', encoding='utf-8') as json_file:
                        json.dump(self.logs, json_file, ensure_ascii=False, indent=4)
                if self.caller:
                    sys.exit()

    def send_audio_back(self, call):
        
        while True:
            
            time.sleep(0.0001)
            if self.audio_sender_queue:
                if not self.audio_sender_queue.empty():  # Check if the queue is empty

                    # print('audio_sender_queue has new item')
                    requests.post(self.call_logs_url, params={'caller_id': self.caller_id,'event_type': 'audio_sender_queue has new item','event_detail': ''}, headers={'accept': 'application/json'})
                    self.audio_array = self.audio_sender_queue.get()
                    self.out_stream_status=True

                    
                    requests.post(self.call_logs_url, params={'caller_id': self.caller_id,'event_type': 'replaced track','event_detail': ''}, headers={'accept': 'application/json'})   
                    self.replace_track=True

                if self.replace_track:
                    # call.write_audio(self.audio_array)
                    self.stream_chunks=split_audio_bytes(self.audio_array)

                    for i in self.stream_chunks:
                        time.sleep(1)
                        call.write_audio(i)

                        if self.out_stream_status==True and self.voiced_chunk_count>=3:
                            print(self.voiced_chunk_count)
                            print("Detected interruption --> Stream Silence")
                            self.stream_chunks.clear()
                            self.out_stream_status=False
                            requests.post(self.call_logs_url, params={'caller_id': self.caller_id,'event_type': 'interrupt detected','event_detail': ''}, headers={'accept': 'application/json'})   
                            self.logs['Detected interruption']=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    self.replace_track=False

                    # with open('logs/'+datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'.json', 'w', encoding='utf-8') as json_file:
                    #     json.dump(self.logs, json_file, ensure_ascii=False, indent=4)
            # # To Handle Interrupt
            # if self.voiced_chunk_count >= 5 and self.out_stream_status==True:
            #     print(self.voiced_chunk_count)
            #     print("Detected interruption --> Stream Silence")
            #     audio_sender.replaceTrack(AudioStreamTrack())

            #     self.logs['Interrupt detected Streaming silence'+str(uuid.uuid4())]=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            #     self.out_stream_status=False
            
            # # To Handle Reminder that Client hasn't spoken for 6 seconds before we disconnect client 
            # if not self.played_reminder and self.silence_count>=REMINDER_CHUNKS:
            #     print("Play Reminder audio")
                
            #     numpy_track=NumpyAudioStreamTrack(warning_array, add_silence=True)
            #     # reminder_track=BytesIOAudioStreamTrack(io.BytesIO(audio_data))
            #     audio_sender.replaceTrack(numpy_track)
            #     self.logs['Play Reminder Audio'+str(uuid.uuid4())]=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
            #     self.played_reminder=True

            # if self.played_reminder:
            #     if self.silence_count>=(REMINDER_CHUNKS+DISCONNECT_CHUNKS):
            #         print("Disconnect Client")

            #         # Save to a JSON file
            #         with open('logs/'+self.client_id+'.json', 'w') as json_file:
            #             json.dump(self.logs, json_file)

            #         self.logs['Disconnected'+str(uuid.uuid4())]=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            #         break

            #     if self.voiced_chunk_count>0:
            #         self.played_reminder=False
            
            # if self.state==self.out_state:
            #     time.sleep(7)
            #     # await pc.close()
            #     # pcs.discard(pc)
            #     # clients.pop(client_id, None)

    def tts(self):

        while True:
            
            time.sleep(0.0001)
            if self.tts_queue and not self.tts_queue.empty():
                print("tts queue has new item")
                requests.post(self.call_logs_url, params={'caller_id': self.caller_id,'event_type': 'tts queue has new item','event_detail': ''}, headers={'accept': 'application/json'})   
                    
                llm_output = self.tts_queue.get() # Get data from the queue
                
                if self.state==1:
                    if llm_output=='yes':
                        audio_array=get_audio_bytes('/home/oem/webrtc/src/callbot/audios/isp2_yes (2).wav')
                        self.logs['TTS Plays: isp2_yes.wav']=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        requests.post(self.call_logs_url, params={'caller_id': self.caller_id,'event_type': 'tts pushes','event_detail': 'isp2_yes.wav'}, headers={'accept': 'application/json'})   
                        self.state+=1
                    elif llm_output=='no':
                        audio_array=get_audio_bytes('/home/oem/webrtc/src/callbot/audios/isp2_no (2).wav')
                        requests.post(self.call_logs_url, params={'caller_id': self.caller_id,'event_type': 'tts pushes','event_detail': 'isp2_no.wav'}, headers={'accept': 'application/json'})
                        self.logs['TTS Plays: isp2_no.wav']=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.state=self.out_state
                    elif llm_output=='repeat':
                        audio_array=get_audio_bytes('/home/oem/webrtc/src/callbot/audios/isp1 (2).wav')
                        self.logs['TTS Plays: isp1.wav']=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        requests.post(self.call_logs_url, params={'caller_id': self.caller_id,'event_type': 'tts pushes','event_detail': 'isp1.wav'}, headers={'accept': 'application/json'})
                        self.state=1
                elif self.state==2:
                    if llm_output=='yes':
                        audio_array=get_audio_bytes('/home/oem/webrtc/src/callbot/audios/isp3_yes (2).wav')
                        self.logs['TTS Plays: isp3_yes.wav']=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        requests.post(self.call_logs_url, params={'caller_id': self.caller_id,'event_type': 'tts pushes','event_detail': 'isp3_yes.wav'}, headers={'accept': 'application/json'})
                        self.state=self.out_state
                    elif llm_output=='no':
                        audio_array=get_audio_bytes('/home/oem/webrtc/src/callbot/audios/isp3_no (2).wav')
                        self.logs['TTS Plays: isp3_no.wav']=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        requests.post(self.call_logs_url, params={'caller_id': self.caller_id,'event_type': 'tts pushes','event_detail': 'isp3_no.wav'}, headers={'accept': 'application/json'})
                        self.state+=1
                    elif llm_output=='repeat':
                        audio_array=get_audio_bytes('/home/oem/webrtc/src/callbot/audios/isp2_yes (2).wav')
                        self.logs['TTS Plays: isp2_yes.wav']=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        requests.post(self.call_logs_url, params={'caller_id': self.caller_id,'event_type': 'tts pushes','event_detail': 'isp2_yes.wav'}, headers={'accept': 'application/json'})
                        self.state=2
                elif self.state==3:
                    if llm_output=='yes':
                        audio_array=get_audio_bytes('/home/oem/webrtc/src/callbot/audios/isp4_yes (2).wav')
                        self.logs['TTS Plays: isp4_yes.wav']=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        requests.post(self.call_logs_url, params={'caller_id': self.caller_id,'event_type': 'tts pushes','event_detail': 'isp4_yes.wav'}, headers={'accept': 'application/json'})
                        self.state+=1
                    elif llm_output=='no':
                        audio_array=get_audio_bytes('/home/oem/webrtc/src/callbot/audios/isp4_no (2).wav')
                        requests.post(self.call_logs_url, params={'caller_id': self.caller_id,'event_type': 'tts pushes','event_detail': 'isp4_no.wav'}, headers={'accept': 'application/json'})
                        self.logs['TTS Plays: isp4_no.wav']=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.state=self.out_state
                    elif llm_output=='repeat':
                        audio_array=get_audio_bytes('/home/oem/webrtc/src/callbot/audios/isp3_no (2).wav')
                        self.logs['TTS Plays: isp3_yes.wav']=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        requests.post(self.call_logs_url, params={'caller_id': self.caller_id,'event_type': 'tts pushes','event_detail': 'isp3_yes.wav'}, headers={'accept': 'application/json'})
                        self.state=3
                elif self.state==4:
                    # if llm_output=='yes':
                    audio_array=get_audio_bytes('/home/oem/webrtc/src/callbot/audios/isp5 (2).wav')
                    self.logs['TTS Plays: isp5_yes.wav']=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    requests.post(self.call_logs_url, params={'caller_id': self.caller_id,'event_type': 'tts pushes','event_detail': 'isp5.wav'}, headers={'accept': 'application/json'})
                print('text converted to audio',self.state)
                requests.post(self.call_logs_url, params={'caller_id': self.caller_id,'event_type': 'TTS pushed audio to audio_sender_queue','event_detail': ''}, headers={'accept': 'application/json'})   
                # print()

                self.logs['TTSOutput'+str(uuid.uuid4())+'.wav']=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                self.audio_sender_queue.put(audio_array)
                print('pushed audio array to audio_sender_queue')

    def llm(self):

        while True:
            
            time.sleep(0.0001)
            if self.llm_queue and not self.llm_queue.empty():
                requests.post(self.call_logs_url, params={'caller_id': self.caller_id,'event_type': 'llm_queue has new item','event_detail': ''}, headers={'accept': 'application/json'})   
                    
                text = self.llm_queue.get() # Get data from the queue

                try:
                    text_embedding =  get_embedding(text)
                    # print(text_embedding)
                except:
                    print('£'*100)
                    print("Could not get text embedding")
                    requests.post(self.call_logs_url, params={'caller_id': self.caller_id,'event_type': 'couldn\'t get text embedding','event_detail': ''}, headers={'accept': 'application/json'})   

                # out_state = self.state+1 # Out state increase by 1

                # print("&"*100)
                # print(np.array(text_embedding).shape)
                # print(np.array(positive_embeds[0]).shape)
                # print("&"*100)

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
                self.logs['LLMOutput: '+ response]=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                requests.post(self.call_logs_url, params={'caller_id': self.caller_id,'event_type': 'LLMOutput','event_detail': response}, headers={'accept': 'application/json'})   
                    
                self.tts_queue.put(response)

    # def save_audio_to_file(audio_chunks, output_file):
    #     with wave.open(output_file, 'wb') as wf:
    #         # Parameters for WAV file
    #         num_channels = 1
    #         sample_width = 2
    #         frame_rate = 8000
    #         num_frames = sum(len(chunk) for chunk in audio_chunks) // sample_width
            
    #         wf.setnchannels(num_channels)
    #         wf.setsampwidth(sample_width)
    #         wf.setframerate(frame_rate)
    #         wf.setnframes(num_frames)
            
    #         # Write audio data to the file
    #         for chunk in audio_chunks:
    #             wf.writeframes(chunk)

    def asr(self):

            asr_base_url='http://192.168.88.10:8028/transcribe_sip_en'

            while True:
                
                time.sleep(0.0001)
                # print(self.vad_dictionary)
                if self.asr_queue:
                    if not self.asr_queue.empty():  # Check if the queue is empty
                        print('audiobytes added to asr queue')
                        requests.post(self.call_logs_url, params={'caller_id': self.caller_id,'event_type': 'asr_queue has new item','event_detail': ''}, headers={'accept': 'application/json'})   
                    

                        audio_bytes = self.asr_queue.get() # Get data from the queue
                        # print(audio_bytes)
                        # self.asr_queue.task_done()
                        
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
                        requests.post(self.call_logs_url, params={'caller_id': self.caller_id,'event_type': 'ASROutput','event_detail': asr_output_text}, headers={'accept': 'application/json'})   
                    
                        self.logs['ASROutput: '+asr_output_text]=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        self.llm_queue.put(asr_output_text)

    def read_vad_dictionary(self):
            print('Reading VAD Dictionary')
            # global silence_found, vad_dictionary, voiced_chunk_count, asr_queue
            c=0 # counter for saving audio chunks as file temporarily

            while True:
                time.sleep(0.0001)
                # print('Reading VAD Dictionary in iteration')
                # print('in read_vad_dictionary',.vad_dictionary)
                # check if there's number of chunks in vad_dictionary with all consecutive silence chunks then pop the chunks samples
                if not self.silence_found:
                    if self.vad_dictionary and all(i==0 for i in list(self.vad_dictionary.values())[- SILENCE_CHUNKS:]):
                        
                        popped_chunks=list(self.vad_dictionary.keys())[:]

                        # # Filter out silence chunks before joining chunks to get only chunks that has audio
                        # audio_chunks = [k for k, v in .vad_dictionary.items() if v != 0]
                        
                        print('popped chunks',len(popped_chunks))
                        audio_bytes=b''.join(popped_chunks)

                        for i in popped_chunks:
                            del self.vad_dictionary[i]
                        
                        # Save the audio bytes as a .wav file to test if VAD is working correctly
                        with wave.open('vad_output/myaudio'+str(c)+'.wav', 'wb') as wf:
                            wf.setnchannels(1)  # Assuming mono audio
                            wf.setsampwidth(2)  # Assuming 16-bit audio
                            wf.setframerate(8000)  # Assuming a sample rate of 8kHz
                            wf.writeframes(audio_bytes)
                        
                        self.voiced_chunk_count=0

                        c+=1
                        self.silence_found=True
                        print("Silence Found")
                        self.logs['Silence Found:']=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        requests.post(self.call_logs_url, params={'caller_id': self.caller_id,'event_type': 'Silence Found','event_detail': ''}, headers={'accept': 'application/json'})   
                        self.asr_queue.put(audio_bytes)
                else:
                    if self.vad_dictionary and list(self.vad_dictionary.values())[-1]==1:
                        print("Detected Voice")
                        self.logs['Detected Voice:']=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        requests.post(self.call_logs_url, params={'caller_id': self.caller_id,'event_type': 'detected voice','event_detail': ''}, headers={'accept': 'application/json'})   
                    
                        self.silence_found=False


    def VAD(self, chunk, threshold_weight=0.9):
        # global voiced_chunk_count, silence_count, speech_threshold, prob_data, vad_dictionary
        # print('Vad Function Started')
        np_chunk = np.frombuffer(chunk, dtype=np.int16)
        np_chunk = np_chunk.astype(np.float32) / 32768.0
        chunk_audio = torch.from_numpy(np_chunk)

        self.speech_prob = model(chunk_audio, SAMPLE_RATE).item()
        self.prob_data.append(self.speech_prob)

        if self.speech_prob >= self.speech_threshold:
            self.voiced_chunk_count += 1
            self.silence_count = 0
            self.vad_dictionary[chunk] = 1
        else:
            self.vad_dictionary[chunk] = 0
            self.silence_count += 1

        # print(vad_dictionary)
        if self.prob_data:
            self.speech_threshold = threshold_weight * max([i**2 for i in self.prob_data]) + (1 - threshold_weight) * min([i**2 for i in self.prob_data])

    def read_buffer_chunks(self):
        
        # global voiced_chunk_count, silence_count, vad_dictionary, prob_data, audio_buffer, last_position
        while True:
                
                time.sleep(0.0001)
                self.audio_buffer.seek(0, io.SEEK_END)  # seek to end of audio
                size = self.audio_buffer.tell()  # size of audio

                if size >= self.last_position + CHUNK_SIZE:
                # if size >= CHUNK_SIZE:
                    self.audio_buffer.seek(self.last_position)  # Seek to the last read position

                    if self.skipped_chunks < self.chunks_to_skip:
                        # Skip the specified number of chunks
                        self.skipped_bytes = min(self.chunks_to_skip * CHUNK_SIZE, size - self.last_position)
                        self.audio_buffer.seek(self.last_position + self.skipped_bytes)
                        self.last_position += self.skipped_bytes
                        self.skipped_chunks += self.skipped_bytes // CHUNK_SIZE
                        continue  # proceed remaining loop

                    # audio_buffer.seek(0)  # Seek to the last read position
                    chunk = self.audio_buffer.read(CHUNK_SIZE)  # Read the next chunk of data
                    # print(chunk)
                    # Implement VAD in this chunk
                    self.VAD(chunk)

                    self.last_position += CHUNK_SIZE

                    # audio_buffer.seek(0)
                    # audio_buffer.truncate()
                    # Truncate the buffer to remove processed data
                    # remaining_data = audio_buffer.read()  # Read the rest of the buffer
                    # audio_buffer.seek(0)  # Move to the start
                    # audio_buffer.truncate()  # Clear the buffer
                    # audio_buffer.write(remaining_data)  # Write back the remaining data

    # # # Function to save audio_chunks to a file
    # def save_audio_to_file(audio_chunks, output_file):
    #     with wave.open(output_file, 'wb') as wf:
    #         # Parameters for WAV file
    #         num_channels = 1
    #         sample_width = 2
    #         frame_rate = 8000
    #         num_frames = sum(len(chunk) for chunk in audio_chunks) // sample_width
            
    #         wf.setnchannels(num_channels)
    #         wf.setsampwidth(sample_width)
    #         wf.setframerate(frame_rate)
    #         wf.setnframes(num_frames)
            
    #         # Write audio data to the file
    #         for chunk in audio_chunks:
    #             wf.writeframes(chunk)

    def handle_call(self, call):
        # global audio_buffer
        try:
            call.answer()
            requests.post(self.call_logs_url, params={'caller_id': self.caller_id,'event_type': 'Call Answered','event_detail': ''}, headers={'accept': 'application/json'})
            call.write_audio(get_audio_bytes('/home/oem/webrtc/src/callbot/audios/isp1.wav'))
            threading.Thread(target=self.read_buffer_chunks, daemon=True).start()
            threading.Thread(target=self.read_vad_dictionary, daemon=True).start()
            threading.Thread(target=self.asr, daemon=True).start()
            threading.Thread(target=self.llm, daemon=True).start()
            threading.Thread(target=self.tts, daemon=True).start()
            threading.Thread(target=self.send_audio_back, daemon=True, args=(call,)).start()
            threading.Thread(target=self.call_hangup, daemon=True, args=(call,)).start()

            while call.state == CallState.ANSWERED:
                audio_bytes = call.read_audio()
                # print(audio_bytes)
                self.audio_buffer.write(convert_8bit_to_16bit(audio_bytes))
                

        except Exception as e:
            print(e)
            
        finally:
            call.hangup()