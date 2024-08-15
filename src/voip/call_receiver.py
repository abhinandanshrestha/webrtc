from pyVoIP.VoIP import VoIPPhone, CallState
import wave
import uuid
import io
import numpy as np
import torch
import math
import threading
import requests
import queue
import datetime
# Loading model for VAD
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
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

speech_threshold = 0.0
voiced_chunk_count = 0
silence_count = 0
prob_data = []
vad_dictionary = {}
buffer_lock = threading.Lock()
audio_buffer = io.BytesIO()  # BytesIO object that will be used to store the audio bytes that the user speaks
silence_found=False
last_position=0
asr_queue=queue.Queue()
llm_queue=queue.Queue()

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

def asr():

        asr_base_url='http://192.168.88.10:8028/transcribe_sip'

        while True:

            if asr_queue:
                if not asr_queue.empty():  # Check if the queue is empty
                    print('audiobytes added to asr queue')

                    audio_bytes = asr_queue.get() # Get data from the queue
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

                    # self.logs['ASROutput: '+asr_output_text]=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    llm_queue.put(asr_output_text)

def read_vad_dictionary():
        print('Reading VAD Dictionary')
        global silence_found, vad_dictionary, voiced_chunk_count, asr_queue
        c=0 # counter for saving audio chunks as file temporarily

        while True:
            # print('Reading VAD Dictionary in iteration')
            # print('in read_vad_dictionary',.vad_dictionary)
            # check if there's number of chunks in vad_dictionary with all consecutive silence chunks then pop the chunks samples
            if not silence_found:
                if vad_dictionary and all(i==0 for i in list(vad_dictionary.values())[- SILENCE_CHUNKS:]):
                    
                    popped_chunks=list(vad_dictionary.keys())[:]

                    # # Filter out silence chunks before joining chunks to get only chunks that has audio
                    # audio_chunks = [k for k, v in .vad_dictionary.items() if v != 0]
                    
                    print('popped chunks',len(popped_chunks))
                    audio_bytes=b''.join(popped_chunks)

                    for i in popped_chunks:
                        del vad_dictionary[i]
                    
                    # Save the audio bytes as a .wav file to test if VAD is working correctly
                    # with wave.open('myaudio'+str(c)+'.wav', 'wb') as wf:
                    #     wf.setnchannels(1)  # Assuming mono audio
                    #     wf.setsampwidth(2)  # Assuming 16-bit audio
                    #     wf.setframerate(8000)  # Assuming a sample rate of 8kHz
                    #     wf.writeframes(audio_bytes)
                    
                    voiced_chunk_count=0

                    c+=1
                    silence_found=True
                    asr_queue.put(audio_bytes)
                    print("Silence Found")

            else:
                if vad_dictionary and list(vad_dictionary.values())[-1]==1:
                    print("Detected Voice")
                    silence_found=False


def VAD(chunk, threshold_weight=0.9):
    global voiced_chunk_count, silence_count, speech_threshold, prob_data, vad_dictionary
    # print('Vad Function Started')
    np_chunk = np.frombuffer(chunk, dtype=np.int16)
    np_chunk = np_chunk.astype(np.float32) / 32768.0
    chunk_audio = torch.from_numpy(np_chunk)

    speech_prob = model(chunk_audio, SAMPLE_RATE).item()
    prob_data.append(speech_prob)

    if speech_prob >= speech_threshold:
        voiced_chunk_count += 1
        silence_count = 0
        vad_dictionary[chunk] = 1
    else:
        vad_dictionary[chunk] = 0
        silence_count += 1

    # print(vad_dictionary)
    if prob_data:
        speech_threshold = threshold_weight * max([i**2 for i in prob_data]) + (1 - threshold_weight) * min([i**2 for i in prob_data])

def read_buffer_chunks():
    global voiced_chunk_count, silence_count, vad_dictionary, prob_data, audio_buffer, last_position
    while True:
            audio_buffer.seek(0, io.SEEK_END)  # seek to end of audio
            size = audio_buffer.tell()  # size of audio

            if size >= last_position + CHUNK_SIZE:
            # if size >= CHUNK_SIZE:
                audio_buffer.seek(last_position)  # Seek to the last read position
                # audio_buffer.seek(0)  # Seek to the last read position
                chunk = audio_buffer.read(CHUNK_SIZE)  # Read the next chunk of data

                # Implement VAD in this chunk
                VAD(chunk)

                last_position += CHUNK_SIZE

                # audio_buffer.seek(0)
                # audio_buffer.truncate()
                # Truncate the buffer to remove processed data
                # remaining_data = audio_buffer.read()  # Read the rest of the buffer
                # audio_buffer.seek(0)  # Move to the start
                # audio_buffer.truncate()  # Clear the buffer
                # audio_buffer.write(remaining_data)  # Write back the remaining data



def convert_8bit_to_16bit(audio_data):
    audio_data = np.frombuffer(audio_data, dtype=np.uint8).astype(np.int16)
    audio_data = (audio_data - 128) * 256  # Scale from 8-bit to 16-bit
    return audio_data.tobytes()

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

def handle_call(call):
    global audio_buffer
    try:
        # buffer = []
        # buff_length = 0

        call.answer()
        threading.Thread(target=read_buffer_chunks, daemon=True).start()
        threading.Thread(target=read_vad_dictionary, daemon=True).start()
        threading.Thread(target=asr, daemon=True).start()

        while call.state == CallState.ANSWERED:
            audio_bytes = call.read_audio()
            audio_buffer.write(convert_8bit_to_16bit(audio_bytes))

            # buff_length += len(audio_bytes) / 8
            # if buff_length <= 5000:
            #     buffer.append(convert_8bit_to_16bit(audio_bytes))
            # else:
            #     # print(buffer)
            #     save_audio_to_file(buffer, f'original_audio_5sec_each_{uuid.uuid4()}.wav')
            #     buffer = []
            #     buff_length = 0

    except Exception as e:
        print(e)
    finally:
        call.hangup()


def start_voip():
    phone = VoIPPhone(
        server="192.168.88.5",
        port=5060,
        username="5001",
        password="iX3TxsD9jWxmZU5",
        myIP="192.168.88.10",
        callCallback=handle_call
    )

    phone.start()
    print("Phone started. Status:", phone.get_status())

    try:
        while True:
            time.sleep(0.1)  # Keep the main thread running
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        phone.stop()
        print("Phone stopped")

if __name__ == "__main__":
    import time
    start_voip()

