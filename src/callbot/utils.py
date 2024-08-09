import wave, torch, torchaudio
import numpy as np

# Global variables that stores peer connections and clients that are connected
pcs = set() # Peer Connections
clients={} # {client_id: Client() Object} dictionary

def getAudioArray(path, sample_rate, channels):
    with wave.open(path, 'rb') as wav_file:
        num_frames = wav_file.getnframes()
        audio_data = wav_file.readframes(num_frames)

    np_array=np.frombuffer(audio_data, dtype=np.int16)
    np_array=np_array.astype(np.float32) /32768.0

    if channels==2:
        return np_array
    elif channels==1:
        audio_tensor = torch.from_numpy(np_array)
        audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, 48000)
        # audio_tensor = torch.stack([audio_tensor, audio_tensor], dim=0)
        return audio_tensor.numpy()


welcome_array=getAudioArray("./audios/welcome_shuvani.wav",sample_rate=22050,channels=1)
warning_array=getAudioArray("./audios/warning.wav",sample_rate=48000, channels=2)

state1_array=getAudioArray("./audios/global_ime_1.wav",sample_rate=24000, channels=1)
state2_yes_array=getAudioArray("./audios/global_ime_2_yes.wav",sample_rate=24000, channels=1)
state2_no_array=getAudioArray("./audios/global_ime_2_no.wav",sample_rate=24000, channels=1)
state3_yes_array=getAudioArray("./audios/global_ime_3_yes.wav",sample_rate=24000, channels=1)
state3_no_array=getAudioArray("./audios/global_ime_3_no.wav",sample_rate=24000, channels=1)
state4_yes_array=getAudioArray("./audios/global_ime_4_yes.wav",sample_rate=24000, channels=1)
state4_no_array=getAudioArray("./audios/global_ime_4_no.wav",sample_rate=24000, channels=1)
state5_array=getAudioArray("./audios/global_ime_5_yes.wav",sample_rate=24000, channels=1)

# # Welcome Audio that has to be played in the starting
# with wave.open("./audios/welcome_shuvani.wav", 'rb') as wav_file:
#     num_frames = wav_file.getnframes()
#     audio_data = wav_file.readframes(num_frames)

# welcome_array=np.frombuffer(audio_data,dtype=np.int16)
# welcome_array=welcome_array.astype(np.float32) /32768.0
# welcome_tensor = torch.from_numpy(welcome_array)
# welcome_data = torchaudio.functional.resample(welcome_tensor, 22050, 48000)
# stereo_welcome = torch.stack([welcome_data, welcome_data], dim=0)
# welcome_array=stereo_welcome.numpy()


# # Warning Audio that has to be played if client doesn't speak for 12 seconds
# with wave.open("./audios/warning.wav", 'rb') as wav_file:
#     num_frames = wav_file.getnframes()
#     audio_data = wav_file.readframes(num_frames)

# np_array=np.frombuffer(audio_data, dtype=np.int16)
# warning_array=np_array.astype(np.float32) /32768.0


# # Warning Audio that has to be played if client doesn't speak for 12 seconds
# with wave.open("./audios/global_ime_1.wav", 'rb') as wav_file:
#     num_frames = wav_file.getnframes()
#     audio_data = wav_file.readframes(num_frames)

# np_array=np.frombuffer(audio_data, dtype=np.int16)
# np_array=np_array.astype(np.float32) /32768.0
# audio_tensor = torch.from_numpy(np_array)
# audio_tensor = torchaudio.functional.resample(audio_tensor, 24000, 48000)
# audio_tensor = torch.stack([audio_tensor, audio_tensor], dim=0)
# audio_array=stereo_welcome.numpy()