import wave, torch, torchaudio
import numpy as np

# Global variables that stores peer connections and clients that are connected
pcs = set() # Peer Connections
clients={} # {client_id: Client() Object} dictionary


# Welcome Audio that has to be played in the starting
with wave.open("./audios/welcome_shuvani.wav", 'rb') as wav_file:
    num_frames = wav_file.getnframes()
    audio_data = wav_file.readframes(num_frames)

welcome_array=np.frombuffer(audio_data,dtype=np.int16)
welcome_array=welcome_array.astype(np.float32) /32768.0
welcome_tensor = torch.from_numpy(welcome_array)
welcome_data = torchaudio.functional.resample(welcome_tensor, 22050, 48000)
stereo_welcome = torch.stack([welcome_data, welcome_data], dim=0)
welcome_array=stereo_welcome.numpy()


# Warning Audio that has to be played if client doesn't speak for 12 seconds
with wave.open("./audios/warning.wav", 'rb') as wav_file:
    num_frames = wav_file.getnframes()
    audio_data = wav_file.readframes(num_frames)

np_array=np.frombuffer(audio_data, dtype=np.int16)
warning_array=np_array.astype(np.float32) /32768.0