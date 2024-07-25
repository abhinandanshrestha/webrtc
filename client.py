import asyncio
import json
import requests
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel, RTCConfiguration, RTCIceServer, MediaStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRecorder
import logging
import sys
import uuid
import av, io
import wave

# Create a child class to MediaRecorder class to record audio data to a buffer instead of a file
class BufferMediaRecorder(MediaRecorder):
    """
    A subclass of MediaRecorder that supports using BytesIO buffer as output.
    
    :param buffer: The buffer containing audio data as a BytesIO object.
    """
    def __init__(self, buffer, format="wav"):
        self.__container = av.open(buffer, format=format, mode="w") 
        self.__tracks = {} 
        super().__init__(buffer, format=format) 


# logging.basicConfig(level=logging.DEBUG)
async def run(client_id):

    client_id=str(uuid.uuid4()) # globally unique id for this client
    print("Client ID:",client_id)

    config = RTCConfiguration(iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")])
    pc = RTCPeerConnection(configuration=config)

    # print(id(pc))
    channel = pc.createDataChannel("chat")

    recorder = MediaRecorder('receivedFromServer_'+client_id+'.wav')
    # audio_buffer = io.BytesIO()
    # recorder = BufferMediaRecorder(audio_buffer)

    @channel.on("open")
    def on_open():
        print(f"Channel opened for client {client_id}")
        # for i in range(100):
        channel.send(f'hello I\'m client {client_id}')
        
    @channel.on("message")
    def on_message(message):
        print(f"Received via RTC Datachannel for client {client_id}: ", message)

    @pc.on("track")
    async def on_track(track):
        print('hello abhinandan')
        print(f"Track{track.kind} received. Make sure .start() is called to start recording")

        if track.kind == "audio":
            recorder.addTrack(track)
            await recorder.start() # start recording to buffer named audio_buffer
            
        @track.on("ended")
        async def on_ended():
            print("Track %s ended", track.kind)
            await recorder.stop()
            # asyncio.ensure_future(save_audio())
            # await pc.close()


    # If we want to stream directly from the Microphone, we can simply pass "audio= <Microphone device name and ffmpeg compatible format>" to MediaPlayer
    # # Capture audio from the audiofile and stream for now
    player = MediaPlayer('test-audios/8.wav')
    # player = MediaPlayer('audiotest.wav')
    audio_track = player.audio

    # Add audio track to the peer connection
    pc.addTrack(audio_track)
    # Add audio track to the peer connection
    # pc.addTrack(MediaPlayer("audio=Microphone Array (Realtek(R) Audio)",format="dshow").audio)

    # Audio Received from the server will be saved to a file for now
    await pc.setLocalDescription(await pc.createOffer())
    sdp_offer = {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "client_id": client_id
    }

    try:
        response = requests.post("http://localhost:8081/offer", data=sdp_offer)
        # print(response)
        if response.status_code == 200:
            answer = response.json()
            # print(answer)
            answer_desc = RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
            await pc.setRemoteDescription(answer_desc)
            while True:
                
                await asyncio.sleep(0.1)
                # print('we can save content of buffer to a file')
                # bytes_content = audio_buffer.getvalue() # Get entire content of BytesIO object
                # print(audio_buffer.tell())
                # output_audio_file = 'receivedFromServer'+client_id+'.wav'

                # # Save content as an audio file
                # # Note that MediaRecorder will always get audio_date of sample_width of 2, channels =2 and framerate = 48000
                # with wave.open(output_audio_file, "wb") as audio_file:
                #     n_channels = 2 # Number of channels
                #     sampwidth = 2  # Sample width in bytes (e.g., 2 bytes for 16-bit audio)
                #     framerate = 48000  # Frame rate (samples per second)
                #     n_frames = len(bytes_content) // sampwidth

                #     audio_file.setnchannels(n_channels)
                #     audio_file.setsampwidth(sampwidth)
                #     audio_file.setframerate(framerate)
                #     audio_file.writeframes(bytes_content)

                # print(sdp_offer)
                
        else:
            logging.error("Failed to get SDP answer: %s", response.content)
    except Exception as e:
        print(e)
        # logging.error("Error during SDP offer/answer exchange: %s", e)
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python client.py <client_id>")
        sys.exit(1)
    client_id = sys.argv[1]
    asyncio.run(run(client_id))
