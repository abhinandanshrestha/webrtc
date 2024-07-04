import asyncio
import json
import requests
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel, RTCConfiguration, RTCIceServer, MediaStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRecorder
import logging
import sys
import uuid

# logging.basicConfig(level=logging.DEBUG)
async def run(client_id):

    client_id=str(uuid.uuid4()) # globally unique id for this client
    print("Client ID:",client_id)

    config = RTCConfiguration(iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")])
    pc = RTCPeerConnection(configuration=config)
    # print(id(pc))
    channel = pc.createDataChannel("chat")

    recorder = MediaRecorder('receivedFromServer_'+client_id+'.wav')

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
        print(f"Track{track.kind} received. Make sure .start() is called to start recording")

        if track.kind == "audio":
            recorder.addTrack(track)
            await recorder.start()
            
        @track.on("ended")
        async def on_ended():
            print("Track %s ended", track.kind)
            await recorder.stop()
            # asyncio.ensure_future(save_audio())
            # await pc.close()

    # # Capture audio from the audiofile and stream for now
    # player = MediaPlayer('./audiotest.wav')
    # audio_track = player.audio

    # # Add audio track to the peer connection
    # pc.addTrack(audio_track)
        # Add audio track to the peer connection
    pc.addTrack(MediaPlayer("audio=Microphone Array (Realtek(R) Audio)",format="dshow").audio)

    # Audio Received from the server will be saved to a file for now
    
    await pc.setLocalDescription(await pc.createOffer())
    sdp_offer = {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "client_id": client_id
    }

    try:
        response = requests.post("http://localhost:8080/offer", data=sdp_offer)
        # print(response)
        if response.status_code == 200:
            answer = response.json()
            # print(answer)
            answer_desc = RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
            await pc.setRemoteDescription(answer_desc)
            while True:
                # print('We are ready to send any data to the server')
                await asyncio.sleep(0.01)
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
