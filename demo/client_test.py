import asyncio
import json
import requests
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, MediaStreamTrack, AudioStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRecorder
import logging
import sys
import uuid

# logging.basicConfig(level=logging.DEBUG)
async def run(client_id):

    client_id = str(uuid.uuid4())  # globally unique id for this client
    print("Client ID:", client_id)

    config = RTCConfiguration(iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")])
    pc = RTCPeerConnection(configuration=config)

    # recorder=MediaRecorder('abhindan-test.wav')

    channel = pc.createDataChannel("chat")

    # Add audio track to the peer connection
    player=MediaPlayer('../audiotest.wav')
    track=player.audio
    pc.addTrack(track)
    # audio_sender=pc.getSenders()[0]

    @channel.on("open")
    def on_open():
        print(f"Channel opened for client {client_id}")
        channel.send(f'hello I\'m client {client_id}')

    @channel.on("message")
    def on_message(message):
        print(f"Received via RTC Datachannel for client {client_id}: ", message)

    # @pc.on("track")
    # async def on_track(track):
    #     print('Hello Abhinandan')
    #     print(f"Track{track.kind} received. Make sure .start() is called to start recording")

    #     if track.kind == "audio":
    #         recorder.addTrack(track)
    #         await recorder.start() # start recording to buffer named audio_buffer
            
    #     @track.on("ended")
    #     async def on_ended():
    #         print("Track %s ended", track.kind)
    #         await recorder.stop()

    # silenceTrack=AudioStreamTrack()

    # player=MediaPlayer('./audiotest.wav')
    # track=player.audio
    

    await pc.setLocalDescription(await pc.createOffer())
    sdp_offer = {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "client_id": client_id
    }

    try:
        response = requests.post("http://localhost:8082/offer", data=sdp_offer)
        if response.status_code == 200:
            answer = response.json()
            answer_desc = RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
            await pc.setRemoteDescription(answer_desc)

            while True:
                await asyncio.sleep(1)

        else:
            logging.error("Failed to get SDP answer: %s", response.content)
    except Exception as e:
        print(e)
        logging.error("Error during SDP offer/answer exchange: %s", e)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python client.py <client_id>")
        sys.exit(1)
    client_id = sys.argv[1]
    asyncio.run(run(client_id))