import asyncio
import json
import requests
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaPlayer
import logging
import sys

# logging.basicConfig(level=logging.DEBUG)
async def run(client_id):
    config = RTCConfiguration(iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")])
    pc = RTCPeerConnection(configuration=config)
    print(id(pc))
    channel = pc.createDataChannel("chat")

    @channel.on("open")
    def on_open():
        print(f"Channel opened for client {client_id}")
        # for i in range(100):
        channel.send(f'hello I\'m client {client_id}')
        
    @channel.on("message")
    def on_message(message):
        print(f"Received via RTC Datachannel for client {client_id}: ", message)

    # Capture audio from the audiofile and stream for now
    player = MediaPlayer('./audiotest.wav')
    audio_track = player.audio

    # Add audio track to the peer connection
    pc.addTrack(audio_track)

    await pc.setLocalDescription(await pc.createOffer())
    sdp_offer = {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "client_id": id(pc)
    }

    # print(sdp_offer)

    try:
        response = requests.post("http://localhost:8000/offer", data=sdp_offer)
        # print(response)
        if response.status_code == 200:
            answer = response.json()
            # print(answer)
            answer_desc = RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
            await pc.setRemoteDescription(answer_desc)
            while True:
                print('We are ready to send any data to the server')
                await asyncio.sleep(5)
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