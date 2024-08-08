import argparse
import asyncio
import json
import logging
import os
import ssl
from starlette.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Form, HTTPException
from starlette.middleware.cors import CORSMiddleware
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, MediaStreamTrack, AudioStreamTrack
import gc
from media_classes import BufferMediaRecorder
from client_handler import Client
from utils import pcs, clients

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()

app = FastAPI()
app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)

@app.get("/")
async def index():
    return {
        "clients":list(clients.keys()),
        "client_logs":[{ client.client_id : client.logs}  for client in list(clients.values())]
    }

# endpoint to accept offer from webrtc client for handshaking
@app.post("/offer")
async def offer_endpoint(sdp: str = Form(...), type: str = Form(...), client_id: str = Form(...)):
    # print(sdp,type,client_id)
    # logging.info(f"Received SDP: {sdp}, type: {type}")
    config = RTCConfiguration(iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")]) # make use of google's stun server
    pc = RTCPeerConnection(configuration=config) # pass the config to configuration to make use of stun server
    pcs.add(client_id) # add peer connection to set of peer connections

    client=Client(client_id)
    clients[client_id] = client  # Add the client to the clients dictionary, which can be cleaned up once the client disconnects

    recorder = BufferMediaRecorder(client.audio_buffer)

    # event handler for data channel
    @pc.on("datachannel")
    def on_datachannel(channel):
        # client_datachannels[client_id]=channel # to make datachannel accessible outside of this scope
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
            # audio_sender=pc.addTrack(NumpyAudioStreamTrack(welcome_array, add_silence=True))
            # print(welcome_array)
            # asyncio.ensure_future(recorder.start())
            asyncio.ensure_future(client.start_recorder(recorder))
            asyncio.ensure_future(client.read_buffer_chunks())
            asyncio.ensure_future(client.read_vad_dictionary())
            asyncio.ensure_future(client.asr())
            asyncio.ensure_future(client.llm())
            asyncio.ensure_future(client.tts())
            asyncio.ensure_future(client.send_audio_back(audio_sender, client_id, pc))
            
            # Start coroutine to handle interrupts
            # for example: if audio is streaming back to client and client speaks in the middle, replaceTrack(AudioStreamTrack()) with silence
            # asyncio.ensure_future(client.send_audio_back(audio_sender))

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
            print("Deleting client_id:", client_id)

            with open('logs/'+client_id+'.json', 'w') as json_file:
                json.dump(client.logs, json_file)

            await recorder.stop()

            clients.pop(client_id, None)  # Remove the client from the clients dictionary
            pcs.discard(pc)  # Remove the pc from the set of peer connections

            await pc.close()

            # Manually trigger garbage collection
            collected = gc.collect()

            # Verify memory release
            print(f"Garbage collector collected {collected} objects.")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8027, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file.")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    ssl_context = None
    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)

    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ssl_keyfile=args.key_file,
        ssl_certfile=args.cert_file,
        log_level="info" if not args.verbose else "debug",
    )


# @app.post("/offer", response_class=JSONResponse)
# async def offer(sdp: str = Form(...), type: str = Form(...), client_id: str = Form(...)):
#     offer = RTCSessionDescription(sdp=sdp, type=type)
#     print(client_id)

#     pc = RTCPeerConnection()
#     pc_id = client_id
#     pcs.add(pc)

#     def log_info(msg, *args):
#         logger.info(pc_id + " " + msg, *args)

#     log_info("Created for offer")

#     # prepare local media
#     player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
#     if args.record_to:
#         recorder = MediaRecorder(args.record_to)
#     else:
#         recorder = MediaBlackhole()

#     @pc.on("datachannel")
#     def on_datachannel(channel):
#         @channel.on("message")
#         def on_message(message):
#             if isinstance(message, str) and message.startswith("ping"):
#                 channel.send("pong" + message[4:])

#     @pc.on("connectionstatechange")
#     async def on_connectionstatechange():
#         log_info("Connection state is %s", pc.connectionState)
#         if pc.connectionState == "failed":
#             await pc.close()
#             pcs.discard(pc)

#     @pc.on("track")
#     def on_track(track):
#         log_info("Track %s received", track.kind)

#         if track.kind == "audio":
#             pc.addTrack(player.audio)
#             recorder.addTrack(track)

#         @track.on("ended")
#         async def on_ended():
#             log_info("Track %s ended", track.kind)
#             await recorder.stop()

#     # handle offer
#     await pc.setRemoteDescription(offer)
#     await recorder.start()

#     # send answer
#     answer = await pc.createAnswer()
#     await pc.setLocalDescription(answer)

#     return JSONResponse(
#         content={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
#     )



