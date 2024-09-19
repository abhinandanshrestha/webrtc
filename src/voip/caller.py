import wave
import time
from pyVoIP.VoIP import VoIPPhone, CallState
from call_handler import CallHandler, convert_8bit_to_16bit, get_audio_bytes
import threading
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--call', type=str, help='Number to call', required=True)

phone = VoIPPhone(
        server="192.168.88.5",
        port=5060,
        username="5001",
        password="iX3TxsD9jWxmZU5",
        myIP="192.168.88.10"
)

phone.start()
callHandler=CallHandler()
callHandler.caller=True
start_thread=False

try:
    args = parser.parse_args()
    call = phone.call(args.call)
    print("Call initiated. Send mode:", call.sendmode)
    # while True:
    #     print(call.state)

    # Wait for the call to be answered
    while call.state != CallState.ANSWERED:
        print("Current call state:", call.state)
    
    # Process audio once the call is answered
    while call.state == CallState.ANSWERED:
        # print("Current call state:", call.state)
        audio_bytes = call.read_audio()
        if audio_bytes:
            callHandler.audio_buffer.write(convert_8bit_to_16bit(audio_bytes))
            if not start_thread:
                call.write_audio(get_audio_bytes('/home/oem/webrtc/src/callbot/audios/global_ime_1.wav'))
                threading.Thread(target=callHandler.read_buffer_chunks, daemon=True).start()
                threading.Thread(target=callHandler.read_vad_dictionary, daemon=True).start()
                threading.Thread(target=callHandler.asr, daemon=True).start()
                threading.Thread(target=callHandler.asr, daemon=True).start()
                threading.Thread(target=callHandler.llm, daemon=True).start()
                threading.Thread(target=callHandler.tts, daemon=True).start()
                threading.Thread(target=callHandler.send_audio_back, daemon=True, args=(call,)).start()
                threading.Thread(target=callHandler.call_hangup, daemon=True, args=(call,)).start()
                start_thread=True

        else:
            print("No audio data received.")
            
    # Handle call end
    if call.state == CallState.ENDED:
        print("Call ended.")

except Exception as e:
    print("An error occurred:", e)

