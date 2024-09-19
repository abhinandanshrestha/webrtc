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

    # Wait for the call to be answered
    while call.state != CallState.ANSWERED:
        print("Current call state:", call.state)
    
    # Process audio once the call is answered
    while call.state == CallState.ANSWERED:
        # print("Current call state:", call.state)
        audio_bytes = call.read_audio()
        print(audio_bytes)
        break
            
    # Handle call end
    if call.state == CallState.ENDED:
        print("Call ended.")

except Exception as e:
    print("An error occurred:", e)

