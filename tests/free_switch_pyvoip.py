
from pyVoIP.VoIP import VoIPPhone, CallState
import pyVoIP
import wave
import uuid
import numpy as np


pyVoIP.DEBUG = True

def start_voip():
    # phone = VoIPPhone(
    #     server="192.168.88.51",
    #     port=5060,
    #     username="4001",
    #     password="123456789",
    #     myIP="192.168.88.10",
    #     # callCallback=handle_call
    # )

    phone = VoIPPhone(
        server="192.168.88.5",
        port=5060,
        username="5001",
        password="iX3TxsD9jWxmZU5",
        myIP="192.168.88.10",
        # callCallback=handle_call
    )

    try:
        phone.start()
        call = phone.call('5000')
        print("Phone started. Status:", phone.get_status())

        # print("Phone started. Status:", phone.get_status())
        print(phone._status)
        # Wait for the call to be answered
        while call.state != CallState.ANSWERED:
            pass
            # print("Current call state:", call.state)
        
        # Process audio once the call is answered
        while call.state == CallState.ANSWERED:
            audio_data = call.read_audio()
            if audio_data:
                print(audio_data)
                # You might want to save or process this audio data
                call.write_audio(audio_data)
            else:
                print("No audio data received.")
                
        # Handle call end
        if call.state == CallState.ENDED:
            print("Call ended.")

    # except RTP.DynamicPayloadType as e:
    #     print(f"RTP Dynamic Payload Type Error: {e}")
    # except RTP.RTPParseError as e:
    #     print(f"RTP Parse Error: {e}")
    # except Exception as e:
    #     print(f"An unexpected error occurred: {e}")

    finally:
        # Ensure the phone is stopped even if an error occurs
        phone.stop()
        print("Phone stopped.")

if __name__ == "__main__":
    import time
    start_voip()