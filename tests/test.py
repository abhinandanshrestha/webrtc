import wave
import time
from pyVoIP.VoIP import VoIPPhone, CallState

phone = VoIPPhone(
        server="192.168.88.5",
        port=5060,
        username="5001",
        password="iX3TxsD9jWxmZU5",
        myIP="192.168.88.10"
)

phone.start()

try:
    call = phone.call('5000')
    print("Call initiated. Send mode:", call.sendmode)
    
    # Wait for the call to be answered
    while call.state != CallState.ANSWERED:
        print("Current call state:", call.state)
    
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

except Exception as e:
    print("An error occurred:", e)

finally:
    # Ensure the call is hung up and resources are cleaned up
    call.hangup()
    phone.stop()