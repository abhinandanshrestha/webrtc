from pyVoIP.VoIP import VoIPPhone, CallState
import time

# def answer(call):
#     print(call.state)
#     time.sleep(5)
#     call.answer()
#     print(call.state)

phone = VoIPPhone(
        server="192.168.98.48",
        port=5060,
        username="5001",
        password="iX3TxsD9jWxmZU5",
        myIP="192.168.88.10",
        sipPort=53563
        # ,callCallback=answer 
)

phone.start()
try:
    
    call = phone.call('5000')
    input('Press anything to stop')
    phone.hangup()
    # print(call)
    # print("Call initiated. Send mode:", call.sendmode)
    # print("Call initiated. Send mode:", phone.get_status())
    # print(dir(call.sip))
    # while True:
    #     print(call.get_status())
    # while call.state != CallState.ANSWERED:
    #     print("Current call state:", call.state)
    
    # while call.state == CallState.ANSWERED:
    #     audio_bytes = call.read_audio()
    #     print(audio_bytes)
    # # Handle call end
    # if call.state == CallState.ENDED:
    #     print("Call ended.")

except Exception as e:
    print("An error occurred:", e)

