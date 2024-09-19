from pyVoIP.VoIP import VoIPPhone, CallState
import time

'''
def handle_call(call):
        # i=0
        call.answer()
        try:
            while call.state == CallState.ANSWERED:
                audio_bytes = call.read_audio()
                # print(audio_bytes)
                # call.write_audio(audio_bytes)
                if audio_bytes!=b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00':
                    call.write_audio(audio_bytes)
                    print(audio_bytes)
                    
                # dtmf = call.get_dtmf()
                # if dtmf == "1":
                #     print('Pressed 1')
                #     # Do something
                #     # call.hangup()
                # elif dtmf == "2":
                #     print('Pressed 2')
                #     # Do something else
                #     # call.hangup()
                #     # print(audio_bytes)

        except Exception as e:
            print(e)
            
        finally:
            call.hangup()


phone = VoIPPhone(
        server="192.168.88.51",
        port=5060,
        username="654321",
        password="pFoZz?a9R$qyjRKEb.cr",
        myIP="192.168.88.10",
        callCallback=handle_call
)

phone.start()
# phone.call('5000')
print("Phone started. Status:", phone.get_status())
print(phone._status)
input("Press any key to exit the VOIP phone session.")
phone.stop()
'''

'''
try:
    # call = phone.call('5000')
    print("Call initiated. Send mode:", call.sendmode)
    
    # Wait for the call to be answered
    while call.state != CallState.ANSWERED:
        print("Current call state:", call.state)
    
    # Process audio once the call is answered
    while call.state == CallState.ANSWERED:
        # print("Current call state:", call.state)
        audio_bytes = call.read_audio()
        if audio_bytes:
            print(audio_bytes)
        else:
            print("No audio data received.")
            
    # Handle call end
    if call.state == CallState.ENDED:
        print("Call ended.")

except Exception as e:
    print("An error occurred:", e)
'''

# from pyVoIP.VoIP import VoIPPhone, CallState

# phone = VoIPPhone(
#         server="192.168.88.5",
#         port=5060,
#         username="5001",
#         password="iX3TxsD9jWxmZU5",
#         myIP="192.168.88.10"
# )

# phone.start()

# try:
#     call = phone.call('5000')
#     print("Call initiated. Send mode:", call.sendmode)
    
#     # Wait for the call to be answered
#     while call.state != CallState.ANSWERED:
#         print("Current call state:", call.state)
    
#     # Process audio once the call is answered
#     while call.state == CallState.ANSWERED:
#         # print("Current call state:", call.state)
#         audio_bytes = call.read_audio()
#         if audio_bytes:
#             print(audio_bytes)
#         else:
#             print("No audio data received.")
            
#     # Handle call end
#     if call.state == CallState.ENDED:
#         print("Call ended.")

# except Exception as e:
#     print("An error occurred:", e)

# from pyVoIP.VoIP import VoIPPhone, CallState
# import wave
# import uuid
# import numpy as np

# # def handle_call(call):
# #         i=0
# #         call.answer()
# #         try:
# #             while call.state == CallState.ANSWERED:
# #                 audio_bytes = call.read_audio()
# #                 # print(audio_bytes)
# #                 # call.write_audio(audio_bytes)
# #                 if audio_bytes!=b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00':
# #                     call.write_audio(audio_bytes)
# #                     print(audio_bytes)
                    
# #                 # dtmf = call.get_dtmf()
# #                 # if dtmf == "1":
# #                 #     print('Pressed 1')
# #                 #     # Do something
# #                 #     # call.hangup()
# #                 # elif dtmf == "2":
# #                 #     print('Pressed 2')
# #                 #     # Do something else
# #                 #     # call.hangup()
# #                 #     # print(audio_bytes)

# #         except Exception as e:
# #             print(e)
            
# #         finally:
# #             call.hangup()

# def start_voip():
#     phone = VoIPPhone(
#         server="192.168.88.5",
#         port=5060,
#         username="5001",
#         password="iX3TxsD9jWxmZU5",
#         myIP="192.168.88.10",
#         # callCallback=handle_call
#     )

#     phone.start()
#     phone.call('9805816686')
#     print("Phone started. Status:", phone.get_status())

#     # print("Phone started. Status:", phone.get_status())
#     print(phone._status)
#     input("Press any key to exit the VOIP phone session.")
#     phone.stop()

# if __name__ == "__main__":
#     import time
#     start_voip()