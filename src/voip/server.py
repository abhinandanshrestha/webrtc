from pyVoIP.VoIP import VoIPPhone
import wave
import uuid
import numpy as np
import threading
from call_handler import CallHandler

# Function to handle each call by creating a CallHandler instance
def handle_call_in_thread(call):
    call_handler = CallHandler()
    threading.Thread(target=call_handler.handle_call, daemon=True, args=(call,)).start()


def start_voip():
    phone = VoIPPhone(
        server="192.168.88.5",
        port=5060,
        username="5001",
        password="iX3TxsD9jWxmZU5",
        myIP="192.168.88.10",
        callCallback=handle_call_in_thread
    )

    phone.start()
    print("Phone started. Status:", phone.get_status())

    try:
        while True:
            time.sleep(0.1)  # Keep the main thread running
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        phone.stop()
        print("Phone stopped")

if __name__ == "__main__":
    import time
    start_voip()

