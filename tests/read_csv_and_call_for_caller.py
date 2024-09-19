import pandas as pd
import subprocess

df= pd.read_csv('clients.csv')

for i in list(df['number']):
    # Command and arguments as a list
    command = ['/home/oem/webrtc/myvenv/bin/python',"src/voip/caller_english.py","-c", str(i)]
    print(command)
    # Run the command
    result=subprocess.run(command, capture_output=True, text=True)
    print("Output:", result.stdout)
    # print(i)