# How Signaling Works:
1 - Peer 1(Client) - creates an offer, sends to webrtc server

	1-Creates offer
	
	2-Sets Local Description
	
2 - Peer2 2(Server) - Receives offers, create an Answer

	1-Sets Remote Description
	
	2-Creates Answer
	
	3-Sets Local Description
	
3 - Peer 2(Server) - Sends the Answer as a response of the post request

4 - Peer 1(Client) - Receives the Answer

	3-Sets Remote Description

5 - Finally the connection is established

# Steps to run:

If there are no wheels for your system or if you wish to build aiortc from source you will need a couple of libraries installed on your system:

Opus for audio encoding / decoding
LibVPX for video encoding / decoding
Linux
On Debian/Ubuntu run:

	apt install libopus-dev libvpx-dev
OS X
On OS X run:

	brew install opus libvpx

Then:

	pip install -r requirements.txt

	cd src/callbot
	python server.py 
	python server.py --port <port>

	cd frontend
	python client_test.py <client_id> # client_id can be anything; the program creates a random number for this client.
>
<!-- > After Recording has been completed, run: python answerer.py -->
>
	