import os
import socket

# Server address details
host = "127.0.0.1"
port = 5001

# Folder where downloaded files are saved
client_folder = "clientfiles"
os.makedirs(client_folder, exist_ok=True)

# Ask user which file to download
filename = input("Enter file name (example: sample.txt): ").strip()

# Connect to server and send filename
c = socket.socket()
c.connect((host, port))
c.send(filename.encode())

# Check server response
status = c.recv(1024).decode()

if status == "NOTFOUND":
	print("File not found on server")
else:
	# Save incoming bytes to clientfiles/<filename>
	path = os.path.join(client_folder, os.path.basename(filename))
	data = b""
	while True:
		chunk = c.recv(1024)
		if not chunk:
			break
		data += chunk

	with open(path, "wb") as f:
		f.write(data)

	print("File received and saved to", path)

# Close client socket
c.close()
