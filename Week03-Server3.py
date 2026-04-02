import os
import socket

# Server address details
host = "127.0.0.1"
port = 5001

# Folder that contains files to send
server_folder = "serverfiles"
os.makedirs(server_folder, exist_ok=True)

# Create TCP socket, bind, and wait for one client
s = socket.socket()
s.bind((host, port))
s.listen(1)

print("Server running on", host, port)
conn, addr = s.accept()
print("Connected by", addr)

# Receive requested filename from client
filename = conn.recv(1024).decode().strip()
filename = os.path.basename(filename)
path = os.path.join(server_folder, filename)

# Send status first, then send file bytes if file exists
if not os.path.exists(path):
	conn.send("NOTFOUND".encode())
else:
	conn.send("FOUND".encode())
	with open(path, "rb") as f:
		data = f.read()
	conn.sendall(data)
	print("Sent file:", filename)

# Close connection and server socket
conn.close()
s.close()
