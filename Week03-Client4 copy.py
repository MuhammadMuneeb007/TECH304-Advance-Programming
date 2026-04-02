import socket

HOST = "127.0.0.1"
PORT = 5001

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST, PORT))

while True:
    message = input("Client 2 - type message: ")
    if message.lower() == "exit":
        break

    client.sendall(message.encode())
    data = client.recv(1024).decode()
    print("Server says:", data)

client.close()