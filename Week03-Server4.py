import socketserver

HOST = "127.0.0.1"
PORT = 5001

class MyTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        print(f"Connected from: {self.client_address}")

        while True:
            data = self.request.recv(1024).strip()
            if not data:
                break

            message = data.decode()
            print(f"From {self.client_address}: {message}")

            reply = f"Server received: {message}"
            self.request.sendall(reply.encode())

        print(f"Disconnected: {self.client_address}")


if __name__ == "__main__":
    server = socketserver.ThreadingTCPServer((HOST, PORT), MyTCPHandler)
    print(f"Server running on {HOST}:{PORT}")
    server.serve_forever()