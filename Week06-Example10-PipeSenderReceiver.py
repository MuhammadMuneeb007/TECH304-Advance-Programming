from multiprocessing import Pipe, Process


def sender(conn):
    conn.send("Hello, Receiver!")
    conn.close()


def receiver(conn):
    message = conn.recv()
    print(f"Receiver got: {message}")
    conn.close()


def main():
    parent_conn, child_conn = Pipe()
    process_sender = Process(target=sender, args=(child_conn,))
    process_receiver = Process(target=receiver, args=(parent_conn,))

    process_sender.start()
    process_receiver.start()

    process_sender.join()
    process_receiver.join()


if __name__ == "__main__":
    main()
