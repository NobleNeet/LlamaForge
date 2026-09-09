"""Relay an upstream stream, cancelling its socket when the client disconnects."""
import select
import socket
import threading


def relay(response, connection, write):
    finished = threading.Event()

    def watch_client():
        while not finished.wait(0.1):
            try:
                readable, _, _ = select.select([connection], [], [], 0)
                if not readable or connection.recv(1, socket.MSG_PEEK):
                    continue
            except (OSError, ValueError):
                pass
            except (TypeError, AttributeError):
                return  # non-socket fixtures; write failures still close upstream
            # Interrupt a blocking upstream readline even during a long prefill.
            # urllib's HTTPResponse owns this socket. Only shut down I/O here;
            # its owning thread closes the buffered response in finally below.
            try:
                response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
            except (AttributeError, OSError):
                pass
            return

    watcher = threading.Thread(target=watch_client, daemon=True, name="stream-disconnect")
    watcher.start()
    try:
        for line in response:
            write(line if line.endswith(b"\n") else line + b"\n")
    finally:
        finished.set()
        if hasattr(response, "close"):
            response.close()
        watcher.join(timeout=0.2)
