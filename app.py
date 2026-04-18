from backend.main import app


if __name__ == "__main__":
    import socket
    import uvicorn

    def _is_port_open(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    selected_port = 8000
    for candidate in (8000, 8001, 8002, 8003):
        if not _is_port_open(candidate):
            selected_port = candidate
            break

    if selected_port != 8000:
        print(f"[INFO] Port 8000 is already in use. Starting backend on port {selected_port}.")

    uvicorn.run("backend.main:app", host="127.0.0.1", port=selected_port, reload=False)
