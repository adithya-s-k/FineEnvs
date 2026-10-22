"""Run a Python script/module with network connection APIs blocked (CI verification)."""

from __future__ import annotations

import ipaddress
import runpy
import socket
import sys


def blocked(*args, **kwargs):
    raise RuntimeError("TraceAudit offline verification forbids network connections")


def main():
    # Windows implements asyncio's internal socketpair using a loopback connection.
    # Permit local IPC while denying every external connection and DNS lookup.
    connect, connect_ex, resolve = (
        socket.socket.connect,
        socket.socket.connect_ex,
        socket.getaddrinfo,
    )

    def local(address):
        try:
            return ipaddress.ip_address(address[0]).is_loopback
        except (ValueError, TypeError, IndexError):
            return False

    def guarded_connect(sock, address):
        if sock.family == getattr(socket, "AF_UNIX", object()) or local(address):
            return connect(sock, address)
        return blocked()

    def guarded_connect_ex(sock, address):
        if sock.family == getattr(socket, "AF_UNIX", object()) or local(address):
            return connect_ex(sock, address)
        return blocked()

    def guarded_resolve(host, *args, **kwargs):
        if host == "localhost" or local((host, 0)):
            return resolve(host, *args, **kwargs)
        return blocked()

    socket.create_connection = blocked
    socket.getaddrinfo = guarded_resolve
    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    args = sys.argv[1:]
    if args[:1] == ["-m"]:
        sys.argv = args[1:]
        runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
    elif args:
        sys.argv = args
        runpy.run_path(args[0], run_name="__main__")
    else:
        raise SystemExit("Usage: offline.py SCRIPT [ARGS] or offline.py -m MODULE [ARGS]")


if __name__ == "__main__":
    main()
