import io
import json
import socket
import urllib.request
import pytest
from dns_fallback import install


def test_relay_only_fallback_and_cache(monkeypatch):
    calls=[]
    def original(host, port, *args):
        if host in ('new.trycloudflare.com','unrelated.example'):
            raise socket.gaierror('node resolver failed')
        return [(socket.AF_INET,socket.SOCK_STREAM,6,'',(host,port))]
    def doh(url, timeout):
        calls.append(url)
        return io.BytesIO(json.dumps({'Status':0,'Answer':[{'type':1,'data':'104.16.230.132'}]}).encode())
    monkeypatch.setattr(socket,'getaddrinfo',original)
    monkeypatch.setattr(urllib.request,'urlopen',doh)
    install();install()
    assert socket.getaddrinfo('new.trycloudflare.com',443)[0][-1]==('104.16.230.132',443)
    socket.getaddrinfo('new.trycloudflare.com',443)
    assert len(calls)==1
    with pytest.raises(socket.gaierror):socket.getaddrinfo('unrelated.example',443)
    assert len(calls)==1
    assert socket.getaddrinfo('127.0.0.1',123)[0][-1]==('127.0.0.1',123)
