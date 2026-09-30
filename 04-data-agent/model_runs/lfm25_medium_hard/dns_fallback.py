"""Resolve temporary relay names when the node resolver returns an error."""
import json
import socket
import time
import urllib.parse
import urllib.request


def install():
    if getattr(socket.getaddrinfo, '_lfm_relay_fallback', False):
        return
    original = socket.getaddrinfo
    cache = {}

    def resolve(host, port, family=0, type=0, proto=0, flags=0):
        try:
            return original(host, port, family, type, proto, flags)
        except socket.gaierror:
            if not isinstance(host, str) or not host.rstrip('.').lower().endswith('.trycloudflare.com') or family not in (0, socket.AF_INET):
                raise
            entry = cache.get(host)
            if not entry or entry[0] < time.monotonic():
                url = 'https://dns.google/resolve?' + urllib.parse.urlencode({'name': host, 'type': 'A'})
                with urllib.request.urlopen(url, timeout=5) as response:
                    payload = json.loads(response.read(65536))
                addresses = [x['data'] for x in payload.get('Answer', []) if x.get('type') == 1]
                if payload.get('Status') != 0 or not addresses:
                    raise socket.gaierror('No relay A records from fallback resolver')
                cache[host] = (time.monotonic() + 60, addresses)
            else:
                addresses = entry[1]
            results = []
            for address in addresses:
                socket.inet_pton(socket.AF_INET, address)
                results.extend(original(address, port, family, type, proto, flags))
            return results

    resolve._lfm_relay_fallback = True
    socket.getaddrinfo = resolve
