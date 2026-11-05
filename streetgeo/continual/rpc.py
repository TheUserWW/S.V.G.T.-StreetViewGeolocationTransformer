"""Authenticated loopback JSON endpoint for a locally generated extension."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import threading


class LocalServer:
    def __enter__(self):
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
                if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+owner.token):
                    self.send_error(403); return
                try:
                    length=int(self.headers.get('Content-Length','0'))
                    if not 0<length<=16*1024*1024: raise ValueError('Invalid body size')
                    value=json.loads(self.rfile.read(length))
                    if not isinstance(value,dict): raise ValueError('Invalid JSON object')
                    result=owner.accept(self.path,value)
                    data=json.dumps(result,allow_nan=False).encode()
                except (ValueError,KeyError,TypeError): self.send_error(400); return
                self.send_response(200); self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
        self.server=ThreadingHTTPServer(('127.0.0.1',self.port),Handler)
        self.server.daemon_threads=True; self.port=self.server.server_port
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True); self.thread.start()
        return self
    def __exit__(self,*args):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)
