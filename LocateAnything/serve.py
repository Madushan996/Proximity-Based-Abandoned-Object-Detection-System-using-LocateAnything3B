"""
Local launcher for the PADS web UI.

Serves ui/index.html on http://localhost:8000 and opens it in your browser.
The page runs locally but submits jobs to your *deployed* Modal GPU endpoint
(set in ui/index.html as MODAL_URL), then plays the annotated result video.

Usage:
    python serve.py            # serve on :8000 and open the browser
    python serve.py 9000       # use a different port

Nothing here needs the heavy ML deps — the model stays on Modal.
"""
import http.server
import socketserver
import sys
import webbrowser
from functools import partial
from pathlib import Path

UI_DIR = Path(__file__).parent / "ui"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000


def main():
    if not (UI_DIR / "index.html").exists():
        raise SystemExit(f"index.html not found in {UI_DIR}")

    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(UI_DIR))
    with socketserver.TCPServer(("127.0.0.1", PORT), handler) as httpd:
        url = f"http://localhost:{PORT}/"
        print(f"PADS UI → {url}  (serving {UI_DIR})")
        print("Submitting jobs to the Modal endpoint configured in ui/index.html.")
        print("Ctrl-C to stop.")
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
