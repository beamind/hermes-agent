"""
Netease Cloud Music client — session persistence, QR login, search, and download.

Adapted from home-ai-assistant's NeteaseMusic.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# Default session path: ~/.config/hermes/netease_session.json
_DEFAULT_SESSION_DIR = Path.home() / ".config" / "hermes"
_DEFAULT_SESSION_PATH = _DEFAULT_SESSION_DIR / "netease_session.json"


class NeteaseMusic:
    """Netease Cloud Music client with session persistence and QR login."""

    def __init__(self, session_path: str | Path | None = None) -> None:
        if session_path is None:
            self._session_path = _DEFAULT_SESSION_PATH
        else:
            self._session_path = Path(session_path)
        import pyncm

        self._session = pyncm.GetCurrentSession()
        self.load_session()

    # ---- session persistence ----

    def load_session(self) -> bool:
        """Restore session from disk. Returns True if loaded and still valid."""
        if not self._session_path.exists():
            return False
        try:
            data = json.loads(self._session_path.read_text(encoding="utf-8"))
            self._session.load(data)
            if self._session.logged_in:
                logger.info("Netease session restored, user: %s", self._session.nickname)
                return True
            logger.info("Netease saved session is expired")
        except Exception:
            logger.warning("Failed to load Netease session", exc_info=True)
        return False

    def save_session(self) -> None:
        """Persist current session to disk."""
        self._session_path.parent.mkdir(parents=True, exist_ok=True)
        data = self._session.dump()
        self._session_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        logger.info("Netease session saved")

    def clear_session(self) -> None:
        """Logout and remove saved session."""
        from pyncm.apis import login

        try:
            login.LoginLogout()
        except Exception:
            pass
        if self._session_path.exists():
            self._session_path.unlink()
        self._session.cookies.clear()
        logger.info("Netease session cleared")

    # ---- login status ----

    @property
    def is_logged_in(self) -> bool:
        return self._session.logged_in

    @property
    def nickname(self) -> str:
        return self._session.nickname or ""

    # ---- QR login flow ----

    def generate_qr(self) -> dict:
        """Generate QR login key and base64 image.

        Returns {"key": str, "qr_img": str (data URI), "url": str}.
        """
        from pyncm.apis import login

        result = login.LoginQrcodeUnikey()
        unikey = result["unikey"]
        url = login.GetLoginQRCodeUrl(unikey)
        logger.info("Netease QR code generated, key=%s", unikey)

        try:
            import qrcode

            img = qrcode.make(url)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            qr_img = f"data:image/png;base64,{b64}"
        except Exception:
            logger.warning("Failed to generate QR image", exc_info=True)
            qr_img = ""

        return {"key": unikey, "qr_img": qr_img, "url": url}

    def check_qr(self, key: str) -> dict:
        """Check QR scan status.

        Returns {"code": int, "message": str}.
        Code: 800=expired, 801=waiting, 802=scanned, 803=confirmed.
        """
        from pyncm.apis import login

        result = login.LoginQrcodeCheck(key)
        code = result.get("code", 0)
        message = result.get("message", "")

        if code == 801:
            logger.info("Netease QR waiting for scan")
        elif code == 802:
            logger.info("Netease QR scanned, waiting for confirmation")
        elif code == 803:
            time.sleep(0.2)
            try:
                status = login.GetCurrentLoginStatus()
                if status.get("code") == 200:
                    login.WriteLoginInfo(status)
                    time.sleep(0.1)
            except Exception:
                logger.exception("Failed to refresh Netease login status after QR confirm")
            logger.info("Netease QR login confirmed, user=%s", self.nickname)
            self.save_session()
        elif code == 800:
            logger.info("Netease QR expired")
        else:
            logger.info("Netease QR check: code=%s, message=%s", code, message)

        return {"code": code, "message": message}

    # ---- search & playback ----

    def search(self, keyword: str, limit: int = 5) -> list[dict]:
        """Search songs by keyword.

        Returns list of {"id", "name", "artist", "album"}.
        """
        from pyncm.apis import cloudsearch

        logger.info("Netease searching: %s", keyword)
        try:
            result = cloudsearch.GetSearchResult(keyword, stype=1, limit=limit)
            songs = result.get("result", {}).get("songs", [])
            logger.info(
                "Netease search returned %d songs for: %s", len(songs), keyword
            )
            return [
                {
                    "id": s["id"],
                    "name": s.get("name", ""),
                    "artist": ", ".join(a.get("name", "") for a in s.get("ar", [])),
                    "album": s.get("al", {}).get("name", ""),
                }
                for s in songs
            ]
        except Exception:
            logger.exception("Netease search failed")
            return []

    def get_play_url(self, song_id: int) -> str | None:
        """Get a temporary playback URL for a song. Returns None if unavailable."""
        from pyncm.apis import track

        logger.info("Netease getting play URL for song %s", song_id)
        try:
            result = track.GetTrackAudio([song_id], bitrate=320000)
            data = result.get("data", [])
            if data and data[0].get("url"):
                logger.info("Netease play URL obtained for song %s", song_id)
                return data[0]["url"]
            logger.warning(
                "Netease play URL unavailable for song %s (possibly VIP/restricted)",
                song_id,
            )
        except Exception:
            logger.exception("Failed to get Netease play URL for %s", song_id)
        return None

    def get_liked_songs(self) -> list[dict[str, str]]:
        """Get liked songs from '我喜欢的音乐' playlist.

        Returns list of {"title": str, "artist": str}.
        """
        from pyncm.apis import playlist, user

        if not self.is_logged_in:
            logger.warning("get_liked_songs called but not logged in")
            return []
        try:
            uid = self._session.uid
            if not uid:
                logger.warning("No user ID in session")
                return []

            result = user.GetUserPlaylists(uid)
            playlists = result.get("playlist", [])
            if not playlists:
                logger.info("No playlists found for user %s", uid)
                return []

            liked = playlists[0]
            liked_id = liked["id"]
            liked_name = liked.get("name", "我喜欢的音乐")
            logger.info(
                "Fetching liked songs from '%s' (id=%s)", liked_name, liked_id
            )

            tracks_result = playlist.GetPlaylistAllTracks(liked_id)
            songs = tracks_result.get("songs", [])
            logger.info("Fetched %d tracks from liked songs playlist", len(songs))

            extracted: list[dict[str, str]] = []
            for s in songs:
                title = s.get("name", "").strip()
                if not title:
                    continue
                artists = s.get("ar", [])
                artist = artists[0].get("name", "").strip() if artists else ""
                extracted.append({"title": title, "artist": artist})

            return extracted
        except Exception:
            logger.exception("Failed to get liked songs")
            return []

    def download_song(self, url: str, save_path: Path) -> bool:
        """Download a song from URL to save_path. Returns True on success."""
        try:
            logger.info("Downloading song to %s", save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info("Song downloaded successfully: %s", save_path)
            return True
        except Exception:
            logger.exception("Failed to download song to %s", save_path)
            return False

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Remove characters that are illegal in filenames."""
        illegal = '\\/:*?"<>|'
        for ch in illegal:
            name = name.replace(ch, "_")
        return name.strip()


# ---------------------------------------------------------------------------
# CLI entry point for QR login
# ---------------------------------------------------------------------------


def _cli_login() -> None:
    """Interactive QR login for terminal use."""
    import tempfile

    nm = NeteaseMusic()
    if nm.is_logged_in:
        print(f"Already logged in as: {nm.nickname}")
        return

    qr_data = nm.generate_qr()
    key = qr_data["key"]
    url = qr_data["url"]

    # Save QR image to temp file
    qr_path: str | None = None
    try:
        img_data = base64.b64decode(qr_data["qr_img"].split(",")[1])
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(img_data)
            qr_path = f.name
        print(f"QR code saved to: {qr_path}")
        # Try to open image viewer
        try:
            import subprocess

            subprocess.Popen(
                ["xdg-open", qr_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
    except Exception:
        print("Please open this URL in your browser and scan with Netease Music app:")
        print(url)

    print("Waiting for QR scan... (Ctrl+C to cancel)")

    try:
        while True:
            time.sleep(2)
            result = nm.check_qr(key)
            code = result["code"]
            if code == 803:
                print(f"Login successful! User: {nm.nickname}")
                if qr_path and os.path.exists(qr_path):
                    os.unlink(qr_path)
                return
            elif code == 800:
                print("QR code expired. Please run again.")
                if qr_path and os.path.exists(qr_path):
                    os.unlink(qr_path)
                return
            elif code in (801, 802):
                print("  ...")
    except KeyboardInterrupt:
        print("\nCancelled.")
        if qr_path and os.path.exists(qr_path):
            os.unlink(qr_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "login":
        _cli_login()
    else:
        print("Usage: python netease_music.py login")
