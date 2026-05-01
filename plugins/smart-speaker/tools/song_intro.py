"""
Song introduction generation and playback for the smart-speaker plugin.

Generates a brief spoken intro before each song using an LLM call,
converts to speech via TTS, and plays it through the local speaker.
Runs synchronously inside the tool handler thread — no asyncio needed.
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config cache (valid for process lifetime)
# ---------------------------------------------------------------------------

_intro_config_cache: dict | None = None
_intro_config_lock = threading.Lock()

# Angle hints for prompt variation — different aspects to focus on
_ANGLE_HINTS = [
    "从歌词意境的角度介绍，让听众感受文字之美",
    "从歌手创作背景的角度介绍，讲述这首歌背后的故事",
    "从情感氛围的角度介绍，描述音乐带来的情绪体验",
    "从音乐风格和编曲的角度介绍，突出听觉特色",
    "从这首歌在歌手生涯中地位的角度介绍",
]


def _load_intro_config() -> dict:
    """Load smart_speaker intro config, cached for process lifetime."""
    global _intro_config_cache
    if _intro_config_cache is not None:
        return _intro_config_cache

    with _intro_config_lock:
        if _intro_config_cache is not None:
            return _intro_config_cache
        try:
            from hermes_cli.config import load_config
            config = load_config()
            ss = config.get("smart_speaker", {})
            _intro_config_cache = {
                "enabled": bool(ss.get("music_intro_enabled", True)),
                "temperature": float(ss.get("music_intro_temperature", 0.9)),
                "max_chars": int(ss.get("music_intro_max_chars", 50)),
                "timeout": float(ss.get("music_intro_timeout", 30)),
            }
        except Exception:
            logger.exception("Failed to load intro config, using defaults")
            _intro_config_cache = {
                "enabled": True,
                "temperature": 0.9,
                "max_chars": 150,
                "timeout": 5,
            }
    return _intro_config_cache


def intro_enabled() -> bool:
    """Return True if song intro feature is enabled in config."""
    return _load_intro_config()["enabled"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_and_play_intro(title: str, artist: str) -> bool:
    """Generate and play a spoken intro for a song.

    All failures are caught internally — the caller always gets a bool
    and can proceed to music playback regardless.

    Returns True if the intro was successfully generated and played.
    """
    if not title:
        return False

    intro_text = _generate_intro_text(title, artist)
    if not intro_text:
        return False

    return _play_intro_audio(intro_text)


# ---------------------------------------------------------------------------
# Intro text generation
# ---------------------------------------------------------------------------

def _generate_intro_text(title: str, artist: str) -> str | None:
    """Call the LLM to produce a 2-3 sentence Chinese intro for the song."""
    cfg = _load_intro_config()
    max_chars = cfg["max_chars"]
    temperature = cfg["temperature"]
    timeout = cfg["timeout"]

    # Resolve LLM client from main agent config
    model, base_url, api_key = _resolve_llm_client()
    if not model or not api_key:
        logger.warning("No LLM model or API key configured for song intro")
        return None

    user_msg = _build_intro_prompt(title, artist)
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个音乐解说员。用中文写1-2句简短介绍（50字以内）。"
                "从不同角度切入：歌词、创作背景、情感氛围或音乐风格。"
                "根据当前时间和季节调整表达。不要评价歌曲好坏，让听众产生期待。"
                "只需输出介绍文字，不要任何额外说明。"
            ),
        },
        {"role": "user", "content": user_msg},
    ]

    try:
        from openai import OpenAI

        client_kwargs: dict = {"api_key": api_key, "timeout": timeout}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=300,
        )

        text = (response.choices[0].message.content or "").strip()
        if not text:
            logger.warning("Song intro LLM returned empty text for %r", title)
            return None

        if len(text) > max_chars:
            text = text[:max_chars]

        logger.info("Generated intro for %r: %s... (%d chars)", title, text[:60], len(text))
        return text

    except Exception:
        logger.exception("Failed to generate song intro for %r", title)
        return None


def _build_intro_prompt(title: str, artist: str) -> str:
    """Build the user prompt with song info and temporal context."""
    now = datetime.now()
    hour = now.hour
    month = now.month
    weekday_num = now.weekday()

    # Time of day
    if 5 <= hour < 8:
        time_of_day = "清晨"
    elif 8 <= hour < 12:
        time_of_day = "上午"
    elif 12 <= hour < 14:
        time_of_day = "中午"
    elif 14 <= hour < 18:
        time_of_day = "下午"
    elif 18 <= hour < 22:
        time_of_day = "晚上"
    else:
        time_of_day = "深夜"

    # Season
    if 3 <= month <= 5:
        season = "春天"
    elif 6 <= month <= 8:
        season = "夏天"
    elif 9 <= month <= 11:
        season = "秋天"
    else:
        season = "冬天"

    # Weekday
    weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekday = weekday_names[weekday_num]
    day_type = "工作日" if weekday_num < 5 else "周末"

    # Angle hint — deterministic per date+song for cache-friendliness,
    # different per day for variety
    angle_index = hash((now.strftime("%Y%m%d"), title)) % len(_ANGLE_HINTS)
    angle_hint = _ANGLE_HINTS[angle_index]

    parts = [f"歌曲：《{title}》"]
    if artist:
        parts.append(f"歌手：{artist}")
    parts.append(f"当前时间：{time_of_day}，{season}{weekday}（{day_type}）")
    parts.append(f"介绍角度：{angle_hint}")
    parts.append("要求：1-2句，50字以内，仅输出介绍文字。")

    return "\n".join(parts)


def _resolve_llm_client() -> tuple[str | None, str | None, str | None]:
    """Resolve (model, base_url, api_key) for song intro generation.

    Prefers DashScope (qwen-turbo via compatible-mode endpoint) because
    the main agent's model may be a coding-specialized endpoint that rejects
    general chat requests (e.g. Kimi coding).
    """
    import os as _os

    try:
        # DashScope OpenAI-compatible endpoint — ideal for Chinese text
        dashscope_key = _os.getenv("TTS_DASHSCOPE_API_KEY", "").strip()
        if not dashscope_key:
            dashscope_key = _os.getenv("DASHSCOPE_API_KEY", "").strip()
        if dashscope_key:
            return (
                "qwen3.6-flash",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                dashscope_key,
            )
    except Exception:
        logger.exception("Failed to resolve DashScope config")

    # Fallback: try the main agent's model config, but skip coding endpoints
    try:
        from hermes_cli.config import load_config

        config = load_config()
        model_cfg = config.get("model", {})

        if isinstance(model_cfg, str):
            model = model_cfg
            base_url = config.get("base_url", "")
            api_key = config.get("api_key", "")
        elif isinstance(model_cfg, dict):
            model = model_cfg.get("model") or model_cfg.get("default", "")
            base_url = model_cfg.get("base_url", "")
            api_key = model_cfg.get("api_key", "")
        else:
            return None, None, None

        if not model:
            return None, None, None

        if not base_url:
            base_url = None

        if not api_key:
            provider = (
                model_cfg.get("provider", "") if isinstance(model_cfg, dict)
                else config.get("provider", "")
            )
            provider_lookup = provider.upper().replace("-", "_") if provider else ""
            candidates = [f"{provider_lookup}_API_KEY"]
            if "-" in provider:
                short = provider.split("-")[0].upper()
                candidates.append(f"{short}_API_KEY")
            candidates.append("OPENAI_API_KEY")
            for candidate in candidates:
                val = _os.getenv(candidate, "").strip()
                if val:
                    api_key = val
                    break

        if not api_key:
            return None, None, None

        return str(model), str(base_url) if base_url else None, str(api_key)

    except Exception:
        logger.exception("Failed to resolve fallback LLM config")
        return None, None, None


# ---------------------------------------------------------------------------
# TTS + Audio Playback
# ---------------------------------------------------------------------------

def _play_intro_audio(text: str) -> bool:
    """Convert text to speech and play through the local speaker.

    Blocks until playback finishes. Returns True on success.
    """
    tmp_path = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(tmp_fd)

        from tools.tts_tool import text_to_speech_tool

        result_json = text_to_speech_tool(text=text, output_path=tmp_path)
        result = json.loads(result_json)
        if not result.get("success"):
            logger.warning("TTS failed for intro: %s", result.get("error", "unknown"))
            return False

        audio_path = result.get("file_path", tmp_path)
        if not os.path.isfile(audio_path) or os.path.getsize(audio_path) == 0:
            logger.warning("TTS produced empty file for intro")
            return False

        return _play_audio_file_sync(audio_path)

    except Exception:
        logger.exception("Failed to play intro audio")
        return False
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _play_audio_file_sync(file_path: str) -> bool:
    """Play an audio file synchronously via sounddevice.

    Handles WAV and MP3 (via ffmpeg). Blocks until playback finishes.
    Designed to be called from a thread.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error("Audio file not found: %s", path)
        return False

    try:
        import sounddevice as sd
    except ImportError:
        logger.warning("sounddevice not available for intro playback")
        return False

    data = path.read_bytes()
    sample_rate = 24000
    audio_np = None

    # Try WAV direct
    if data[:4] == b"RIFF":
        import wave
        with io.BytesIO(data) as buf:
            with wave.open(buf, "rb") as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                frames = wf.readframes(wf.getnframes())
                audio_np = np.frombuffer(frames, dtype=np.int16)
                if channels > 1:
                    audio_np = audio_np.reshape(-1, channels)

    # Try MP3 via ffmpeg
    if audio_np is None and shutil.which("ffmpeg"):
        try:
            result = subprocess.run(
                ["ffmpeg", "-i", str(path), "-f", "wav", "-"],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0 and len(result.stdout) > 44:
                import wave
                with io.BytesIO(result.stdout) as buf:
                    with wave.open(buf, "rb") as wf:
                        sample_rate = wf.getframerate()
                        channels = wf.getnchannels()
                        frames = wf.readframes(wf.getnframes())
                        audio_np = np.frombuffer(frames, dtype=np.int16)
                        if channels > 1:
                            audio_np = audio_np.reshape(-1, channels)
        except Exception:
            logger.exception("ffmpeg decode failed")

    if audio_np is None:
        logger.error("Could not decode audio for intro playback")
        return False

    try:
        sd.stop()
        for attempt in range(2):
            try:
                sd.play(audio_np, samplerate=sample_rate)
                sd.wait()
                logger.info("Song intro playback completed")
                return True
            except Exception as e:
                if attempt == 0 and "Device unavailable" in str(e):
                    time.sleep(0.5)
                    continue
                raise
    except Exception:
        logger.exception("Error playing intro audio")
        return False

    return False
