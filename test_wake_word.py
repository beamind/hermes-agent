#!/usr/bin/env python3
"""
远程音箱唤醒测试脚本

测试流程：
1. 调用百炼 DashScope TTS 生成音频"小爱同学播放王菲的流年"
2. 播放这个音频文件
3. 检查 voice gateway 日志，验证唤醒/ASR/播放/TTS 链路
4. 手动确认音箱行为

用法：
    python test_wake_word.py           # 交互式（按 Enter 开始播放）
    AUTO_TEST=1 python test_wake_word.py  # 自动模式（跳过确认）
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. 环境初始化
# ---------------------------------------------------------------------------

# 加载 ~/.hermes/.env（API keys）
_hermes_env = os.path.expanduser("~/.hermes/.env")
if os.path.exists(_hermes_env):
    try:
        from dotenv import load_dotenv
        load_dotenv(_hermes_env)
        print(f"✅ 已加载 ~/.hermes/.env")
    except ImportError:
        with open(_hermes_env, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())
        print(f"✅ 已加载 ~/.hermes/.env (手动解析)")

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from hermes_constants import get_hermes_dir
from tools.tts_tool import text_to_speech_tool

TEST_TEXT = "小爱同学播放王菲的流年"
OUTPUT_DIR = str(get_hermes_dir("cache/audio", "audio_cache"))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "test_wake_word.mp3")

# gateway 相关路径
PID_FILE = REPO_ROOT / "hermes-gateway.pid"
LOG_FILE = Path(os.path.expanduser("~/.hermes/logs/gateway.log"))
AGENT_LOG_FILE = Path(os.path.expanduser("~/.hermes/logs/agent.log"))

# 日志搜索关键字（中英文双语覆盖）
WAKE_KEYWORDS = ["Wake word detected", "wake", "唤醒"]
ASR_KEYWORDS = ["ASR result", "transcrib", "识别结果", "Transcription"]
PLAY_KEYWORDS = ["play_music", "control_playback", "get_playback_status", "MusicPlayer", "播放"]
# 注意：排除 "stopping TTS"（唤醒时暂停 TTS 是正常行为，不是 TTS 生成事件）
TTS_KEYWORDS = ["text_to_speech", "Generating speech", "TTS generation", "语音合成"]
TTS_EXCLUDE_KEYWORDS = ["stopping TTS", "pausing music, stopping"]
INTRO_KEYWORDS = ["Song intro playback completed", "Generated intro for"]


# ---------------------------------------------------------------------------
# 步骤 1: 生成 TTS 音频
# ---------------------------------------------------------------------------
def generate_test_audio():
    print("=" * 60)
    print("🎵 步骤 1: 生成测试音频 (DashScope CosyVoice)")
    print(f"   文本: {TEST_TEXT}")
    print(f"   输出: {OUTPUT_FILE}")
    print("=" * 60)

    try:
        raw = text_to_speech_tool(text=TEST_TEXT, output_path=OUTPUT_FILE)

        # text_to_speech_tool 始终返回 JSON 字符串
        result = json.loads(raw)
        if result.get("success"):
            audio_path = result.get("file_path", OUTPUT_FILE)
            print(f"✅ 音频生成成功")
            print(f"   路径: {audio_path}")
            print(f"   Provider: {result.get('provider', 'unknown')}")
            return audio_path
        else:
            print(f"❌ TTS 返回失败: {result.get('error', '未知错误')}")
            return None
    except json.JSONDecodeError:
        print(f"⚠️  TTS 返回值不是 JSON: {raw[:200]}")
        return None
    except Exception as e:
        print(f"❌ 音频生成异常: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# 步骤 2: 播放音频
# ---------------------------------------------------------------------------
def play_audio_file(audio_path):
    print("\n" + "=" * 60)
    print("🔊 步骤 2: 播放测试音频")
    print(f"   文件: {audio_path}")
    print("=" * 60)

    if not os.path.exists(audio_path):
        print(f"❌ 音频文件不存在: {audio_path}")
        return False

    players = [
        ["paplay", audio_path],                               # PulseAudio/PipeWire
        ["aplay", audio_path],                                 # ALSA
        ["ffplay", "-nodisp", "-autoexit", audio_path],       # ffmpeg
        ["mpg123", audio_path],                                # mpg123
        ["afplay", audio_path],                                # macOS
    ]

    for player in players:
        try:
            result = subprocess.run(player, capture_output=True, timeout=60)
            if result.returncode == 0:
                print(f"✅ 播放成功 (使用 {player[0]})")
                return True
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            print(f"⏱️  播放中（{player[0]} 超时，可能仍在播放）")
            return True
        except Exception as e:
            print(f"⚠️  {player[0]} 失败: {e}")
            continue

    print("❌ 无法播放，请安装: sudo apt install pulseaudio-utils 或 ffmpeg")
    return False


# ---------------------------------------------------------------------------
# 步骤 3: 检查 Gateway 运行状态
# ---------------------------------------------------------------------------
def check_gateway_running():
    """检查 voice gateway 是否在运行"""
    print("\n" + "=" * 60)
    print("📊 步骤 3: 检查 Voice Gateway 状态")
    print("=" * 60)

    # 方式 1: run.sh 的 PID 文件
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)  # 发信号 0 仅检测进程是否存在
            print(f"✅ Voice Gateway 正在运行 (PID: {pid}, 来自 run.sh)")
            return True
        except (ValueError, OSError):
            print(f"⚠️  PID 文件存在但进程已退出，清理中...")
            PID_FILE.unlink(missing_ok=True)

    # 方式 2: pgrep 搜索（匹配 "hermes gateway" 和 "gateway.run" 两种启动方式）
    try:
        result = subprocess.run(
            ["pgrep", "-f", "hermes.*gateway|gateway\\.run"],
            capture_output=True, text=True
        )
        pids = result.stdout.strip()
        if pids:
            print(f"✅ Voice Gateway 正在运行 (PID: {pids.replace(chr(10), ', ')})")
            return True
    except FileNotFoundError:
        pass

    print("❌ Voice Gateway 未运行！")
    print("   启动方式: bash run.sh start")
    print("   或: python -m gateway.run")
    return False


# ---------------------------------------------------------------------------
# 步骤 4: 日志分析
# ---------------------------------------------------------------------------
def _grep_log(lines, keywords, exclude_keywords=None):
    """在日志行列表中搜索关键字，返回匹配的行（排除 exclude 关键字）"""
    matched = []
    for line in lines:
        line_lower = line.lower()
        if any(kw.lower() in line_lower for kw in keywords):
            if exclude_keywords and any(ex.lower() in line_lower for ex in exclude_keywords):
                continue
            matched.append(line.strip())
    return matched


def check_gateway_logs():
    """分析 gateway 日志，检查唤醒链路"""
    print("\n" + "=" * 60)
    print("📊 步骤 4: Gateway 日志分析")
    print("=" * 60)

    if not LOG_FILE.exists():
        print(f"⚠️  日志文件不存在: {LOG_FILE}")
        return

    try:
        # 读取最近 300 行
        result = subprocess.run(
            ["tail", "-n", "300", str(LOG_FILE)],
            capture_output=True, text=True
        )
        lines = result.stdout.splitlines()

        wake_lines = _grep_log(lines, WAKE_KEYWORDS)
        asr_lines = _grep_log(lines, ASR_KEYWORDS)
        play_lines = _grep_log(lines, PLAY_KEYWORDS)
        tts_lines = _grep_log(lines, TTS_KEYWORDS, TTS_EXCLUDE_KEYWORDS)
        intro_lines = _grep_log(lines, INTRO_KEYWORDS)
        # Also check agent.log for intro events (generated by tool handlers)
        if AGENT_LOG_FILE.exists():
            agent_result = subprocess.run(
                ["tail", "-n", "300", str(AGENT_LOG_FILE)],
                capture_output=True, text=True
            )
            agent_lines = agent_result.stdout.splitlines()
            agent_intro_lines = _grep_log(agent_lines, INTRO_KEYWORDS)
            intro_lines = list(dict.fromkeys(intro_lines + agent_intro_lines))

        print(f"\n   关键字匹配统计 (最近 300 行):")
        print(f"   🔔 唤醒 (wake):  {len(wake_lines)} 条")
        print(f"   🎤 ASR (识别):   {len(asr_lines)} 条")
        print(f"   🎵 播放 (music):  {len(play_lines)} 条")
        print(f"   🔊 TTS (语音):   {len(tts_lines)} 条")
        print(f"   🎙️  介绍 (intro):  {len(intro_lines)} 条")

        # 分析结果
        if wake_lines:
            print(f"\n   ✅ 检测到唤醒事件！")
        else:
            print(f"\n   ⚠️  未检测到唤醒事件（可能 gateway 未运行或刚启动）")

        if play_lines:
            print(f"   ✅ 检测到音乐播放事件！")
        else:
            print(f"   ⚠️  未检测到音乐播放事件")

        if tts_lines:
            print(f"   ⚠️  检测到 TTS 事件（预期应无 TTS 回复，因为 play_music 已自声明感官反馈）")
        else:
            print(f"   ✅ 无 TTS 事件（预期行为：play_music 自声明 audio 反馈，网关抑制 TTS）")

        if intro_lines:
            print(f"   ✅ 检测到歌曲介绍播报事件！")
        else:
            print(f"   ⚠️  未检测到歌曲介绍播报（可能 intro 功能未启用或 LLM 未调用 play_music）")

        # 显示最近的唤醒相关日志
        if wake_lines:
            print(f"\n   📋 最近唤醒日志:")
            for l in wake_lines[-3:]:
                print(f"      {l[:300]}")

        if asr_lines:
            print(f"\n   📋 最近 ASR 日志:")
            for l in asr_lines[-3:]:
                print(f"      {l[:300]}")

        if tts_lines:
            print(f"\n   📋 最近 TTS 日志 (预期为空):")
            for l in tts_lines[-3:]:
                print(f"      {l[:300]}")

        if intro_lines:
            print(f"\n   📋 最近介绍播报日志:")
            for l in intro_lines[-3:]:
                print(f"      {l[:300]}")

    except Exception as e:
        print(f"   ⚠️  日志分析失败: {e}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    print("\n" + "🎤" * 30)
    print("   远程音箱唤醒测试")
    print(f"   测试文本: \"{TEST_TEXT}\"")
    print("🎤" * 30 + "\n")

    # ----- 环境检查 -----
    print("📋 环境检查:")

    dashscope_key = os.getenv("TTS_DASHSCOPE_API_KEY", "").strip()
    if dashscope_key:
        print(f"   ✅ TTS_DASHSCOPE_API_KEY: {dashscope_key[:10]}...")
    else:
        print("   ❌ TTS_DASHSCOPE_API_KEY 未设置")
        print("      请在 ~/.hermes/.env 中设置 TTS_DASHSCOPE_API_KEY=your-key")
        print("      或: export TTS_DASHSCOPE_API_KEY=your-key")
        print("      获取: https://modelstudio.console.alibabacloud.com/")
        return 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"   ✅ 输出目录: {OUTPUT_DIR}")

    # ----- 步骤 1: 生成 TTS 音频 -----
    audio_path = generate_test_audio()
    if not audio_path:
        print("\n❌ 测试中断: 无法生成 TTS 音频")
        return 1

    print(f"\n📁 文件信息:")
    print(f"   路径: {audio_path}")
    if os.path.exists(audio_path):
        size_kb = os.path.getsize(audio_path) / 1024
        print(f"   大小: {size_kb:.1f} KB")
    else:
        print(f"   ⚠️  文件不存在！")
        return 1

    # ----- 步骤 2: 检查 gateway @before play-----
    gateway_ok = check_gateway_running()
    if not gateway_ok:
        print("\n⚠️  Gateway 未运行。播放音频后无法验证唤醒效果。")
        print("   如需完整测试，请先在另一个终端执行: bash run.sh start")
        print("   本次将继续播放音频，但将跳过日志分析。")

    # ----- 等待确认 -----
    print("\n" + "=" * 60)
    print("⏳ 准备播放...")
    print("   请确保:")
    print("   1. 音箱已开机并联网")
    print("   2. 麦克风已开启")
    print("   3. 音量适中（音箱 + 本机输出）")
    print("=" * 60)

    if os.environ.get("AUTO_TEST"):
        print("   (AUTO_TEST=1，自动继续)")
    else:
        try:
            input("\n按 Enter 开始播放 (Ctrl+C 取消)...")
        except EOFError:
            print("   (非交互式环境，自动继续)")

    # 记录播放前的时间戳，用于日志分析定位
    play_time = time.strftime("%H:%M:%S")
    print(f"\n⏱️  播放时间: {play_time}")

    # ----- 步骤 3: 播放音频 -----
    if not play_audio_file(audio_path):
        print("\n❌ 测试中断: 无法播放音频")
        return 1

    # ----- 步骤 4: 等待并检查 -----
    print("\n" + "=" * 60)
    print("⏳ 等待音箱响应 (15 秒)...")
    print("   在此期间请观察音箱行为")
    print("=" * 60)

    for i in range(15, 0, -5):
        print(f"   ...{i}s")
        time.sleep(5)

    # ----- 日志分析 -----
    if gateway_ok:
        check_gateway_logs()
    else:
        print("\n⚠️  跳过日志分析 (Gateway 未运行)")

    # ----- 最终报告 -----
    print("\n" + "=" * 60)
    print("📋 测试完成 — 手动检查清单")
    print("=" * 60)
    print(f"""
   [ ] 1. 音箱是否被唤醒？ (听到"小爱同学"的唤醒提示音/灯效)
   [ ] 2. 是否开始播放王菲的《流年》？
   [ ] 3. 是否有 TTS 语音回应？ (如"好的，正在播放...")
        → 预期: 无 TTS 语音回应（play_music 工具自声明 sensory_feedback: audio，
          网关应抑制 auto-TTS）
   [ ] 4. 音乐播放是否正常？ (无杂音、不卡顿、音量正常)
   [ ] 5. 播放完毕后音箱是否回到待唤醒状态？

{"💡 Gateway 未运行，本次仅测试了 TTS 生成 + 音频播放。" if not gateway_ok else ""}
""")

    if not gateway_ok:
        print("💡 要测试完整唤醒链路，请先运行 bash run.sh start 然后重新执行本脚本。")

    print(f"💡 播放时间戳: {play_time}，可在日志中搜索该时间附近的记录。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
