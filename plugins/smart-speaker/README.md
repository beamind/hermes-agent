# Smart Speaker Plugin

本地音乐播放 + 网易云音乐回退插件，为 Hermes 提供智能音箱能力。

## 系统依赖

### Debian / Ubuntu
```bash
sudo apt-get update
sudo apt-get install -y libmpv-dev mpv
```

### macOS
```bash
brew install mpv
```

### Arch Linux
```bash
sudo pacman -S mpv
```

## Python 依赖

```bash
cd /path/to/hermes-agent
source venv/bin/activate
uv pip install python-mpv pyncm
```

## 配置

```bash
# 音乐库路径（本地文件 + 网易云下载缓存）
hermes config set smart_speaker.music_library_path /path/to/your/music
```

## 网易云音乐登录（可选）

匿名搜索和播放已可用。登录后可获取高品质 URL 和喜欢的歌单。

```bash
cd /path/to/hermes-agent
source venv/bin/activate
python plugins/smart-speaker/tools/netease_music.py login
```

按提示扫码登录，session 会自动保存在 `~/.config/hermes/netease_session.json`。

## 提供的工具

- `play_music` — 播放音乐（本地优先，无匹配自动回退网易云）
- `control_playback` — 播放控制（暂停/恢复/停止/下一首/上一首/音量）
- `get_playback_status` — 获取播放状态

### play_music 参数

- `query` — 搜索关键词（歌名、歌手或两者）
- `source` — 搜索来源：
  - `"auto"`（默认）— 先搜本地，无匹配回退网易云
  - `"local"` — 仅本地
  - `"netease"` — 仅网易云
