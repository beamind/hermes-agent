# Smart Speaker Plugin

本地音乐播放插件，为 Hermes 提供智能音箱能力。

## 系统依赖

本插件需要以下系统级依赖（**不会自动安装，需手动配置**）：

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

Hermes 使用 `uv` 管理虚拟环境，安装方式如下：

```bash
cd /path/to/hermes-agent
VIRTUAL_ENV=$(pwd)/venv uv pip install python-mpv
```

> 如果 `uv` 提示找不到虚拟环境，确认先 `source venv/bin/activate` 再运行。

## 配置

```bash
hermes config set smart_speaker.music_library_path /path/to/your/music
```

## 提供的工具

- `play_music` — 播放音乐（支持本地库搜索）
- `control_playback` — 播放控制（暂停/恢复/下一首/音量）
- `get_playback_status` — 获取播放状态
