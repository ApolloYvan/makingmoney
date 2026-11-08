#!/usr/bin/env python3
"""把视频链接（或本地音视频文件）变成带时间戳的中文文字稿。

用法：
    python3 video2text.py <视频链接或本地文件> [--out 目录] [--model small|medium]
                          [--cookies-browser chrome|safari|firefox|edge] [--lang zh] [--keep-proxy]

顺序：读视频信息 → 有字幕就直接用字幕 → 没有就下载音频 → 转成 16kHz 单声道 wav →
faster-whisper 本地转写 → 检查结果是否像样。只用标准库和 faster-whisper，兼容 Python 3.9。
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

PROXY_VARS = ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
SUB_LANGS = "ai-zh,zh-Hans,zh-CN,zh,ai-en,en"
MIN_CHARS = 50


def fail(msg, code=1):
    print("错误：" + msg, file=sys.stderr)
    sys.exit(code)


def check_deps(need_download):
    missing = []
    if need_download and not shutil.which("yt-dlp"):
        missing.append("yt-dlp（brew install yt-dlp）")
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg（brew install ffmpeg）")
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        missing.append("faster-whisper（pip3 install faster-whisper）")
    if missing:
        fail("缺少依赖：" + "、".join(missing))


def run(cmd, env):
    p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return p.returncode, p.stdout, p.stderr


def explain_download_error(stderr):
    if "567" in stderr or "412" in stderr:
        return ("视频网站拒绝了下载（HTTP 567/412）。常见原因是请求走了境外代理或者在云端运行。"
                "请在国内网络、关掉代理的电脑上运行；还不行就加 --cookies-browser chrome 用本机浏览器的登录状态。")
    if any(k in stderr for k in ("CERTIFICATE_VERIFY_FAILED", "Connection refused", "Network is unreachable", "timed out", "Name or service not known")):
        return ("连不上网站。如果这台机器必须通过代理才能上网（比如云端会话、公司网络），加 --keep-proxy 再试；"
                "国内网站本身更适合在关掉代理的电脑上运行。")
    if "premium" in stderr.lower() or "大会员" in stderr:
        return "这个视频需要会员才能看，不处理会员内容。"
    return "下载失败，yt-dlp 的输出最后几行：\n" + "\n".join(stderr.strip().splitlines()[-5:])


def safe_name(s):
    s = re.sub(r'[\\/:*?"<>|\s]+', "_", s).strip("_")
    return s[:60] or "video"


def fmt_ts(sec):
    sec = int(sec)
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return "%d:%02d:%02d" % (h, m, s) if h else "%02d:%02d" % (m, s)


def srt_to_lines(text):
    """SRT/VTT 字幕转成 [mm:ss] 文本行，去掉序号和重复行。"""
    lines, ts, last = [], None, None
    for raw in text.splitlines():
        line = raw.strip()
        m = re.match(r"(\d+):(\d\d):(\d\d)[,.]\d+\s*-->", line) or re.match(r"(\d\d):(\d\d)[,.]\d+\s*-->", line)
        if m:
            g = [int(x) for x in m.groups()]
            ts = g[0] * 3600 + g[1] * 60 + g[2] if len(g) == 3 else g[0] * 60 + g[1]
            continue
        if not line or line.isdigit() or line.startswith(("WEBVTT", "NOTE", "Kind:", "Language:")):
            continue
        line = re.sub(r"<[^>]+>", "", line)
        if line and line != last:
            lines.append("[%s] %s" % (fmt_ts(ts or 0), line))
            last = line
    return lines


def transcribe(wav, model_name, lang):
    import numpy as np
    from faster_whisper import WhisperModel
    # macOS 上新版 numpy 调用 Accelerate 时会报 divide by zero / overflow 之类的 RuntimeWarning，
    # 实测不影响结果，这里屏蔽掉，免得被误认为出错。结果好坏由后面的字数检查判断。
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print("加载语音模型 %s（第一次运行会先下载，small 约 500MB）……" % model_name, file=sys.stderr)
    model = WhisperModel(model_name, compute_type="int8")
    prompt = "以下是普通话的句子，使用简体中文。" if lang == "zh" else None
    # 自己读 16kHz 单声道 wav 交给模型，不走 faster-whisper 自带的 PyAV 解码：新版 PyAV 和它不兼容。
    import wave
    with wave.open(str(wav), "rb") as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    with np.errstate(all="ignore"):
        segments, info = model.transcribe(audio, language=lang, vad_filter=True, initial_prompt=prompt)
        lines = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                lines.append("[%s] %s" % (fmt_ts(seg.start), text))
                print("\r已转写到 %s" % fmt_ts(seg.end), end="", file=sys.stderr)
    print("", file=sys.stderr)
    return lines


def main():
    ap = argparse.ArgumentParser(description="视频链接或本地文件 → 文字稿")
    ap.add_argument("source", help="视频链接（B 站、YouTube 等 yt-dlp 支持的网站）或本地音视频文件")
    ap.add_argument("--out", default=str(Path.home() / "Downloads" / "视频文字稿"), help="输出的上级目录")
    ap.add_argument("--model", default="small", help="faster-whisper 模型：small（默认）或 medium（更准、更慢）")
    ap.add_argument("--cookies-browser", help="用本机浏览器的登录状态，如 chrome、safari；只在本机读取")
    ap.add_argument("--lang", default="zh", help="语言，默认 zh")
    ap.add_argument("--keep-proxy", action="store_true", help="保留终端里的代理设置（默认清掉，国内网站走代理常被拒绝）")
    ap.add_argument("--no-subs", action="store_true", help="不用字幕，直接转写音频")
    a = ap.parse_args()

    local = Path(a.source).expanduser()
    is_local = local.exists()
    check_deps(need_download=not is_local)

    env = dict(os.environ)
    if not a.keep_proxy:
        for k in PROXY_VARS:
            env.pop(k, None)
    cookies = ["--cookies-from-browser", a.cookies_browser] if a.cookies_browser else []

    meta = {"source": a.source}
    if is_local:
        meta.update(title=local.stem)
    else:
        code, out, err = run(["yt-dlp", "-J", "--no-playlist", "--no-warnings"] + cookies + [a.source], env)
        if code != 0:
            fail(explain_download_error(err))
        info = json.loads(out)
        meta.update({k: info.get(k) for k in ("id", "title", "uploader", "duration", "upload_date", "view_count", "webpage_url", "description")})

    outdir = Path(a.out).expanduser() / safe_name("%s_%s" % (meta.get("upload_date") or "", meta.get("title") or ""))
    outdir.mkdir(parents=True, exist_ok=True)
    lines, method = [], None

    if not is_local and not a.no_subs:
        code, _, err = run(["yt-dlp", "--skip-download", "--write-subs", "--sub-langs", SUB_LANGS, "--no-playlist",
                            "-o", str(outdir / "sub")] + cookies + [a.source], env)
        subs = sorted(p for p in outdir.glob("sub*") if p.suffix in (".srt", ".vtt"))
        if subs:
            lines = srt_to_lines(subs[0].read_text(encoding="utf-8", errors="ignore"))
            method = "字幕：" + subs[0].name

    if len("".join(lines)) < MIN_CHARS:
        if is_local:
            media = local
        else:
            code, _, err = run(["yt-dlp", "-f", "ba/b", "--no-playlist", "-o", str(outdir / "audio.%(ext)s")] + cookies + [a.source], env)
            found = [p for p in outdir.glob("audio.*") if p.suffix != ".wav"]
            if code != 0 or not found:
                fail(explain_download_error(err))
            media = found[0]
        wav = outdir / "audio.wav"
        code, _, err = run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(media), "-ar", "16000", "-ac", "1", str(wav)], env)
        if code != 0:
            fail("ffmpeg 转换失败：" + err.strip()[-300:])
        lines = transcribe(wav, a.model, a.lang)
        method = "语音转写：faster-whisper %s" % a.model
        wav.unlink()

    chars = len("".join(re.sub(r"^\[[\d:]+\] ", "", x) for x in lines))
    meta.update(method=method, chars=chars)
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "transcript.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if chars < MIN_CHARS:
        fail("只识别出 %d 个字，结果不可用。可能是视频没有人声，或者音频有问题；可以换 --model medium 再试。文件在 %s" % (chars, outdir), 2)
    dur = meta.get("duration")
    if dur and chars < dur * 1.0:
        print("提醒：%d 秒的视频只识别出 %d 个字，偏少，可能有大段音乐或者识别不全。" % (int(dur), chars), file=sys.stderr)
    print("完成（%s，%d 字）" % (method, chars), file=sys.stderr)
    print(outdir / "transcript.txt")


if __name__ == "__main__":
    main()
