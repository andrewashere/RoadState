#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from openai import OpenAI


def file_url(path: str) -> str:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Video not found: {p}")
    return f"file://{p.as_posix()}"


def load_prompt(prompt: str | None, prompt_yaml: str | None) -> str:
    if prompt:
        return prompt.strip()
    if prompt_yaml:
        data = yaml.safe_load(Path(prompt_yaml).read_text())
        if isinstance(data, dict):
            for k in ("user_prompt", "prompt", "text"):
                v = data.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        if isinstance(data, str) and data.strip():
            return data.strip()
        raise ValueError(f"Unrecognized YAML format: {prompt_yaml}")
    raise ValueError("Provide --prompt or --prompt-yaml")


def main() -> int:
    ap = argparse.ArgumentParser(description="Send one video + prompt to vLLM (OpenAI-compatible).")
    ap.add_argument("--server", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="nvidia/Cosmos-Reason2-8B")
    ap.add_argument("--video", required=True, help="Video path on THIS machine (must be readable by vLLM allowed-local-media-path).")
    ap.add_argument("--prompt", help="Inline prompt text.")
    ap.add_argument("--prompt-yaml", help="YAML with user_prompt/prompt/text.")
    ap.add_argument("--fps", type=float, default=None, help="Sample FPS (sent as mm_processor_kwargs).")
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--json", action="store_true", help="Output JSON instead of raw text.")
    args = ap.parse_args()

    prompt_text = load_prompt(args.prompt, args.prompt_yaml)
    video_path = str(Path(args.video).expanduser().resolve())
    video = file_url(video_path)

    client = OpenAI(api_key="EMPTY", base_url=args.server)

    extra_body = {}
    if args.fps is not None:
        extra_body["mm_processor_kwargs"] = {"fps": args.fps, "do_sample_frames": True}

    resp = client.chat.completions.create(
        model=args.model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "video_url", "video_url": {"url": video}},
            ],
        }],
        max_completion_tokens=args.max_tokens,
        extra_body=extra_body if extra_body else None,
    )

    content = None
    if getattr(resp, "choices", None):
        content = resp.choices[0].message.content

    if args.json:
        out = {
            "server": args.server,
            "model": args.model,
            "video": video_path,
            "video_url": video,
            "max_tokens": args.max_tokens,
            "fps": args.fps,
            "prompt_yaml": args.prompt_yaml,
            "result": content,
        }
        print(json.dumps(out, indent=2))
        return 0

   
    if content is None:
        print(resp.model_dump_json(indent=2))
        return 2
    print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())