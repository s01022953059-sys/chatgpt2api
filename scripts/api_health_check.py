#!/usr/bin/env python3
"""第三方 API 健康检查脚本。

定期探测第三方 API 是否可用，记录状态到日志文件。
可配合 cron 定时运行：*/10 * * * * python3 /app/scripts/api_health_check.py

用法：
    python3 scripts/api_health_check.py           # 单次检查
    python3 scripts/api_health_check.py --watch   # 持续监控（每 10 分钟）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx


# 与 services/config.py::DEFAULT_OPENAI_API_USER_AGENT 保持一致；
# 留空时使用此兜底值，避免被 Cloudflare Error 1010 拦截。
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/145.0.0.0 Safari/537.36"
)


def get_config() -> dict:
    config_path = Path(os.environ.get("CHATGPT2API_DATA_DIR", "/app/data")).parent / "config.json"
    if not config_path.exists():
        config_path = Path("config.json")
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def get_api_config() -> tuple[str, str]:
    """Resolve third-party API base and key.

    Environment variables take precedence over config.json so that secrets
    can be injected without persisting to disk.
    """
    config = get_config()
    api_key = str(
        os.environ.get("CHATGPT2API_OPENAI_API_KEY")
        or config.get("openai_api_key")
        or ""
    ).strip()
    api_base = str(
        os.environ.get("CHATGPT2API_OPENAI_API_BASE")
        or config.get("openai_api_base")
        or ""
    ).strip().rstrip("/")
    return api_base, api_key


def get_user_agent() -> str:
    """Resolve User-Agent for third-party API calls.

    Environment variable `CHATGPT2API_OPENAI_API_USER_AGENT` takes precedence
    over `config.json`; falls back to DEFAULT_USER_AGENT to avoid Cloudflare
    Error 1010 (browser signature block) when none is configured.
    """
    config = get_config()
    value = str(
        os.environ.get("CHATGPT2API_OPENAI_API_USER_AGENT")
        or config.get("openai_api_user_agent")
        or ""
    ).strip()
    return value or DEFAULT_USER_AGENT


def get_proxy() -> str:
    """Resolve outbound proxy for third-party API calls (empty = direct).

    Environment variable `CHATGPT2API_OPENAI_API_PROXY` takes precedence over
    `config.json`. Used when the host IP is blocked by the upstream WAF.
    """
    config = get_config()
    return str(
        os.environ.get("CHATGPT2API_OPENAI_API_PROXY")
        or config.get("openai_api_proxy")
        or ""
    ).strip()


def check_api(api_base: str, api_key: str, user_agent: str = DEFAULT_USER_AGENT) -> dict:
    """检查 API 状态，返回结果 dict。"""
    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "api_base": api_base,
    }
    try:
        resp = httpx.get(
            f"{api_base}/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": user_agent,
            },
            timeout=15,
            proxy=get_proxy() or None,
        )
        result["status_code"] = resp.status_code
        result["ok"] = resp.status_code == 200
        if resp.status_code == 401:
            result["category"] = "invalid_api_key"
        elif resp.status_code == 403:
            result["category"] = "permission_or_balance"
        elif resp.status_code == 429:
            result["category"] = "rate_limited"
        elif resp.status_code >= 500:
            result["category"] = "upstream_error"
        else:
            result["category"] = "ok" if result["ok"] else "request_error"
        if resp.status_code != 200:
            try:
                err = resp.json()
                result["error"] = str(err.get("error", {}).get("message", "")) or str(err)
            except Exception:
                result["error"] = resp.text[:200]
        else:
            try:
                models = resp.json().get("data", [])
                result["models_count"] = len(models)
            except Exception:
                pass
    except httpx.TimeoutException:
        result["ok"] = False
        result["category"] = "upstream_error"
        result["error"] = "请求超时（15s）"
    except Exception as e:
        result["ok"] = False
        result["category"] = "upstream_error"
        result["error"] = str(e)[:200]
    return result


def main():
    parser = argparse.ArgumentParser(description="第三方 API 健康检查")
    parser.add_argument("--watch", action="store_true", help="持续监控模式")
    parser.add_argument("--interval", type=int, default=600, help="监控间隔（秒，默认 600）")
    args = parser.parse_args()

    api_base, api_key = get_api_config()
    user_agent = get_user_agent()

    if not api_key or not api_base:
        print("未配置 openai_api_key 或 openai_api_base，退出")
        sys.exit(1)

    log_path = Path(os.environ.get("CHATGPT2API_DATA_DIR", "/app/data")) / "api_health.log"

    def run_once():
        result = check_api(api_base, api_key, user_agent)
        status = "✓ OK" if result.get("ok") else f"✗ FAIL: {result.get('error', 'unknown')}"
        line = f"[{result['timestamp']}] {status}"
        print(line)
        with open(log_path, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        # 保留最近 1000 条
        try:
            lines = log_path.read_text().strip().split("\n")
            if len(lines) > 1000:
                log_path.write_text("\n".join(lines[-1000:]) + "\n")
        except Exception:
            pass
        return result

    if args.watch:
        print(f"持续监控模式，间隔 {args.interval} 秒，日志: {log_path}")
        while True:
            run_once()
            time.sleep(args.interval)
    else:
        result = run_once()
        sys.exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()