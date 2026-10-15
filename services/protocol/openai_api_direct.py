"""Direct OpenAI-compatible API adapter for image generation/editing.

When a third-party API key is configured (openai_api_key in config.json),
image generation and editing requests bypass the account pool and are
forwarded directly to the configured API endpoint using standard
OpenAI-compatible /v1/images/generations and /v1/images/edits endpoints.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
import uuid
from typing import Any, Iterator

import httpx

from services.config import config
from services.protocol.conversation import ImageOutput, save_image_bytes
from utils.log import logger


def _is_direct_mode() -> bool:
    """Check if direct API key mode is enabled."""
    return bool(config.openai_api_key)


def _get_api_base() -> str:
    """Return the API base URL, stripped of trailing slash."""
    return str(config.openai_api_base or "https://api.openai.com").strip().rstrip("/")


def _get_api_key() -> str:
    return str(config.openai_api_key or "").strip()


def _get_user_agent() -> str:
    """Return the User-Agent string for third-party OpenAI-compatible API calls.

    Reads from `config.openai_api_user_agent` (env var
    `CHATGPT2API_OPENAI_API_USER_AGENT` or `config.json` field), falling back
    to a built-in Chrome desktop UA so that Cloudflare Error 1010 (browser
    signature block) is not triggered by the default `python-httpx/<ver>` UA.
    """
    return str(getattr(config, "openai_api_user_agent", "") or "").strip()


def _get_proxy() -> str:
    """Return the outbound proxy URL for third-party API calls (empty = direct).

    Reads `config.openai_api_proxy` (env `CHATGPT2API_OPENAI_API_PROXY` or
    `config.json` field). Used when the VPS datacenter IP is blocked by the
    upstream's Cloudflare WAF (HTTP 403 "Sorry, you have been blocked").
    """
    return str(getattr(config, "openai_api_proxy", "") or "").strip()


# Pro 模型精度映射: 2K -> 2048x2048, 4K -> 2560x2560 (API 最大可用)
PRO_SIZE_MAP = {
    "2K": "2048x2048",
    "4K": "2560x2560",
}


def _diagnostic_id() -> str:
    return uuid.uuid4().hex[:12]


def _safe_body_preview(resp: httpx.Response, limit: int = 300) -> str:
    """Return a bounded response preview without image payloads."""
    try:
        value = resp.json()
        if isinstance(value, dict):
            sanitized: dict[str, Any] = {"keys": sorted(str(key) for key in value)}
            error = value.get("error")
            if isinstance(error, dict):
                sanitized["error"] = {
                    key: str(error[key])[:180]
                    for key in ("type", "code", "message")
                    if key in error
                }
            for key in ("task_id", "id", "status", "message", "code"):
                if key in value and key not in sanitized:
                    sanitized[key] = str(value[key])[:180]
            return json.dumps(sanitized, ensure_ascii=False)[:limit]
        return json.dumps({"type": type(value).__name__}, ensure_ascii=False)[:limit]
    except Exception:
        return " ".join(resp.text[:limit].split())


def _response_headers(resp: httpx.Response) -> dict[str, str]:
    return {
        key: value
        for key in ("request-id", "x-request-id", "cf-ray", "content-type")
        if (value := resp.headers.get(key))
    }


def _resolve_size(model: str, size: str | None) -> str | None:
    """根据模型和精度配置解析最终 size。

    - gpt-image-2-pro: 根据 config.openai_api_pro_size 自动设置 2K/4K
    - 其他模型: 用用户传入的 size，默认 1024x1024
    """
    if model == "gpt-image-2-pro":
        pro_size = getattr(config, "openai_api_pro_size", "2K")
        return PRO_SIZE_MAP.get(pro_size, "2048x2048")
    # Legacy UI presets used rounded ratios that this provider rejects.
    if model == "gpt-image-2.5-flare":
        size = {
            "1024x1365": "768x1024",
            "1365x1024": "1024x768",
            "1920x1088": "1536x864",
            "1088x1920": "864x1536",
        }.get(size, size)
    return size or "1024x1024"


def _fetch_image_as_b64(url: str) -> str:
    """Download an image from URL and return base64-encoded string.

    Handles three cases:
    1. data URI (data:image/png;base64,...) — extract base64 directly,
       no HTTP request needed. Some third-party APIs return data URIs
       in the "url" field when response_format=url, which can be 1MB+
       and would crash httpx with "URL too long".
    2. HTTP/HTTPS URL — download via httpx.get.
    3. Raw base64 string (no scheme prefix) — return as-is.
    """
    if url.startswith("data:"):
        # data:image/png;base64,iVBOR...
        comma_idx = url.find(",")
        if comma_idx != -1:
            return url[comma_idx + 1:]
        return ""
    if url.startswith("http://") or url.startswith("https://"):
        resp = httpx.get(url, timeout=120, follow_redirects=True, proxy=_get_proxy() or None)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode("ascii")
    # Assume raw base64
    return url


def _poll_direct_image_result(
    initial_data: dict[str, Any],
    api_key: str,
    diagnostic_id: str,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Poll third-party asynchronous image tasks until completion."""
    status = str(initial_data.get("status") or "").lower()
    poll_url = str(initial_data.get("poll_url") or "").strip()
    if status not in {"queued", "pending", "processing", "running"} or not poll_url:
        return initial_data

    if progress_callback:
        progress_callback("api_direct_polling")
    deadline = time.monotonic() + 1200
    attempt = 0
    # 轮询请求使用分离的 connect/read 超时，避免因上游不可达等原因
    # 每个轮询周期阻塞 60s。connect 10s 足够检测网络问题。
    poll_timeout = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=10.0)
    while time.monotonic() < deadline:
        attempt += 1
        time.sleep(5)
        try:
            resp = httpx.get(
                poll_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": _get_user_agent(),
                },
                timeout=poll_timeout,
                follow_redirects=True,
                proxy=_get_proxy() or None,
            )
        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
            logger.warning({
                "event": "api_direct_poll_network_error",
                "diagnostic_id": diagnostic_id,
                "attempt": attempt,
                "error": str(exc)[:100],
            })
            continue
        if resp.status_code != 200:
            raise RuntimeError(
                f"第三方 API 异步任务轮询失败 (HTTP {resp.status_code})，诊断 ID: {diagnostic_id}。"
            )
        try:
            data = resp.json()
        except Exception as exc:
            raise RuntimeError(f"第三方 API 轮询返回了无法解析的 JSON，诊断 ID: {diagnostic_id}。") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"第三方 API 轮询返回格式不兼容，诊断 ID: {diagnostic_id}。")
        task_status = str(data.get("status") or "").lower()
        logger.debug({
            "event": "api_direct_poll",
            "diagnostic_id": diagnostic_id,
            "attempt": attempt,
            "status": task_status,
            "progress": data.get("progress"),
        })
        if task_status in {"completed", "succeeded", "success"}:
            return data
        if task_status in {"failed", "error", "cancelled", "canceled"}:
            error = data.get("error") or data.get("message") or "unknown error"
            raise RuntimeError(f"第三方 API 图片任务失败，诊断 ID: {diagnostic_id}：{str(error)[:300]}")

    raise RuntimeError(f"第三方 API 图片任务轮询超时，诊断 ID: {diagnostic_id}。")


def _generate_images_direct(
    prompt: str,
    model: str,
    n: int,
    size: str | None,
    quality: str,
    response_format: str,
    base_url: str | None,
    progress_callback: Any = None,
    reference_images: list[str] | None = None,
) -> list[ImageOutput]:
    """Call the third-party API for image generation, return ImageOutput list.

    `reference_images`（可选）为 base64 图片列表；提供时通过 generations
    端点的 image 字段附带参考图（部分第三方网关以此支持图生图）。
    """
    api_base = _get_api_base()
    api_key = _get_api_key()

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "n": n,
    }
    resolved_size = _resolve_size(model, size)
    if resolved_size:
        payload["size"] = resolved_size
    if quality and quality != "auto":
        payload["quality"] = quality
    if reference_images:
        payload["image"] = [f"data:image/png;base64,{item}" for item in reference_images if item]

    if progress_callback:
        progress_callback("api_direct_requesting")

    diagnostic_id = _diagnostic_id()
    started_at = time.monotonic()
    logger.debug({
        "event": "api_direct_image_generation",
        "diagnostic_id": diagnostic_id,
        "api_base": api_base,
        "model": model,
        "prompt_preview": prompt[:100],
        "n": n,
        "size": size,
        "resolved_size": resolved_size,
    })

    # 第三方 API 网关（如 Cloudflare）可能因处理时间长而返回 524/504/522，
    # 或在传输过程中断开连接。复杂 prompt + Pro 模式可能需要 10-20 分钟，
    # 因此放宽到 1200s（20 分钟）。
    # 524 重试 4 次、间隔 45s：524 表示源站 100 秒未处理完，立即重试
    # 大概率继续排队超时；拉长间隔给源站更多恢复时间能提高重试成功率。
    max_retries = 4
    retry_interval = 10
    gateway_retry_interval = 45
    # 分离 connect 和 read 超时：connect 10s 快速检测网络不可达，
    # read 1200s 给慢速图片生成充足时间，避免因上游不可达导致
    # 每个请求阻塞 20 分钟（httpx 的 timeout=1200 整数会把 connect
    # 也设成 1200s）。
    request_timeout = httpx.Timeout(connect=10.0, read=1200.0, write=1200.0, pool=10.0)
    resp = None
    for attempt in range(max_retries):
        try:
            resp = httpx.post(
                f"{api_base}/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": _get_user_agent(),
                },
                json=payload,
                timeout=request_timeout,
                proxy=_get_proxy() or None,
            )
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError, httpx.TimeoutException) as exc:
            # 连接中途断开 / 连接超时 / 读取超时：API 不可达或传输中断
            if attempt < max_retries - 1:
                logger.info({"event": "api_direct_conn_retry", "attempt": attempt + 1, "error": str(exc)[:100]})
                time.sleep(retry_interval)
                continue
            raise RuntimeError(
                f"第三方 API 连接中断：{str(exc)[:150]}。"
                f"API 服务端在传输图片数据时断开连接，可能是处理超时或服务不稳定。建议稍后重试。"
            ) from exc
        # Only retry on gateway timeouts; 403/429 etc are non-recoverable
        if resp.status_code == 200 or resp.status_code not in (504, 524, 522):
            break
        if attempt < max_retries - 1:
            logger.info({"event": "api_direct_retry", "attempt": attempt + 1, "status": resp.status_code})
            time.sleep(gateway_retry_interval)

    if resp.status_code != 200:
        error_detail = ""
        try:
            error_body = resp.json()
            error_detail = str(error_body.get("error", {}).get("message", "")) or str(error_body)
        except Exception:
            error_detail = resp.text[:500]
        if resp.status_code == 400 and "unsupported size ratio" in error_detail.lower():
            raise RuntimeError(
                f"上游不支持图片尺寸 {resolved_size}，请更换图片比例或分辨率。"
                f"诊断 ID: {diagnostic_id}；详情: {error_detail}"
            )
        if resp.status_code in (504, 524, 522):
            raise RuntimeError(
                f"第三方 API 处理超时 (HTTP {resp.status_code})，诊断 ID: {diagnostic_id}。"
                f"已重试 {max_retries} 次。复杂 prompt 或 Pro 2K/4K 模式在第三方网关上处理时间可能较长。"
                f"建议：稍后重试，或简化 prompt/降低模型复杂度。详情: {error_detail}"
            )
        if resp.status_code == 401:
            raise RuntimeError(
                f"第三方 API Key 无效或未授权 (HTTP 401)，诊断 ID: {diagnostic_id}。详情: {error_detail}"
            )
        if resp.status_code == 403:
            raise RuntimeError(
                f"第三方 API 拒绝访问 (HTTP 403)，诊断 ID: {diagnostic_id}：{error_detail}。"
                f"通常是账户余额不足、模型无权限或 API Key 已失效，请检查第三方 API 账户状态。"
            )
        if resp.status_code == 429:
            raise RuntimeError(
                f"第三方 API 请求频率超限 (HTTP 429)，诊断 ID: {diagnostic_id}：{error_detail}。"
                f"请稍后重试，或降低请求频率。"
            )
        raise RuntimeError(f"API direct image generation failed (HTTP {resp.status_code}), 诊断 ID: {diagnostic_id}: {error_detail}")

    try:
        data = resp.json()
    except Exception as exc:
        logger.warning({
            "event": "api_direct_invalid_json",
            "diagnostic_id": diagnostic_id,
            "status": resp.status_code,
            "headers": _response_headers(resp),
            "body_preview": _safe_body_preview(resp),
            "duration_ms": int((time.monotonic() - started_at) * 1000),
        })
        raise RuntimeError(f"第三方 API 返回了无法解析的 JSON，诊断 ID: {diagnostic_id}。") from exc

    if not isinstance(data, dict):
        logger.warning({
            "event": "api_direct_incompatible_response",
            "diagnostic_id": diagnostic_id,
            "status": resp.status_code,
            "response_type": type(data).__name__,
            "headers": _response_headers(resp),
        })
        raise RuntimeError(f"第三方 API 返回格式不兼容，诊断 ID: {diagnostic_id}。")

    # 第三方 API 的异步任务可能因内部超时返回 "failed"，
    # 这不是我们的问题而是上游处理慢。自动重新创建任务重试。
    poll_max_retries = 2
    for poll_attempt in range(poll_max_retries):
        try:
            data = _poll_direct_image_result(data, api_key, diagnostic_id, progress_callback)
            break
        except RuntimeError as exc:
            if "图片任务失败" not in str(exc) or poll_attempt >= poll_max_retries - 1:
                raise
            logger.info({
                "event": "api_direct_poll_retry",
                "diagnostic_id": diagnostic_id,
                "poll_attempt": poll_attempt + 1,
                "error": str(exc)[:200],
            })
            # 重新创建任务（同样使用分离的 connect/read 超时，并捕获连接异常）
            try:
                resp = httpx.post(
                    f"{api_base}/v1/images/generations",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "User-Agent": _get_user_agent(),
                    },
                    json=payload,
                    timeout=request_timeout,
                    proxy=_get_proxy() or None,
                )
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError, httpx.TimeoutException) as retry_exc:
                raise RuntimeError(
                    f"第三方 API 重试创建任务失败（连接异常），诊断 ID: {diagnostic_id}。"
                ) from retry_exc
            if resp.status_code != 200:
                raise RuntimeError(
                    f"第三方 API 重试创建任务失败 (HTTP {resp.status_code})，诊断 ID: {diagnostic_id}。"
                ) from exc
            data = resp.json()
            time.sleep(5)
    else:
        raise RuntimeError(f"第三方 API 图片任务多次重试后仍失败，诊断 ID: {diagnostic_id}。")
    images = data.get("data")
    if images is None:
        logger.warning({
            "event": "api_direct_incompatible_response",
            "diagnostic_id": diagnostic_id,
            "status": resp.status_code,
            "response_keys": sorted(str(key) for key in data),
            "body_preview": _safe_body_preview(resp),
            "headers": _response_headers(resp),
        })
        raise RuntimeError(f"第三方 API 返回格式不兼容，未找到 data 图片字段，诊断 ID: {diagnostic_id}。")
    if not isinstance(images, list):
        raise RuntimeError(f"第三方 API 返回格式不兼容，data 不是数组，诊断 ID: {diagnostic_id}。")
    images = images or []

    if progress_callback:
        progress_callback("api_direct_processing")

    outputs: list[ImageOutput] = []
    for i, img in enumerate(images):
        b64 = str(img.get("b64_json") or "").strip()
        if not b64 and img.get("url"):
            b64 = _fetch_image_as_b64(str(img["url"]))

        if not b64:
            logger.warning({"event": "api_direct_empty_image", "index": i, "img_keys": list(img.keys())})
            continue

        saved_url = save_image_bytes(base64.b64decode(b64), base_url)
        outputs.append(ImageOutput(
            kind="result",
            model=model,
            index=i,
            total=len(images),
            data=[{
                "b64_json": b64,
                "url": saved_url,
                "revised_prompt": str(img.get("revised_prompt") or prompt),
            }],
        ))

    if not outputs:
        logger.warning({
            "event": "api_direct_empty_result",
            "diagnostic_id": diagnostic_id,
            "status": resp.status_code,
            "image_count": len(images),
            "response_keys": sorted(str(key) for key in data),
            "body_preview": _safe_body_preview(resp),
            "headers": _response_headers(resp),
            "duration_ms": int((time.monotonic() - started_at) * 1000),
        })
        raise RuntimeError(f"API direct: no images returned，诊断 ID: {diagnostic_id}")

    return outputs


def _edit_images_direct(
    prompt: str,
    images_b64: list[str],
    model: str,
    n: int,
    size: str | None,
    quality: str,
    response_format: str,
    base_url: str | None,
    progress_callback: Any = None,
) -> list[ImageOutput]:
    """Call the third-party API for image editing, return ImageOutput list."""
    api_base = _get_api_base()
    api_key = _get_api_key()
    # This provider supports 2.5 reference images through generations, not edits.
    if model == "gpt-image-2.5-flare":
        return _generate_images_direct(prompt=prompt, model=model, n=n, size=size,
            quality=quality, response_format=response_format, base_url=base_url,
            progress_callback=progress_callback, reference_images=images_b64)
    original_model = model

    # 保留旧 pro 模型的兼容回退；其他模型不改名。
    if model == "gpt-image-2-pro":
        logger.info({
            "event": "api_direct_edit_model_fallback",
            "original_model": model,
            "fallback_model": "gpt-image-2",
            "reason": "third-party API /v1/images/edits only supports gpt-image-2",
        })
        model = "gpt-image-2"

    if progress_callback:
        progress_callback("api_direct_edit_requesting")

    diagnostic_id = _diagnostic_id()
    logger.debug({
        "event": "api_direct_image_edit",
        "diagnostic_id": diagnostic_id,
        "api_base": api_base,
        "model": model,
        "prompt_preview": prompt[:100],
        "num_images": len(images_b64),
    })

    files: list[tuple[str, tuple[str, bytes, str]]] = []
    for i, b64 in enumerate(images_b64):
        img_bytes = base64.b64decode(b64)
        files.append(("image", (f"image_{i}.png", img_bytes, "image/png")))

    data: dict[str, str] = {
        "model": model,
        "prompt": prompt,
    }
    data["n"] = str(n)
    data["response_format"] = response_format or "b64_json"
    # 注意：第三方 API 的 /v1/images/edits 端点不接受 size 参数。
    # 当 size 包含在 multipart 表单中时，API 返回 400 "NextPart: EOF"。
    # 编辑功能由原图尺寸决定输出，无需指定 size。
    if quality and quality != "auto":
        data["quality"] = quality

    # 第三方 API 网关（如 Cloudflare）可能因图片编辑处理时间长而返回 524/504/522，
    # 或在传输过程中断开连接。复杂 prompt 可能需要 10-20 分钟，超时放宽到 1200s。
    # 524 重试 4 次、间隔 45s：给第三方服务端更多恢复时间。
    max_retries = 4
    retry_interval = 10
    gateway_retry_interval = 45
    # 分离 connect 和 read 超时：connect 10s 快速检测网络不可达，
    # read 1200s 给慢速图片编辑充足时间。
    request_timeout = httpx.Timeout(connect=10.0, read=1200.0, write=1200.0, pool=10.0)
    resp = None
    for attempt in range(max_retries):
        try:
            resp = httpx.post(
                f"{api_base}/v1/images/edits",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": _get_user_agent(),
                },
                data=data,
                files=files,
                timeout=request_timeout,
                proxy=_get_proxy() or None,
            )
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError, httpx.TimeoutException) as exc:
            if attempt < max_retries - 1:
                logger.info({"event": "api_direct_edit_conn_retry", "attempt": attempt + 1, "error": str(exc)[:100]})
                time.sleep(retry_interval)
                continue
            raise RuntimeError(
                f"第三方 API 连接中断：{str(exc)[:150]}。"
                f"API 服务端在传输图片数据时断开连接，可能是处理超时或服务不稳定。建议稍后重试。"
            ) from exc
        if resp.status_code == 200 or resp.status_code not in (504, 524, 522):
            break
        if attempt < max_retries - 1:
            logger.info({"event": "api_direct_edit_retry", "attempt": attempt + 1, "status": resp.status_code})
            time.sleep(gateway_retry_interval)

    if resp is not None and resp.status_code != 200:
        error_detail = ""
        try:
            error_body = resp.json()
            error_detail = str(error_body.get("error", {}).get("message", "")) or str(error_body)
        except Exception:
            error_detail = resp.text[:500]
        if resp.status_code == 404:
            # 上游网关关闭了 /v1/images/edits 端点（"unsupported path"/"not found"）。
            # 降级到 generations 端点，用 image 字段承载参考图（部分网关
            # 以此支持图生图），并恢复用户原始模型（generations 支持 pro）。
            logger.warning({
                "event": "api_direct_edit_endpoint_missing_fallback",
                "diagnostic_id": diagnostic_id,
                "status": resp.status_code,
                "error_detail": error_detail[:120],
                "fallback": "generations + image field",
                "original_model": original_model,
            })
            if progress_callback:
                progress_callback("api_direct_edit_via_generations")
            return _generate_images_direct(
                prompt=prompt,
                model=original_model,
                n=n,
                size=size,
                quality=quality,
                response_format=response_format,
                base_url=base_url,
                progress_callback=progress_callback,
                reference_images=images_b64,
            )
        if resp.status_code in (504, 524, 522):
            raise RuntimeError(
                f"第三方 API 处理超时 (HTTP {resp.status_code})，诊断 ID: {diagnostic_id}。"
                f"已重试 {max_retries} 次。复杂 prompt 或多图合成在第三方网关上处理时间可能较长。"
                f"建议：稍后重试，或简化 prompt/减少参考图。详情: {error_detail}"
            )
        if resp.status_code == 403:
            raise RuntimeError(
                f"第三方 API 拒绝访问 (HTTP 403)：{error_detail}。"
                f"通常是账户余额不足或 API Key 已失效，请检查第三方 API 账户状态。"
            )
        if resp.status_code == 429:
            raise RuntimeError(
                f"第三方 API 请求频率超限 (HTTP 429)：{error_detail}。"
                f"请稍后重试，或降低请求频率。"
            )
        raise RuntimeError(f"API direct image edit failed (HTTP {resp.status_code}): {error_detail}")

    resp_data = resp.json()

    # 如果第三方 API 返回 poll_url（异步模式），需要轮询等待结果
    poll_max_retries = 2
    for poll_attempt in range(poll_max_retries):
        try:
            resp_data = _poll_direct_image_result(resp_data, api_key, diagnostic_id, progress_callback)
            break
        except RuntimeError as exc:
            if "图片任务失败" not in str(exc) or poll_attempt >= poll_max_retries - 1:
                raise
            logger.info({
                "event": "api_direct_edit_poll_retry",
                "diagnostic_id": diagnostic_id,
                "poll_attempt": poll_attempt + 1,
                "error": str(exc)[:200],
            })
            # 重新创建编辑任务（同样使用分离的 connect/read 超时，并捕获连接异常）
            try:
                resp = httpx.post(
                    f"{api_base}/v1/images/edits",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "User-Agent": _get_user_agent(),
                    },
                    data=data,
                    files=files,
                    timeout=request_timeout,
                    proxy=_get_proxy() or None,
                )
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError, httpx.TimeoutException) as retry_exc:
                raise RuntimeError(
                    f"第三方 API 重试创建编辑任务失败（连接异常），诊断 ID: {diagnostic_id}。"
                ) from retry_exc
            if resp.status_code != 200:
                raise RuntimeError(
                    f"第三方 API 重试创建编辑任务失败 (HTTP {resp.status_code})，诊断 ID: {diagnostic_id}。"
                ) from exc
            resp_data = resp.json()
            time.sleep(5)
    else:
        raise RuntimeError(f"第三方 API 图片编辑任务多次重试后仍失败，诊断 ID: {diagnostic_id}。")

    result_images = resp_data.get("data") or []

    if progress_callback:
        progress_callback("api_direct_edit_processing")

    outputs: list[ImageOutput] = []
    for i, img in enumerate(result_images):
        b64 = str(img.get("b64_json") or "").strip()
        if not b64 and img.get("url"):
            b64 = _fetch_image_as_b64(str(img["url"]))

        if not b64:
            continue

        saved_url = save_image_bytes(base64.b64decode(b64), base_url)
        outputs.append(ImageOutput(
            kind="result",
            model=model,
            index=i,
            total=len(result_images),
            data=[{
                "b64_json": b64,
                "url": saved_url,
                "revised_prompt": str(img.get("revised_prompt") or prompt),
            }],
        ))

    if not outputs:
        raise RuntimeError("API direct: no edited images returned")

    return outputs


def generate_images(body: dict[str, Any]) -> list[ImageOutput]:
    """Entry point for direct image generation. Returns ImageOutput list."""
    return _generate_images_direct(
        prompt=str(body.get("prompt") or ""),
        model=str(body.get("model") or "gpt-image-2.5-flare"),
        n=int(body.get("n") or 1),
        size=body.get("size"),
        quality=str(body.get("quality") or "auto"),
        response_format=str(body.get("response_format") or "b64_json"),
        base_url=body.get("base_url") or None,
        progress_callback=body.get("progress_callback"),
    )


def edit_images(body: dict[str, Any]) -> list[ImageOutput]:
    """Entry point for direct image editing. Returns ImageOutput list."""
    return _edit_images_direct(
        prompt=str(body.get("prompt") or ""),
        images_b64=body.get("images") or [],
        model=str(body.get("model") or "gpt-image-2.5-flare"),
        n=int(body.get("n") or 1),
        size=body.get("size"),
        quality=str(body.get("quality") or "auto"),
        response_format=str(body.get("response_format") or "b64_json"),
        base_url=body.get("base_url") or None,
        progress_callback=body.get("progress_callback"),
    )
