"""Tests for the third-party OpenAI-compatible API outbound proxy support.

Covers:
- config layer: openai_api_proxy resolution (empty default / config.json /
  env var precedence)
- injection: proxy passed to every outbound httpx call
  (generations / edits / poll / image download)
"""
from __future__ import annotations

import base64
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from services.protocol import openai_api_direct


def _response(status_code: int, payload: object, url: str = "https://example.test") -> httpx.Response:
    request = httpx.Request("POST", url)
    return httpx.Response(status_code, json=payload, request=request)


def _build_config_store(data: dict) -> object:
    """Create a ConfigStore pointed at a temp file with the given payload."""
    import json

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        from services.config import ConfigStore

        return ConfigStore(path)


class OpenaiApiProxyConfigTests(unittest.TestCase):
    """Verify the config-layer proxy resolver."""

    def setUp(self) -> None:
        self._env_backup = os.environ.pop("CHATGPT2API_OPENAI_API_PROXY", None)

    def tearDown(self) -> None:
        os.environ.pop("CHATGPT2API_OPENAI_API_PROXY", None)
        if self._env_backup is not None:
            os.environ["CHATGPT2API_OPENAI_API_PROXY"] = self._env_backup

    def test_unconfigured_returns_empty(self) -> None:
        store = _build_config_store({"auth-key": "test-auth"})
        self.assertEqual(store.openai_api_proxy, "")

    def test_configured_value_in_config_json(self) -> None:
        store = _build_config_store(
            {
                "auth-key": "test-auth",
                "openai_api_proxy": "http://127.0.0.1:40080",
            }
        )
        self.assertEqual(store.openai_api_proxy, "http://127.0.0.1:40080")

    def test_configured_value_via_env_var(self) -> None:
        store = _build_config_store({"auth-key": "test-auth"})
        os.environ["CHATGPT2API_OPENAI_API_PROXY"] = "http://env-proxy:8118"
        self.assertEqual(store.openai_api_proxy, "http://env-proxy:8118")

    def test_env_var_takes_precedence_over_config_json(self) -> None:
        store = _build_config_store(
            {
                "auth-key": "test-auth",
                "openai_api_proxy": "http://file-proxy:8118",
            }
        )
        os.environ["CHATGPT2API_OPENAI_API_PROXY"] = "http://env-proxy:8118"
        self.assertEqual(store.openai_api_proxy, "http://env-proxy:8118")

    def test_whitespace_only_value_returns_empty(self) -> None:
        store = _build_config_store(
            {
                "auth-key": "test-auth",
                "openai_api_proxy": "   ",
            }
        )
        self.assertEqual(store.openai_api_proxy, "")

    def test_get_exposes_field_in_payload(self) -> None:
        store = _build_config_store(
            {
                "auth-key": "test-auth",
                "openai_api_proxy": "http://127.0.0.1:40080",
            }
        )
        payload = store.get()
        self.assertEqual(payload.get("openai_api_proxy"), "http://127.0.0.1:40080")


class OpenaiApiProxyInjectionTests(unittest.TestCase):
    """Proxy must be forwarded to every outbound httpx call."""

    @patch.object(openai_api_direct.time, "sleep", return_value=None)
    @patch.object(openai_api_direct, "_get_proxy", return_value="http://proxy.test:8118")
    @patch.object(openai_api_direct, "_get_user_agent", return_value="custom-ua")
    @patch.object(openai_api_direct, "_get_api_key", return_value="sk-test")
    @patch.object(openai_api_direct, "_get_api_base", return_value="https://example.test")
    @patch.object(openai_api_direct.httpx, "post")
    @patch.object(openai_api_direct, "save_image_bytes", return_value="/images/test.png")
    def test_generations_post_carries_proxy(
        self, _save, post, _base, _key, _ua, _proxy, _sleep,
    ) -> None:
        encoded = base64.b64encode(b"fake-png").decode()
        post.return_value = _response(200, {"data": [{"b64_json": encoded}]})

        openai_api_direct.generate_images({"prompt": "cat", "model": "gpt-image-2"})

        self.assertEqual(post.call_args.kwargs.get("proxy"), "http://proxy.test:8118")

    @patch.object(openai_api_direct, "_get_proxy", return_value="http://proxy.test:8118")
    @patch.object(openai_api_direct, "_get_user_agent", return_value="custom-ua")
    @patch.object(openai_api_direct, "_get_api_key", return_value="sk-test")
    @patch.object(openai_api_direct, "_get_api_base", return_value="https://example.test")
    @patch.object(openai_api_direct.httpx, "post")
    @patch.object(openai_api_direct, "save_image_bytes", return_value="/images/test.png")
    def test_edits_post_carries_proxy(
        self, _save, post, _base, _key, _ua, _proxy,
    ) -> None:
        encoded = base64.b64encode(b"fake-png").decode()
        post.return_value = _response(200, {"data": [{"b64_json": encoded}]})

        openai_api_direct.edit_images(
            {"prompt": "cat", "model": "gpt-image-2", "images": [encoded]}
        )

        self.assertEqual(post.call_args.kwargs.get("proxy"), "http://proxy.test:8118")

    @patch.object(openai_api_direct.time, "sleep", return_value=None)
    @patch.object(openai_api_direct, "_get_proxy", return_value="http://proxy.test:8118")
    @patch.object(openai_api_direct, "_get_user_agent", return_value="custom-ua")
    @patch.object(openai_api_direct.httpx, "get")
    def test_poll_get_carries_proxy(self, get, _ua, _proxy, _sleep) -> None:
        get.return_value = _response(
            200, {"status": "completed", "data": [{"b64_json": "aGk="}]},
            url="https://upstream.test/poll/1",
        )
        result = openai_api_direct._poll_direct_image_result(
            {"status": "queued", "poll_url": "https://upstream.test/poll/1"},
            api_key="sk-test",
            diagnostic_id="diag-proxy",
        )
        self.assertEqual(result["data"][0]["b64_json"], "aGk=")
        self.assertEqual(get.call_args.kwargs.get("proxy"), "http://proxy.test:8118")

    @patch.object(openai_api_direct, "_get_proxy", return_value="http://proxy.test:8118")
    @patch.object(openai_api_direct.httpx, "get")
    def test_image_download_carries_proxy(self, get, _proxy) -> None:
        get.return_value = type("Resp", (), {
            "content": b"img",
            "raise_for_status": lambda self: None,
        })()
        openai_api_direct._fetch_image_as_b64("https://cdn.test/img.png")
        self.assertEqual(get.call_args.kwargs.get("proxy"), "http://proxy.test:8118")


class OpenaiApiNoProxyTests(unittest.TestCase):
    """Without proxy config, httpx must receive proxy=None (direct)."""

    @patch.object(openai_api_direct.time, "sleep", return_value=None)
    @patch.object(openai_api_direct, "_get_proxy", return_value="")
    @patch.object(openai_api_direct, "_get_user_agent", return_value="custom-ua")
    @patch.object(openai_api_direct, "_get_api_key", return_value="sk-test")
    @patch.object(openai_api_direct, "_get_api_base", return_value="https://example.test")
    @patch.object(openai_api_direct.httpx, "post")
    @patch.object(openai_api_direct, "save_image_bytes", return_value="/images/test.png")
    def test_generations_post_proxy_is_none(
        self, _save, post, _base, _key, _ua, _proxy, _sleep,
    ) -> None:
        encoded = base64.b64encode(b"fake-png").decode()
        post.return_value = _response(200, {"data": [{"b64_json": encoded}]})

        openai_api_direct.generate_images({"prompt": "cat", "model": "gpt-image-2"})

        self.assertIsNone(post.call_args.kwargs.get("proxy"))


if __name__ == "__main__":
    unittest.main()
