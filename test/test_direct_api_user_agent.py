"""Unit tests for third-party OpenAI-compatible API User-Agent injection.

The Cloudflare Error 1010 ("Access denied based on your browser's signature")
incident showed that the default `python-httpx/<ver>` User-Agent is blocked
by WAF. These tests verify that:

- `ConfigStore.openai_api_user_agent` returns a built-in Chrome UA fallback
  when nothing is configured.
- A user-supplied value (env var or config.json) overrides the fallback.
- All outbound httpx calls in `openai_api_direct` carry the resolved UA in
  the `User-Agent` request header.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from services.config import ConfigStore
from services.protocol import openai_api_direct


def _build_config_store(data: dict | None = None) -> ConfigStore:
    """Create a ConfigStore pointed at a temp file with the given payload."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"
        if data is not None:
            path.write_text(json.dumps(data), encoding="utf-8")
        return ConfigStore(path)


# ---------------------------------------------------------------------------
# ConfigStore: openai_api_user_agent property
# ---------------------------------------------------------------------------


class OpenaiApiUserAgentConfigTests(unittest.TestCase):
    """Verify the config-layer User-Agent resolver."""

    def setUp(self) -> None:
        self._env_backup = os.environ.pop("CHATGPT2API_OPENAI_API_USER_AGENT", None)

    def tearDown(self) -> None:
        os.environ.pop("CHATGPT2API_OPENAI_API_USER_AGENT", None)
        if self._env_backup is not None:
            os.environ["CHATGPT2API_OPENAI_API_USER_AGENT"] = self._env_backup

    def test_unconfigured_falls_back_to_builtin_chrome_ua(self) -> None:
        store = _build_config_store({"auth-key": "test-auth"})
        ua = store.openai_api_user_agent
        self.assertTrue(ua.startswith("Mozilla/5.0"))
        self.assertIn("AppleWebKit/537.36", ua)
        self.assertIn("Safari/537.36", ua)
        # Must NOT be the default httpx UA (that's what gets blocked).
        self.assertFalse(ua.startswith("python-httpx"))

    def test_configured_value_in_config_json_overrides_fallback(self) -> None:
        store = _build_config_store(
            {
                "auth-key": "test-auth",
                "openai_api_user_agent": "custom-ua/1.0",
            }
        )
        self.assertEqual(store.openai_api_user_agent, "custom-ua/1.0")

    def test_configured_value_via_env_var_overrides_fallback(self) -> None:
        os.environ["CHATGPT2API_OPENAI_API_USER_AGENT"] = "env-ua/2.0"
        store = _build_config_store({"auth-key": "test-auth"})
        self.assertEqual(store.openai_api_user_agent, "env-ua/2.0")

    def test_env_var_takes_precedence_over_config_json(self) -> None:
        os.environ["CHATGPT2API_OPENAI_API_USER_AGENT"] = "env-wins/3.0"
        store = _build_config_store(
            {
                "auth-key": "test-auth",
                "openai_api_user_agent": "json-loses/4.0",
            }
        )
        self.assertEqual(store.openai_api_user_agent, "env-wins/3.0")

    def test_whitespace_only_value_falls_back_to_builtin(self) -> None:
        store = _build_config_store(
            {
                "auth-key": "test-auth",
                "openai_api_user_agent": "   ",
            }
        )
        ua = store.openai_api_user_agent
        self.assertTrue(ua.startswith("Mozilla/5.0"))

    def test_get_exposes_field_in_payload(self) -> None:
        store = _build_config_store({"auth-key": "test-auth"})
        self.assertIn("openai_api_user_agent", store.get())
        self.assertTrue(store.get()["openai_api_user_agent"].startswith("Mozilla/5.0"))


# ---------------------------------------------------------------------------
# openai_api_direct: User-Agent injected on every outbound call
# ---------------------------------------------------------------------------


def _response(status_code: int, payload: object, url: str = "https://example.test") -> httpx.Response:
    request = httpx.Request("POST", url)
    return httpx.Response(status_code, json=payload, request=request)


class OpenaiApiDirectUserAgentInjectionTests(unittest.TestCase):
    """Verify that the resolved UA is sent on every outbound httpx call."""

    @patch.object(openai_api_direct, "_get_user_agent", return_value="custom-ua/1.0")
    @patch.object(openai_api_direct, "_get_api_key", return_value="sk-test")
    @patch.object(openai_api_direct, "_get_api_base", return_value="https://example.test")
    @patch.object(openai_api_direct.httpx, "post")
    def test_generations_post_carries_user_agent(self, post, _base, _key, _ua):
        encoded = "ZmFrZS1wbmc="  # base64 of "fake-png"
        post.return_value = _response(200, {"data": [{"b64_json": encoded}]})
        with patch.object(openai_api_direct, "save_image_bytes", return_value="/images/test.png"):
            openai_api_direct.generate_images({"prompt": "cat", "model": "gpt-image-2"})

        # At least one post call must include the custom User-Agent.
        seen_agents = [
            call.kwargs.get("headers", {}).get("User-Agent")
            for call in post.call_args_list
        ]
        self.assertTrue(seen_agents, "httpx.post was never called")
        self.assertIn("custom-ua/1.0", seen_agents)

    @patch.object(openai_api_direct, "_get_user_agent", return_value="custom-ua/1.0")
    @patch.object(openai_api_direct, "_get_api_key", return_value="sk-test")
    @patch.object(openai_api_direct, "_get_api_base", return_value="https://example.test")
    @patch.object(openai_api_direct.httpx, "post")
    def test_edits_post_carries_user_agent(self, post, _base, _key, _ua):
        encoded = "ZmFrZS1wbmc="
        post.return_value = _response(200, {"data": [{"b64_json": encoded}]})
        with patch.object(openai_api_direct, "save_image_bytes", return_value="/images/test.png"):
            openai_api_direct.edit_images(
                {
                    "prompt": "cat",
                    "model": "gpt-image-2",
                    "images": [encoded],
                }
            )

        seen_agents = [
            call.kwargs.get("headers", {}).get("User-Agent")
            for call in post.call_args_list
        ]
        self.assertTrue(seen_agents, "httpx.post was never called for edits")
        self.assertIn("custom-ua/1.0", seen_agents)

    @patch.object(openai_api_direct.time, "sleep", return_value=None)
    @patch.object(openai_api_direct, "_get_user_agent", return_value="custom-ua/1.0")
    @patch.object(openai_api_direct.httpx, "get")
    def test_poll_request_carries_user_agent(self, get, _ua, _sleep):
        # First poll still processing; second poll completed to break the loop.
        request = httpx.Request("GET", "https://example.test/v1/images/poll/abc")
        get.side_effect = [
            httpx.Response(200, json={"status": "processing"}, request=request),
            httpx.Response(200, json={"status": "completed", "data": []}, request=request),
        ]
        with patch.object(openai_api_direct, "_fetch_image_as_b64", return_value="bm90LWFuLWltYWdl"):
            openai_api_direct._poll_direct_image_result(
                {"status": "processing", "poll_url": "https://example.test/v1/images/poll/abc"},
                api_key="sk-test",
                diagnostic_id="diag-1",
            )

        seen_agents = [
            call.kwargs.get("headers", {}).get("User-Agent")
            for call in get.call_args_list
        ]
        self.assertTrue(seen_agents, "httpx.get was never called during polling")
        self.assertIn("custom-ua/1.0", seen_agents)

    @patch.object(openai_api_direct, "_get_user_agent", return_value="Mozilla/5.0 (builtin) Chrome/145")
    @patch.object(openai_api_direct, "_get_api_key", return_value="sk-test")
    @patch.object(openai_api_direct, "_get_api_base", return_value="https://example.test")
    @patch.object(openai_api_direct.httpx, "post")
    def test_builtin_fallback_ua_is_used_when_nothing_configured(self, post, _base, _key, _ua):
        """The fallback UA must not be the default `python-httpx/<ver>`."""
        encoded = "ZmFrZS1wbmc="
        post.return_value = _response(200, {"data": [{"b64_json": encoded}]})
        with patch.object(openai_api_direct, "save_image_bytes", return_value="/images/test.png"):
            openai_api_direct.generate_images({"prompt": "cat", "model": "gpt-image-2"})

        sent_ua = post.call_args.kwargs["headers"]["User-Agent"]
        self.assertNotIn("python-httpx", sent_ua)
        self.assertIn("Mozilla/5.0", sent_ua)

    @patch.object(openai_api_direct, "_get_user_agent", return_value="custom-ua/1.0")
    @patch.object(openai_api_direct, "_get_api_key", return_value="sk-test")
    @patch.object(openai_api_direct, "_get_api_base", return_value="https://example.test")
    @patch.object(openai_api_direct.httpx, "post")
    def test_authorization_header_is_still_present(self, post, _base, _key, _ua):
        """Adding User-Agent must not regress the Authorization header."""
        encoded = "ZmFrZS1wbmc="
        post.return_value = _response(200, {"data": [{"b64_json": encoded}]})
        with patch.object(openai_api_direct, "save_image_bytes", return_value="/images/test.png"):
            openai_api_direct.generate_images({"prompt": "cat", "model": "gpt-image-2"})

        headers = post.call_args.kwargs["headers"]
        self.assertEqual(headers.get("Authorization"), "Bearer sk-test")
        self.assertEqual(headers.get("User-Agent"), "custom-ua/1.0")
        self.assertEqual(headers.get("Content-Type"), "application/json")


if __name__ == "__main__":
    unittest.main()
