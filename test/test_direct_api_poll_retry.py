"""Tests for automatic retry when third-party API task returns 'failed'.

The third-party API sometimes marks a task as 'failed' due to its own
internal timeout (not our fault). The code should automatically re-create
the task and retry, rather than giving up immediately.
"""

from __future__ import annotations

import base64
import unittest
from unittest.mock import patch

import httpx

from services.protocol import openai_api_direct


def _response(status_code: int, payload: object, url: str = "https://example.test") -> httpx.Response:
    request = httpx.Request("POST", url)
    return httpx.Response(status_code, json=payload, request=request)


class PollRetryTests(unittest.TestCase):
    """When polling returns 'failed', the caller should re-create and retry."""

    @patch.object(openai_api_direct.time, "sleep", return_value=None)
    @patch.object(openai_api_direct, "_get_user_agent", return_value="custom-ua")
    @patch.object(openai_api_direct, "_get_api_key", return_value="sk-test")
    @patch.object(openai_api_direct, "_get_api_base", return_value="https://example.test")
    @patch.object(openai_api_direct.httpx, "post")
    @patch.object(openai_api_direct.httpx, "get")
    @patch.object(openai_api_direct, "save_image_bytes", return_value="/images/test.png")
    def test_generations_poll_failed_triggers_retry_and_succeeds(
        self, _save, get, post, _base, _key, _ua, _sleep,
    ):
        """First task fails during polling → re-create → second task succeeds."""
        encoded = base64.b64encode(b"fake-png").decode()

        # post.call 1: create first task (returns poll_url)
        # post.call 2: re-create after failure (returns poll_url)
        post.side_effect = [
            _response(200, {"status": "processing", "poll_url": "https://example.test/poll/1"}),
            _response(200, {"status": "processing", "poll_url": "https://example.test/poll/2"}),
        ]

        # get.call 1: first poll → failed
        # get.call 2: second poll → completed
        get.side_effect = [
            _response(200, {"status": "failed", "error": "internal timeout"}, url="https://example.test/poll/1"),
            _response(200, {"status": "completed", "data": [{"b64_json": encoded}]}, url="https://example.test/poll/2"),
        ]

        outputs = openai_api_direct.generate_images({"prompt": "cat", "model": "gpt-image-2"})

        self.assertEqual(len(outputs), 1)
        # post should have been called twice (create + re-create)
        self.assertEqual(post.call_count, 2)

    @patch.object(openai_api_direct.time, "sleep", return_value=None)
    @patch.object(openai_api_direct, "_get_user_agent", return_value="custom-ua")
    @patch.object(openai_api_direct, "_get_api_key", return_value="sk-test")
    @patch.object(openai_api_direct, "_get_api_base", return_value="https://example.test")
    @patch.object(openai_api_direct.httpx, "post")
    @patch.object(openai_api_direct.httpx, "get")
    @patch.object(openai_api_direct, "save_image_bytes", return_value="/images/test.png")
    def test_generations_poll_failed_both_attempts_raises(
        self, _save, get, post, _base, _key, _ua, _sleep,
    ):
        """Both attempts fail → should raise RuntimeError."""
        post.side_effect = [
            _response(200, {"status": "processing", "poll_url": "https://example.test/poll/1"}),
            _response(200, {"status": "processing", "poll_url": "https://example.test/poll/2"}),
        ]
        get.side_effect = [
            _response(200, {"status": "failed", "error": "timeout 1"}, url="https://example.test/poll/1"),
            _response(200, {"status": "failed", "error": "timeout 2"}, url="https://example.test/poll/2"),
        ]

        with self.assertRaisesRegex(RuntimeError, "图片任务失败"):
            openai_api_direct.generate_images({"prompt": "cat", "model": "gpt-image-2"})

        self.assertEqual(post.call_count, 2)

    @patch.object(openai_api_direct, "_get_user_agent", return_value="custom-ua")
    @patch.object(openai_api_direct, "_get_api_key", return_value="sk-test")
    @patch.object(openai_api_direct, "_get_api_base", return_value="https://example.test")
    @patch.object(openai_api_direct.httpx, "post")
    def test_edit_model_fallback_to_gpt_image_2(self, post, _base, _key, _ua):
        """Edits should always use gpt-image-2, not gpt-image-2-pro."""
        encoded = base64.b64encode(b"fake-png").decode()
        post.return_value = _response(200, {"data": [{"b64_json": encoded}]})

        with patch.object(openai_api_direct, "save_image_bytes", return_value="/images/test.png"):
            openai_api_direct.edit_images({
                "prompt": "cat",
                "model": "gpt-image-2-pro",  # should be overridden to gpt-image-2
                "images": [encoded],
            })

        sent_data = post.call_args.kwargs["data"]
        self.assertEqual(sent_data["model"], "gpt-image-2")

    @patch.object(openai_api_direct, "_get_api_key", return_value="sk-test")
    @patch.object(openai_api_direct, "_get_api_base", return_value="https://example.test")
    @patch.object(openai_api_direct.httpx, "post")
    def test_edit_preserves_flare_model(self, post, _base, _key):
        encoded = base64.b64encode(b"fake-png").decode()
        post.return_value = _response(200, {"data": [{"b64_json": encoded}]})
        with patch.object(openai_api_direct, "save_image_bytes", return_value="/images/test.png"):
            openai_api_direct.edit_images({
                "prompt": "cat", "model": "gpt-image-2.5-flare", "images": [encoded],
            })
        self.assertEqual(post.call_args.args[0], "https://example.test/v1/images/generations")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "gpt-image-2.5-flare")
        self.assertEqual(post.call_args.kwargs["json"]["image"], [f"data:image/png;base64,{encoded}"])

    @patch.object(openai_api_direct, "_get_user_agent", return_value="custom-ua")
    @patch.object(openai_api_direct, "_get_api_key", return_value="sk-test")
    @patch.object(openai_api_direct, "_get_api_base", return_value="https://example.test")
    @patch.object(openai_api_direct.httpx, "post")
    @patch.object(openai_api_direct, "save_image_bytes", return_value="/images/test.png")
    def test_edit_404_falls_back_to_generations_with_image(
        self, _save, post, _base, _key, _ua,
    ):
        """edits 端点 404（unsupported path）时应降级到 generations + image 字段。"""
        encoded = base64.b64encode(b"fake-png").decode()
        post.side_effect = [
            _response(404, {"error": {"message": "unsupported path", "type": "invalid_request_error"}}),
            _response(200, {"data": [{"b64_json": encoded}]}),
        ]

        outputs = openai_api_direct.edit_images({
            "prompt": "cat",
            "model": "gpt-image-2-pro",  # 降级后应恢复原始模型
            "images": [encoded],
        })

        self.assertEqual(len(outputs), 1)
        self.assertEqual(post.call_count, 2)
        # 第一次调用：edits 端点
        first_call = post.call_args_list[0]
        self.assertIn("/v1/images/edits", first_call.args[0])
        # 第二次调用：generations 端点 + image 字段 + 原始模型
        second_call = post.call_args_list[1]
        self.assertIn("/v1/images/generations", second_call.args[0])
        sent_payload = second_call.kwargs["json"]
        self.assertEqual(sent_payload["model"], "gpt-image-2-pro")
        self.assertEqual(
            sent_payload["image"],
            [f"data:image/png;base64,{encoded}"],
        )

    @patch.object(openai_api_direct, "_get_user_agent", return_value="custom-ua")
    @patch.object(openai_api_direct, "_get_api_key", return_value="sk-test")
    @patch.object(openai_api_direct, "_get_api_base", return_value="https://example.test")
    @patch.object(openai_api_direct.httpx, "post")
    def test_edit_403_does_not_fall_back(self, post, _base, _key, _ua):
        """403（余额/权限问题）不应触发降级，直接报错。"""
        encoded = base64.b64encode(b"fake-png").decode()
        post.return_value = _response(403, {"error": {"message": "insufficient balance"}})

        with self.assertRaisesRegex(RuntimeError, "403"):
            openai_api_direct.edit_images({
                "prompt": "cat",
                "model": "gpt-image-2",
                "images": [encoded],
            })
        self.assertEqual(post.call_count, 1)


class FetchImageAsB64Tests(unittest.TestCase):
    """_fetch_image_as_b64 must handle data URIs without calling httpx."""

    def test_data_uri_png(self):
        raw = b"\x89PNG\r\n\x1a\n"  # minimal PNG header
        encoded = base64.b64encode(raw).decode()
        data_uri = f"data:image/png;base64,{encoded}"
        result = openai_api_direct._fetch_image_as_b64(data_uri)
        self.assertEqual(result, encoded)

    def test_data_uri_with_special_chars_in_metadata(self):
        raw = b"fake-image-data"
        encoded = base64.b64encode(raw).decode()
        # data URI with extra metadata before the comma
        data_uri = f"data:image/png;charset=utf-8;base64,{encoded}"
        result = openai_api_direct._fetch_image_as_b64(data_uri)
        self.assertEqual(result, encoded)

    def test_raw_base64_returned_as_is(self):
        raw = b"some-image-bytes"
        encoded = base64.b64encode(raw).decode()
        result = openai_api_direct._fetch_image_as_b64(encoded)
        self.assertEqual(result, encoded)

    @patch.object(openai_api_direct.httpx, "get")
    def test_http_url_still_downloads(self, get):
        raw = b"downloaded-bytes"
        get.return_value = type("Resp", (), {
            "content": raw,
            "raise_for_status": lambda self: None,
        })()
        result = openai_api_direct._fetch_image_as_b64("https://example.com/img.png")
        self.assertEqual(result, base64.b64encode(raw).decode())
        get.assert_called_once_with("https://example.com/img.png", timeout=120, follow_redirects=True, proxy=None)


if __name__ == "__main__":
    unittest.main()
