from __future__ import annotations

import unittest
from unittest.mock import patch

import httpx

from services.protocol import openai_api_direct


class DirectImagePollingTests(unittest.TestCase):
    def _response(self, status_code: int, payload: object) -> httpx.Response:
        request = httpx.Request("GET", "https://example.test/v1/images/poll/abc")
        return httpx.Response(status_code, json=payload, request=request)

    @patch.object(openai_api_direct.time, "sleep", return_value=None)
    @patch.object(openai_api_direct.httpx, "get")
    def test_completed_returns_data(self, get, _sleep):
        # First poll: still processing. Second poll: completed with URL.
        get.side_effect = [
            self._response(200, {"status": "processing"}),
            self._response(200, {"status": "completed", "data": []}),
        ]
        with patch.object(openai_api_direct, "_fetch_image_as_b64", return_value="bm90LWFuLWltYWdl"):
            result = openai_api_direct._poll_direct_image_result(
                {"status": "processing", "poll_url": "https://example.test/v1/images/poll/abc"},
                api_key="sk-test",
                diagnostic_id="diag-1",
            )
        self.assertEqual(result.get("status"), "completed")
        self.assertEqual(get.call_count, 2)

    @patch.object(openai_api_direct.time, "sleep", return_value=None)
    @patch.object(openai_api_direct.httpx, "get")
    def test_failed_status_raises(self, get, _sleep):
        get.return_value = self._response(
            200, {"status": "failed", "error": {"message": "provider rejected"}}
        )
        with self.assertRaisesRegex(RuntimeError, "图片任务失败，诊断 ID: diag-1"):
            openai_api_direct._poll_direct_image_result(
                {"status": "processing", "poll_url": "https://example.test/v1/images/poll/abc"},
                api_key="sk-test",
                diagnostic_id="diag-1",
            )

    @patch.object(openai_api_direct.time, "sleep", return_value=None)
    @patch.object(openai_api_direct.httpx, "get")
    def test_invalid_json_raises(self, get, _sleep):
        request = httpx.Request("GET", "https://example.test/v1/images/poll/abc")
        get.return_value = httpx.Response(200, content=b"<html>not json</html>", request=request)
        with self.assertRaisesRegex(RuntimeError, "无法解析的 JSON，诊断 ID: diag-1"):
            openai_api_direct._poll_direct_image_result(
                {"status": "processing", "poll_url": "https://example.test/v1/images/poll/abc"},
                api_key="sk-test",
                diagnostic_id="diag-1",
            )

    @patch.object(openai_api_direct.time, "sleep", return_value=None)
    @patch.object(openai_api_direct.httpx, "get")
    def test_non_200_raises(self, get, _sleep):
        get.return_value = self._response(500, {"error": "boom"})
        with self.assertRaisesRegex(RuntimeError, "轮询失败 .HTTP 500"):
            openai_api_direct._poll_direct_image_result(
                {"status": "processing", "poll_url": "https://example.test/v1/images/poll/abc"},
                api_key="sk-test",
                diagnostic_id="diag-1",
            )

    def test_initial_without_poll_url_returns_unchanged(self):
        result = openai_api_direct._poll_direct_image_result(
            {"status": "processing"},
            api_key="sk-test",
            diagnostic_id="diag-1",
        )
        self.assertEqual(result, {"status": "processing"})


if __name__ == "__main__":
    unittest.main()