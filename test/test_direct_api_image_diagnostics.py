from __future__ import annotations

import base64
import unittest
from unittest.mock import patch

import httpx

from services.protocol import openai_api_direct


class DirectImageDiagnosticsTests(unittest.TestCase):
    def response(self, status_code: int, payload: object) -> httpx.Response:
        request = httpx.Request("POST", "https://example.test/v1/images/generations")
        return httpx.Response(status_code, json=payload, request=request)

    @patch.object(openai_api_direct, "_get_api_key", return_value="sk-test-secret")
    @patch.object(openai_api_direct, "_get_api_base", return_value="https://example.test")
    @patch.object(openai_api_direct.httpx, "post")
    def test_empty_data_includes_diagnostic_id_without_secret(self, post, _base, _key):
        post.return_value = self.response(200, {"data": []})
        with self.assertRaisesRegex(RuntimeError, r"no images returned，诊断 ID: [0-9a-f]{12}") as context:
            openai_api_direct.generate_images({"prompt": "cat", "model": "gpt-image-2"})
        self.assertNotIn("sk-test-secret", str(context.exception))

    @patch.object(openai_api_direct, "_get_api_key", return_value="sk-test-secret")
    @patch.object(openai_api_direct, "_get_api_base", return_value="https://example.test")
    @patch.object(openai_api_direct.httpx, "post")
    def test_missing_data_is_incompatible(self, post, _base, _key):
        post.return_value = self.response(200, {"task_id": "task-1", "status": "processing"})
        with self.assertRaisesRegex(RuntimeError, r"返回格式不兼容.*诊断 ID"):
            openai_api_direct.generate_images({"prompt": "cat", "model": "gpt-image-2"})

    @patch.object(openai_api_direct, "_get_api_key", return_value="sk-test-secret")
    @patch.object(openai_api_direct, "_get_api_base", return_value="https://example.test")
    @patch.object(openai_api_direct.httpx, "post")
    def test_401_is_identified_as_invalid_key(self, post, _base, _key):
        post.return_value = self.response(401, {"error": {"message": "invalid api key"}})
        with self.assertRaisesRegex(RuntimeError, r"API Key 无效.*HTTP 401"):
            openai_api_direct.generate_images({"prompt": "cat", "model": "gpt-image-2"})

    @patch.object(openai_api_direct, "_get_api_key", return_value="sk-test-secret")
    @patch.object(openai_api_direct, "_get_api_base", return_value="https://example.test")
    @patch.object(openai_api_direct.httpx, "post")
    def test_standard_b64_response_succeeds(self, post, _base, _key):
        encoded = base64.b64encode(b"fake-png").decode()
        post.return_value = self.response(200, {"data": [{"b64_json": encoded}]})
        with patch.object(openai_api_direct, "save_image_bytes", return_value="/images/test.png"):
            outputs = openai_api_direct.generate_images({"prompt": "cat", "model": "gpt-image-2"})
        self.assertEqual(len(outputs), 1)


if __name__ == "__main__":
    unittest.main()
