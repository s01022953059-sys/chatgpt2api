import base64
import unittest
from unittest.mock import patch
import httpx
from services.protocol import openai_api_direct as api


class DirectSizeTests(unittest.TestCase):
    def test_legacy_presets(self):
        for old, new in [('1024x1365', '768x1024'), ('1365x1024', '1024x768'),
                         ('1920x1088', '1536x864'), ('1088x1920', '864x1536')]:
            with self.subTest(size=old):
                self.assertEqual(api._resolve_size('gpt-image-2.5-flare', old), new)
                self.assertEqual(api._resolve_size('gpt-image-2', old), old)

    def test_other_sizes_unchanged(self):
        for size in ['1024x1024', '2048x2048', '3840x2160', '768x1024', '1234x987', 'auto']:
            self.assertEqual(api._resolve_size('gpt-image-2.5-flare', size), size)
        self.assertEqual(api._resolve_size('gpt-image-2.5-flare', None), '1024x1024')

    @patch.object(api, '_get_api_key', return_value='test')
    @patch.object(api, '_get_api_base', return_value='https://example.test')
    @patch.object(api, 'save_image_bytes', return_value='/images/test.png')
    @patch.object(api.httpx, 'post')
    def test_generation_and_reference_edit_normalize(self, post, *_):
        encoded = base64.b64encode(b'fake-png').decode()
        post.return_value = httpx.Response(200, json={'data': [{'b64_json': encoded}]})
        for handler in [api.generate_images, api.edit_images]:
            handler({'model': 'gpt-image-2.5-flare', 'prompt': 'test', 'size': '1024x1365', 'images': [encoded]})
            self.assertEqual(post.call_args.kwargs['json']['size'], '768x1024')

    @patch.object(api, '_get_api_key', return_value='test')
    @patch.object(api, '_get_api_base', return_value='https://example.test')
    @patch.object(api.httpx, 'post')
    def test_unsupported_size_has_actionable_error_without_retry(self, post, *_):
        post.return_value = httpx.Response(400, json={'error': {'message': 'unsupported size ratio: 1234:987'}})
        with self.assertRaisesRegex(RuntimeError, '上游不支持图片尺寸 1234x987'):
            api.generate_images({'model': 'gpt-image-2.5-flare', 'prompt': 'test', 'size': '1234x987'})
        self.assertEqual(post.call_count, 1)
