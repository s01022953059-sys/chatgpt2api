import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from services import image_quota_service as quota
from api.system import create_router


class ImageQuotaTests(unittest.TestCase):
    def estimate(self, funds=4531938, price=0.15, ratio=1, unit=500000, model='flare'):
        return quota.estimate_quota({'quota': funds}, {'quota_per_unit': unit},
            {'data': [{'model_name': 'flare', 'quota_type': 1, 'model_price': price,
                       'enable_groups': ['image-1']}]}, {'image-1': {'ratio': ratio}}, model)

    def test_floor_and_decimal_boundary(self):
        self.assertEqual(self.estimate()['estimated_images'], 60)
        self.assertEqual(self.estimate(75000)['estimated_images'], 1)
        self.assertEqual(self.estimate(74999)['estimated_images'], 0)
        self.assertEqual(self.estimate(4755000)['estimated_images'], 63)

    def test_zero_debt_and_group_multiplier(self):
        self.assertEqual(self.estimate(0)['estimated_images'], 0)
        self.assertEqual(self.estimate(-10)['estimated_images'], 0)
        self.assertEqual(self.estimate(ratio=2)['estimated_images'], 30)

    def test_invalid_numbers_do_not_become_zero(self):
        for args in [{'funds': None}, {'price': 0}, {'unit': 0}, {'ratio': 'NaN'}, {'funds': 'Infinity'}]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.estimate(**args)

    @patch.object(quota, 'config', SimpleNamespace(openai_api_key='key', openai_api_base='https://maiai.ai', openai_api_models=['flare'], openai_api_proxy='http://proxy'))
    @patch.object(quota, 'TOKEN_FILE')
    @patch.object(quota.httpx, 'Client')
    def test_authenticated_reads_and_no_secret_response(self, client, token):
        token.read_text.return_value = 'secret-for-test'
        payloads = [{'data': {'quota': 4531938}}, {'data': {'quota_per_unit': 500000}},
            {'data': [{'model_name': 'flare', 'quota_type': 1, 'model_price': .15, 'enable_groups': ['image-1']}]},
            {'data': {'image-1': {'ratio': 1}}}]
        session = client.return_value.__enter__.return_value
        session.get.side_effect = [httpx.Response(200, json={'success': True, **p}, request=httpx.Request('GET', 'https://maiai.ai')) for p in payloads]
        result = quota.get_image_quota('flare')
        self.assertEqual(result['estimated_images'], 60)
        self.assertNotIn('secret-for-test', str(result))
        self.assertEqual(client.call_args.kwargs['proxy'], 'http://proxy')
        self.assertFalse(client.call_args.kwargs['follow_redirects'])
        calls=session.get.call_args_list
        self.assertIn('Authorization', calls[0].kwargs['headers'])
        self.assertEqual(calls[1].kwargs['headers'], {})
        self.assertEqual(calls[2].kwargs['headers'], {})

    @patch.object(quota, 'config', SimpleNamespace(openai_api_key='key', openai_api_base='https://maiai.ai', openai_api_models=['flare'], openai_api_proxy=''))
    @patch.object(quota, 'TOKEN_FILE')
    @patch.object(quota.httpx, 'Client')
    def test_query_failure_is_unknown_and_does_not_leak(self, client, token):
        token.read_text.return_value = 'secret-for-test'
        client.return_value.__enter__.return_value.get.side_effect = httpx.ConnectError('secret-for-test')
        self.assertEqual(quota.get_image_quota('flare')['status'], 'unavailable')
        token.read_text.side_effect = FileNotFoundError()
        self.assertEqual(quota.get_image_quota('flare')['status'], 'unavailable')

    @patch.object(quota, 'TOKEN_FILE')
    def test_credentials_not_sent_to_changed_provider_or_wrong_model(self, token):
        with patch.object(quota,'config',SimpleNamespace(openai_api_key='key',openai_api_base='https://other.test',openai_api_models=['flare'])):
            self.assertEqual(quota.get_image_quota('flare')['status'], 'unavailable')
        token.read_text.assert_not_called()

    @patch.object(quota, 'config', SimpleNamespace(openai_api_key=''))
    def test_pool_mode(self):
        self.assertEqual(quota.get_image_quota('flare')['source'], 'pool')

    def test_endpoint_requires_admin(self):
        app=FastAPI();app.include_router(create_router('test'))
        with patch('api.system.require_admin',side_effect=HTTPException(403)), patch('api.system.get_image_quota') as get:
            response=TestClient(app).get('/api/image-quota')
            self.assertEqual(response.status_code,403);get.assert_not_called()


if __name__ == '__main__': unittest.main()
