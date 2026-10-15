"""Read MaiAI account funds; never substitute an unlimited API-token allowance."""
from decimal import Decimal, InvalidOperation

import httpx

from services.config import BASE_DIR, config

TOKEN_FILE = BASE_DIR / '.maiai-balance-token'
MAIAI_BASE = 'https://maiai.ai'


def _number(value, *, positive=False):
    if isinstance(value, bool) or value is None:
        raise ValueError('Missing number')
    number = Decimal(str(value))
    if not number.is_finite() or (positive and number <= 0):
        raise ValueError('Invalid number')
    return number


def estimate_quota(account, status, pricing, groups, model):
    unit = _number(status.get('quota_per_unit'), positive=True)
    balance = max(Decimal(0), _number(account.get('quota'))) / unit
    item = next(x for x in pricing['data'] if x.get('model_name') == model)
    if item.get('quota_type') != 1:
        raise ValueError('Model is not priced per request')
    eligible = item.get('enable_groups', [])
    # This image model has one billing group. Do not guess for ambiguous routes.
    if len(eligible) != 1:
        raise ValueError('Ambiguous billing group')
    ratio = _number(groups[eligible[0]]['ratio'], positive=True)
    price = _number(item.get('model_price'), positive=True) * ratio
    return {'status': 'ok', 'source': 'balance', 'model': model,
            'balance_usd': str(balance), 'price_usd': str(price),
            'estimated_images': int(balance // price)}


def get_image_quota(model):
    if not config.openai_api_key:
        return {'status': 'ok', 'source': 'pool'}
    unavailable = {'status': 'unavailable', 'source': 'balance', 'model': model}
    # The separate system credential is scoped here to its intended destination.
    if config.openai_api_base.rstrip('/') != MAIAI_BASE or model not in config.openai_api_models:
        return unavailable
    try:
        token = TOKEN_FILE.read_text(encoding='utf-8').strip()
        if not token:
            return unavailable
        with httpx.Client(proxy=config.openai_api_proxy or None, trust_env=False,
                          timeout=10, follow_redirects=False) as client:
            def read(path, authenticated=False):
                response = client.get(MAIAI_BASE + path, headers={
                    'Authorization': 'Bearer ' + token,
                } if authenticated else {})
                response.raise_for_status()
                data = response.json()
                if data.get('success') is not True:
                    raise ValueError('Upstream query failed')
                return data
            account = read('/api/user/self', True)['data']
            status = read('/api/status')['data']
            pricing = read('/api/pricing')
            groups = read('/api/user/self/groups', True)['data']
        return estimate_quota(account, status, pricing, groups, model)
    except (OSError, httpx.HTTPError, ValueError, InvalidOperation, KeyError, TypeError, StopIteration):
        # Do not return/log upstream bodies, tokens, or stale estimates as current.
        return unavailable
