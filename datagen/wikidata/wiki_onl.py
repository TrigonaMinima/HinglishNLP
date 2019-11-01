import requests


def run_api_endpoint(payload):
    request_url = 'https://www.wikidata.org/w/api.php'
    r = requests.get(request_url, payload)
    data = r.json()
    return data
