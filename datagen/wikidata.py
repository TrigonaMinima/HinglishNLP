from wikidata.wiki_onl import run_api_endpoint


def search_entities(query, language='en'):
    payload = {
        'action': 'wbsearchentities',
        'format': 'json',
        'limit': 1,
        'language': language,
        'search': query,
    }

    data = run_api_endpoint(payload)

    # entities = data.get('search')
    # label = entities[0].get('label') if entities else None

    # return label
    return data


if __name__ == '__main__':
    search_list = ['UK', 'US', 'America', 'Einstein', 'Dumbledor']
    for word in search_list:
        print(search_entities(word))
        # print(f'{word} is {search_entities(word)}')
