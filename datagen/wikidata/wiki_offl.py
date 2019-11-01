import itertools

from qwikidata.entity import WikidataItem, WikidataProperty
from qwikidata.json_dump import WikidataJsonDump


def get_type_to_entity_class():
    return {"item": WikidataItem, "property": WikidataProperty}


def get_wikidatajsondump_obj(filename):
    return WikidataJsonDump(filename)


def label_pairs(entity):
    """
    Gets document label in English and Hindi and returns the strings as a pair.
    """
    eng = entity.get_label().lower()
    hi = entity.get_label(lang="hi").lower()
    if not eng or not hi:
        return []
    return [(eng, hi)]


def desc_pairs(entity):
    """
    Gets document dedscription in English and Hindi and returns the strings as
    a pair.
    """
    eng = entity.get_description().lower()
    hi = entity.get_description(lang="hi").lower()
    if not eng or not hi:
        return []
    return [(eng, hi)]


def alias_pairs(entity):
    """
    Gets all the aliases of the document in English and Hindi and returns a
    cross product of the 2 lists.
    """
    eng = set([i for i in entity.get_aliases() if i])
    hi = set([i for i in entity.get_aliases(lang="hi") if i])

    common = eng.intersection(hi)
    eng = eng.difference(common)
    hi = hi.difference(common)

    if not eng or not hi:
        return []
    return list(itertools.product(eng, hi))


def get_all_pairs_entity(entity_dict):
    """
    Takes in an entity dict and then returns all the transliteration pairs
    present in the data for the topic.
    """
    type_to_entity_class = get_type_to_entity_class()

    entity_type = entity_dict["type"]
    entity = type_to_entity_class[entity_type](entity_dict)

    pairs = label_pairs(entity) + desc_pairs(entity) + alias_pairs(entity)
    pairs = set(pairs)
    return pairs
