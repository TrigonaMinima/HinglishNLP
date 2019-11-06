# This script is an ad-hoc work only meant to quickly transliterate hindi words.
# It doesn't work for a LOT of cases and it has a memory leak somewhere
#
# add hunterian system: https://en.wikipedia.org/wiki/Hunterian_transliteration
# add wx notation system

dont_append_a = [
    'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ',
    'ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ै', 'ो', 'ौ',
    'ं', 'ः', '्', 'ँ', '़', ' '
]

ignore_n = ['ँ', 'ं']

conversions = {
    ' ': [' '],
    '्': [''],
    'क': ['k', "ck", "q", "c"],
    'ख': ['kh'],
    'ग': ['g'],
    'घ': ['gh'],
    'ङ': ['ng'],
    'च': ['ch'],
    'छ': ['chh', 'ch'],
    'ज': ['j', 'z', "g", "za"],
    'झ': ['jh', 'zh'],
    'ञ': ['nj'],
    'ट': ['t'],
    'ठ': ['th', 't'],
    'ड': ['d'],
    'ढ': ['dh'],
    'ण': ['n'],
    'त': ['t'],
    'थ': ['th'],
    'द': ['d'],
    'ध': ['dh', "d"],
    'न': ['n'],
    'प': ['p'],
    'फ': ['f', 'ph'],
    'ब': ['b'],
    'भ': ['bh'],
    'म': ['m'],
    'य': ['y'],
    'र': ['r'],
    'ल': ['l'],
    'व': ['v', 'w'],
    'श': ['s', 'sh', 'shh'],
    'ष': ['s', 'sh', 'shh'],
    'स': ['s', "c"],
    'ह': ['h'],
    'अ': ['a', ""],
    'आ': ['a', 'aa', ""],
    'इ': ['e', 'i'],
    'ई': ['ee', 'i', "e"],
    'उ': ['u', 'oo'],
    'ऊ': ['u', 'oo'],
    'ए': ['e', "a", "ae", ""],
    'ऐ': ['ai', 'ae', ""],
    'ओ': ['o'],
    'औ': ['ou', 'o'],
    'ा': ['a', 'aa'],
    'ि': ['e', 'i', "ei", "ie", "", "ea"],
    'ी': ['i', 'ee', "ei", "ie", "", "e", "ea"],
    'ु': ['oo', 'u'],
    'ू': ['oo', 'u'],
    'े': ['e', "ai", "a"],
    'ै': ['ai', 'ae', "a"],
    'ो': ['o'],
    'ौ': ['ou', 'o'],
    'ं': ['n', "m", "am", "an"],
    'ँ': ['n'],
    'ः': ['h'],
    "ॉ": ["o", "a"],
    # "ॅ": ["o", "e"],
    "ृ": ["ri"],
    "़": ["a"],
}


def transliterations(hindi_string):
    """
    Returns a set of all the possible transliterations for the given
    hindi_string
    """
    if not hindi_string:
        return []

    translits = ['']
    add_a = False
    length = len(hindi_string)
    for count, letter in enumerate(hindi_string):
        # print(count, letter, translits)

        if hindi_string[-1] in ignore_n and count == length - 1:
            for matra in ignore_n:
                conversions[matra].append('')

        if letter not in conversions:
            translits = add_next_letter(translits, letter)

        if add_a and letter not in dont_append_a:
            translits = add_next_sound(translits, 'a')

        letter_sounds = conversions.get(letter, [''])
        translits = add_next_letter(translits, letter_sounds)
        add_a = letter not in dont_append_a

    if hindi_string[-2] == '्':
        translits = add_next_letter(translits, ['a', "e"])

    return set(translits)


def add_next_sound(translits, letter):
    translits.extend([translit + letter for translit in translits])
    return translits


def add_next_letter(translits, letters):
    translits_with_letters = []
    for letter in letters:
        translits_with_letters.extend(
            [translit + letter for translit in translits])
    return translits_with_letters


if __name__ == "__main__":
    print(transliterations('लौल'))
    print(transliterations('लोल'))
    print(transliterations("तुग़लक़"))
