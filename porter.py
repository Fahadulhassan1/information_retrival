# Contains all functions related to the porter stemming algorithm.

from document import Document


def get_measure(term: str) -> int:
    vowels = "aeiou"
    consonants = ''.join(set('abcdefghijklmnopqrstuvwxyz') - set(vowels))
    m = 0
    current_is_vowel = term[0] in vowels
    for letter in term[1:]:
        if current_is_vowel and letter in consonants:
            m += 1
            current_is_vowel = False
        elif not current_is_vowel and letter in vowels:
            current_is_vowel = True
    return m



def condition_v(stem: str) -> bool:
    """
    Returns whether condition *v* is true for a given stem (= the stem contains a vowel).
    :param stem: Word stem to check
    :return: True if the condition *v* holds
    """
    vowels = "aeiou"
    for letter in stem:
        if letter in vowels:
            return True
    return False



def condition_d(stem: str) -> bool:
    """
    Returns whether condition *d is true for a given stem (= the stem ends with a double consonant (e.g. -TT, -SS)).
    :param stem: Word stem to check
    :return: True if the condition *d holds
    """
    consonants = ''.join(set('abcdefghijklmnopqrstuvwxyz') - set("aeiou"))
    return len(stem) > 1 and stem[-1] == stem[-2] and stem[-1] in consonants


def cond_o(stem: str) -> bool:
    """
    Returns whether condition *o is true for a given stem (= the stem ends cvc, where the second c is not W, X or Y
    (e.g. -WIL, -HOP)).
    :param stem: Word stem to check
    :return: True if the condition *o holds
    """
    if len(stem) < 3:
        return False
    vowels = "aeiou"
    consonants = ''.join(set('abcdefghijklmnopqrstuvwxyz') - set(vowels))
    return (stem[-1] in consonants and
            stem[-1] not in "wxy" and
            stem[-2] in vowels and
            stem[-3] in consonants)


def stem_term(term: str) -> str:
    """
    Stems a given term of the English language using the Porter stemming algorithm.
    :param term:
    :return:
    """
    # TODO: Implement this function. (PR03)
    # Note: See the provided file "porter.txt" for information on how to implement it!
    if len(term) <= 2:
        return term

    # Step 1a
    if term.endswith("sses"):
        term = term[:-2]
    elif term.endswith("ies"):
        term = term[:-2]
    elif term.endswith("ss"):
        term = term
    elif term.endswith("s"):
        term = term[:-1]

    # Step 1b
    if term.endswith("eed"):
        if get_measure(term[:-3]) > 0:
            term = term[:-1]
    elif term.endswith("ed") or term.endswith("ing"):
        stem = term[:-2] if term.endswith("ed") else term[:-3]
        if condition_v(stem):
            term = stem
            if term.endswith("at") or term.endswith("bl") or term.endswith("iz"):
                term += "e"
            elif condition_d(term) and not (term.endswith("l") or term.endswith("s") or term.endswith("z")):
                term = term[:-1]
            elif get_measure(term) == 1 and cond_o(term):
                term += "e"

    # Step 1c
    if term.endswith("y") and condition_v(term[:-1]):
        term = term[:-1] + "i"

    # Steps 2 to 5...
    # Step 2
        if term.endswith('ational') and get_measure(term[:-5]) > 0:
            term = term[:-5] + 'e'
        elif term.endswith('tional') and get_measure(term[:-2]) > 0:
            term = term[:-2]
        elif term.endswith('enic') and get_measure(term[:-1]) > 0:
            term = term[:-1] + 'e'
        elif term.endswith('anci') and get_measure(term[:-1]) > 0:
            term = term[:-1] + 'e'
        elif term.endswith('izer') and get_measure(term[:-1]) > 0:
            term = term[:-1]
        elif term.endswith('abli') and get_measure(term[:-1]) > 0:
            term = term[:-1] + 'e'
        elif term.endswith('alli') and get_measure(term[:-2]) > 0:
            term = term[:-2]
        elif term.endswith('entli') and get_measure(term[:-2]) > 0:
            term = term[:-1]
        elif term.endswith('eli') and get_measure(term[:-2]) > 0:
            term = term[:-2]
        elif term.endswith('ousli') and get_measure(term[:-2]) > 0:
            term = term[:-2]
        elif term.endswith('ization') and get_measure(term[:-5]) > 0:
            term = term[:-5] + 'e'
        elif term.endswith('ation') and get_measure(term[:-4]) > 0:
            term = term[:-4] + 'e'
        elif term.endswith('ator') and get_measure(term[:-2]) > 0:
            term = term[:-2] + 'e'
        elif term.endswith('alism') and get_measure(term[:-3]) > 0:
            term = term[:-3]
        elif term.endswith('iveness') and get_measure(term[:-4]) > 0:
            term = term[:-4]
        elif term.endswith('fulness') and get_measure(term[:-4]) > 0:
            term = term[:-4]
        elif term.endswith('ousness') and get_measure(term[:-4]) > 0:
            term = term[:-4]
        elif term.endswith('aliti') and get_measure(term[:-3]) > 0:
            term = term[:-3]
        elif term.endswith('iviti') and get_measure(term[:-3]) > 0:
            term = term[:-3] + 'e'
        elif term.endswith('biliti') and get_measure(term[:-5]) > 0:
            term = term[:-5] + 'le'
        elif term.endswith('xflurti') and get_measure(term[:-6]) > 0:
            term = term[:-6] + 'ti'

        if term.endswith('icate') and get_measure(term[:-3]) > 0:
            term = term[:-3]
        elif term.endswith('ative') and get_measure(term[:-5]) > 0:
            term = term[:-5]
        elif term.endswith('alize') and get_measure(term[:-3]) > 0:
            term = term[:-3]
        elif term.endswith('icite') and get_measure(term[:-3]) > 0:
            term = term[:-3]
        elif term.endswith('ical') and get_measure(term[:-2]) > 0:
            term = term[:-2]
        elif term.endswith('ful') and get_measure(term[:-3]) > 0:
            term = term[:-3]
        elif term.endswith('ness') and get_measure(term[:-4]) > 0:
            term = term[:-4]
        
        if term.endswith('ement') and get_measure(term[:-5]) > 0:
            term = term[:-5]
        elif term.endswith(('ance', 'ence', 'able', 'ible', 'ment')) and get_measure(term[:-4]) > 1:
            term = term[:-4]
        elif term.endswith(('ant', 'ent', 'ism', 'ate', 'iti', 'ous', 'ive', 'ize')) and get_measure(term[:-3]) > 1:
            term = term[:-3]
        elif term.endswith(('al', 'er', 'ic', 'ou')) and get_measure(term[:-2]) > 1:
            term = term[:-2]
        elif term.endswith('ion') and get_measure(term[:-3]) > 1 and (term[-1] == 's' or term[-1] == 't'):
            term = term[:-3]
        
        if term.endswith('e'):
            if get_measure(term[:-1]) > 1 or (get_measure(term[:-1]) == 1 and not cond_o(term[:-1])):
                term = term[:-1]

        # Step 5b
        if get_measure(term) > 1 and term.endswith('ll'):
            term = term[:-1]
    # Apply further steps as per the algorithm description

    return term

def stem_all_documents(collection: list[Document]):
    """
    For each document in the given collection, this method uses the stem_term() function on all terms in its term list.
    Warning: The result is NOT saved in the document's term list, but in the extra field stemmed_terms!
    :param collection: Document collection to process
    """
    # TODO: Implement this function. (PR03)
    for document in collection:
        document.stemmed_terms = [stem_term(term) for term in document.terms]


def stem_query_terms(query: str) -> str:
    """
    Stems all terms in the provided query string.
    :param query: User query, may contain Boolean operators and spaces.
    :return: Query with stemmed terms
    """
    # TODO: Implement this function. (PR03)
    terms = query.split()
    stemmed_terms = [stem_term(term) for term in terms]
    return ' '.join(stemmed_terms)
