# Contains all functions that deal with stop word removal.

from document import Document
from collections import Counter

import re
import os

RAW_DATA_PATH = "raw_data"


def remove_symbols(text_string: str) -> str:
    """
    Removes all punctuation marks and similar symbols from a given string.
    Occurrences of "'s" are removed as well.
    :param text:
    :return:
    """

    # Remove "'s" but keep the rest of the text
    text_string = re.sub(r"'s\b", "", text_string)
    # Remove all other punctuation marks
    text_string = re.sub(r"[^\w\s]", "", text_string)
    return text_string


def is_stop_word(term: str, stop_word_list: list[str]) -> bool:
    """
    Checks if a given term is a stop word.
    :param stop_word_list: List of all considered stop words.
    :param term: The term to be checked.
    :return: True if the term is a stop word.
    """
    return term.lower() in stop_word_list


def remove_stop_words_from_term_list(term_list: list[str]) -> list[str]:
    """
    Takes a list of terms and removes all terms that are stop words.
    :param term_list: List that contains the terms
    :return: List of terms without stop words
    """
    raw_collection_file = os.path.join(RAW_DATA_PATH, "englishST.txt")

    # Hint:  Implement the functions remove_symbols() and is_stop_word() first and use them here.
    stop_word_list = load_stop_word_list(
        raw_collection_file
    )  # Load stop words from file
    cleaned_terms = []
    for term in term_list:
        # Remove symbols from term
        term = remove_symbols(term)
        # Check if term is a stop word
        if not is_stop_word(term, stop_word_list):
            cleaned_terms.append(term)
    return cleaned_terms


def filter_collection(collection: list[Document]):
    """
    For each document in the given collection, this method takes the term list and filters out the stop words.
    Warning: The result is NOT saved in the documents term list, but in an extra field called filtered_terms.
    :param collection: Document collection to process
    """
    for document in collection:
        filtered_terms = remove_stop_words_from_term_list(document.terms)
        document.filtered_terms = filtered_terms


def load_stop_word_list(raw_file_path: str) -> list[str]:
    """
    Loads a text file that contains stop words and saves it as a list. The text file is expected to be formatted so that
    each stop word is in a new line, e. g. like englishST.txt
    :param raw_file_path: Path to the text file that contains the stop words
    :return: List of stop words
    """
    with open(raw_file_path, "r") as file:
        stop_words = file.read().splitlines()
    stop_words = [word.lower() for word in stop_words]
    return stop_words


def create_stop_word_list_by_frequency(collection: list[Document]) -> list[str]:
    """
    Uses the method of J. C. Crouch (1990) to generate a stop word list by finding high and low frequency terms in the
    provided collection.
    :param collection: Collection to process
    :return: List of stop words
    """
    term_frequency = {}
    for doc in collection:
        for term in doc.terms:
            term_frequency[term] = term_frequency.get(term, 0) + 1

    # Set threshold for high and low-frequency terms
    threshold = 50  # Adjust as needed

    # Identify high and low-frequency terms
    high_frequency_terms = [
        term for term, frequency in term_frequency.items() if frequency > threshold
    ]
    # we assume low frequency terms are stopwords
    low_frequency_terms = [
        term for term, frequency in term_frequency.items() if frequency <= threshold
    ]
    # Converting back to a list

    return low_frequency_terms
