import hashlib
from bitarray import bitarray
from abc import ABC, abstractmethod

from document import Document
import re
import numpy as np
from collections import defaultdict
from typing import List, Dict
from document import Document
from porter import stem_term
import math

import porter
class RetrievalModel(ABC):
    @abstractmethod
    def document_to_representation(
        self, document: Document, stopword_filtering=False, stemming=False
    ):
        """
        Converts a document into its model-specific representation.
        This is an abstract method and not meant to be edited. Implement it in the subclasses!
        :param document: Document object to be represented
        :param stopword_filtering: Controls, whether the document should first be freed of stopwords
        :param stemming: Controls, whether stemming is used on the document's terms
        :return: A representation of the document. Data type and content depend on the implemented model.
        """
        raise NotImplementedError()

    @abstractmethod
    def query_to_representation(self, query: str):
        """
        Determines the representation of a query according to the model's concept.
        :param query: Search query of the user
        :return: Query representation in whatever data type or format is required by the model.
        """
        raise NotImplementedError()

    @abstractmethod
    def match(self, document_representation, query_representation) -> float:
        """
        Matches the query and document presentation according to the model's concept.
        :param document_representation: Data that describes one document
        :param query_representation:  Data that describes a query
        :return: Numerical approximation of the similarity between the query and document representation. Higher is
        "more relevant", lower is "less relevant".
        """
        raise NotImplementedError()


class LinearBooleanModel(RetrievalModel):
    # TODO: Implement all abstract methods and __init__() in this class. (PR02)
    def __init__(self):
        self.documents = []

    def document_to_representation(
        self, document: Document, stopword_filtering=False, stemming=False
    ):
        """
        Converts a document into a list of terms (words).
        :param document: Document object to be represented
        :param stopword_filtering: Controls, whether the document should first be freed of stopwords
        :param stemming: Controls, whether stemming is used on the document's terms
        :return: A list of terms representing the document
        """
        terms = document.terms
        terms = [term.lower() for term in terms]  # Convert all terms to lowercase

        if stopword_filtering:
            terms = [term for term in terms if term not in document.filtered_terms]
        return terms

    def query_to_representation(self, query: str):
        """
        Converts a query into a list of terms (words).
        :param query: Search query of the user
        :return: A list of terms representing the query
        """
        return query.lower().split()

    def match(self, document_representation, query_representation) -> float:
        """
        Matches the query and document presentation based on Boolean search.
        :param document_representation: List of terms that describes one document
        :param query_representation: List of terms that describes a query
        :return: 1.0 if the query term is in the document, 0.0 otherwise
        """
        for query_term in query_representation:
            if query_term in document_representation:
                return 1.0
        return 0.0

    def __str__(self):
        return "Boolean Model (Linear)"


class InvertedListBooleanModel(RetrievalModel):
    def __init__(self):
        self.inverted_index = {}
        self.docs = []
        self.is_ready = False

    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        terms = set()

        if stopword_filtering:
            terms.update(term.lower() for term in document.filtered_terms)
        else:
            terms.update(term.lower() for term in document.raw_text.split() if term)

        if stemming:
            terms = {porter.stem_term(term) for term in terms}

        return terms

    def query_to_representation(self, query: str, stemming=False):
    # Split the query into terms and operators
        terms = re.split(r'(\W)', query.lower())

        # Remove empty strings and whitespace-only strings
        terms = [term for term in terms if term.strip()]
        
        if stemming:
            terms = [porter.stem_term(term) if term.isalnum() else term for term in terms]

        return terms

    def match(self, document_representation, query_representation):
        return all(term in document_representation for term in query_representation)

    def build_inverted_list(self, documents, stopword_filtering=False, stemming=False):
        self.docs = documents
        self.inverted_index = {}
        for doc_id, document in enumerate(documents):
            terms = self.document_to_representation(document, stopword_filtering, stemming)
            for term in terms:
                if term not in self.inverted_index:
                    self.inverted_index[term] = set()
                self.inverted_index[term].add(doc_id)
        self.is_ready = True

    def __str__(self):
        return 'Boolean Model (Inverted Index)'

class SignatureBasedBooleanModel:
    def __init__(self, collection):
        self.collection = collection  # List of Document objects
        self.signatures = {}  # Dictionary to store document signatures
        self.F = 64  # Size of signature bit array
        self.D = 4   # Block size (terms per block)
        self.p = 157  # Prime number for hashing
        self.m = 5   # Number of hash functions
        self.build_signature_index()  # Build signature index on initialization

    def build_signature_index(self):
        """
        Build the signature index for the entire document collection.
        """
        for doc in self.collection:
            document_signatures = []
            terms = doc.terms  # Assuming `terms` is a list of terms from the document

            for i in range(0, len(terms), self.D):
                block_terms = terms[i:i + self.D]
                block_signature = np.zeros(self.F, dtype=int)
                for term in block_terms:
                    term_signature = self.generate_signature(term)
                    block_signature |= term_signature  # Combine term signatures using bitwise OR
                document_signatures.append(block_signature)

            self.signatures[doc.document_id] = document_signatures

    def generate_signature(self, word):
        """
        Generate a bit signature for a given term using m hash functions.
        """
        signature = np.zeros(self.F, dtype=int)

        for i in range(self.m):
            hash_value = 0
            for char in word:
                hash_value = (hash_value + ord(char)) * (i * self.p)
            hash_value = hash_value % self.F
            signature[hash_value] = 1  # Set bit to 1 for the computed hash value

        return signature

    def document_to_representation(self, document, stopword_filtering=False, stemming=False):
        """
        Get the precomputed signature for a document by its ID.
        """
        return self.signatures.get(document.document_id, [])

    def query_to_representation(self, query: str):
        """
        Convert the query into its signature representation.
        """
        query_terms = query.lower().split()  # Split query into terms
        query_signature = np.zeros(self.F, dtype=int)

        for term in query_terms:
            term_signature = self.generate_signature(term)
            query_signature |= term_signature  # Combine term signatures using bitwise OR

        return query_signature

    def match(self, document_representation, query_representation):
        """
        Check if the query signature is a subset of any of the document's block signatures.
        Returns 1.0 if a match is found, otherwise 0.0.
        """
        for block_signature in document_representation:
            if np.all((block_signature & query_representation) == query_representation):
                return 1.0
        return 0.0

    def search(self, query: str):
        """
        Perform search for a query by matching against all document signatures.
        """
        query_representation = self.query_to_representation(query)
        results = []

        for doc in self.collection:
            doc_representation = self.document_to_representation(doc)
            if self.match(doc_representation, query_representation):
                results.append(doc)

        return results
    
    def _create_signature(self, terms):
        """
        Creates a signature bitarray for the given terms.
        """
        signature = bitarray(self.F)
        signature.setall(0)

        for term in terms:
            hash_value = self._hash_function(term)
            for i in range(self.D):
                pos = (hash_value + i) % self.F
                signature[pos] = 1
        return signature


class VectorSpaceModel:
    def __init__(self, collection: List[Document]):
        self.collection = collection
        self.inverted_index, self.term_to_index = self.build_inverted_index()
        self.document_vectors = self.build_document_vectors()

    def __str__(self):
        return 'Vector Space Model'

    def build_inverted_index(self) -> (Dict[str, List[int]], Dict[str, int]):
        """
        Build an inverted index where terms map to document IDs.
        Also create a term-to-index map for efficient vector indexing.
        """
        inverted_index = defaultdict(list)
        term_to_index = {}  # Term to index mapping for efficient vector access
        index_counter = 0

        for idx, doc in enumerate(self.collection):
            unique_terms = set(doc.terms)
            for term in unique_terms:
                if term not in inverted_index:
                    inverted_index[term].append(idx)
                    if term not in term_to_index:
                        term_to_index[term] = index_counter
                        index_counter += 1
                else:
                    inverted_index[term].append(idx)

        return inverted_index, term_to_index

    def build_document_vectors(self) -> Dict[int, np.ndarray]:
        """
        Build document vectors using TF-IDF weighting.
        """
        doc_vectors = {}
        N = len(self.collection)  # Total number of documents

        # Pre-calculate document frequencies (DF) for each term
        df = {term: len(set(self.inverted_index[term])) for term in self.inverted_index}

        for idx, doc in enumerate(self.collection):
            vector = np.zeros(len(self.term_to_index))
            term_counts = defaultdict(int)

            for term in doc.terms:
                term_counts[term] += 1

            for term, count in term_counts.items():
                if term in self.inverted_index:
                    # Term Frequency (TF)
                    tf = count
                    # Inverse Document Frequency (IDF)
                    idf = math.log(N / (df[term] + 1))  # Added 1 to avoid division by zero
                    vector[self.term_to_index[term]] = tf * idf

            # Normalize the vector to avoid bias
            norm = np.linalg.norm(vector)
            if norm != 0:
                vector = vector / norm
            doc_vectors[idx] = vector

        return doc_vectors

    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        """
        Convert a document into a vector representation.
        """
        doc_id = self.collection.index(document)
        return self.document_vectors[doc_id]

    def query_to_representation(self, query: str):
        """
        Convert a query into a vector representation using TF-IDF weighting.
        Uses augmented normalized term frequency for the query.
        """
        query_terms = query.lower().split()
        query_vector = np.zeros(len(self.term_to_index))
        N = len(self.collection)

        term_counts = defaultdict(int)
        for term in query_terms:
            term_counts[term] += 1

        # Calculate max TF for normalization in augmented TF
        max_tf = max(term_counts.values())

        # Calculate DF for query terms and update query vector
        df = {term: len(set(self.inverted_index[term])) for term in term_counts if term in self.inverted_index}

        for term, count in term_counts.items():
            if term in self.inverted_index:
                # Augmented Term Frequency (TF)
                augmented_tf = 0.5 + (0.5 * count / max_tf)
                # Inverse Document Frequency (IDF)
                idf = math.log(N / (df.get(term, 0) + 1))  # Avoid division by zero
                query_vector[self.term_to_index[term]] = augmented_tf * idf

        # Normalize the query vector
        norm = np.linalg.norm(query_vector)
        if norm != 0:
            query_vector = query_vector / norm
        return query_vector

    def match(self, document_representation, query_representation) -> float:

        dot_product = np.dot(document_representation, query_representation)
        norm_doc = np.linalg.norm(document_representation)
        norm_query = np.linalg.norm(query_representation)
        if norm_doc == 0 or norm_query == 0:
            return 0.0
        return dot_product / (norm_doc * norm_query)

    def search(self, query: str) -> List[Document]:
        start_time = time.time()

        query_vector = self.query_to_representation(query)
        scores = []

        for idx, doc_vector in self.document_vectors.items():
            score = self.match(doc_vector, query_vector)
            if score > 0:
                scores.append((score, self.collection[idx]))

        scores.sort(reverse=True, key=lambda x: x[0])

        end_time = time.time()
        print(f"Search executed in {end_time - start_time:.4f} seconds")

        return [doc for score, doc in scores]
class FuzzySetModel(RetrievalModel):
    # TODO: Implement all abstract methods. (PR04)
    def __init__(self):
        raise NotImplementedError()  # TODO: Remove this line and implement the function.

    def __str__(self):
        return "Fuzzy Set Model"
