import json
import os

import cleanup
import extraction
import models
import porter
from document import Document
import re
import numpy as np
from document import Document
import math

import time

# Important paths:
RAW_DATA_PATH = "raw_data"
DATA_PATH = "data"
COLLECTION_PATH = os.path.join(DATA_PATH, "my_collection.json")
STOPWORD_FILE_PATH = os.path.join(DATA_PATH, "stopwords.json")

# Menu choices:
(
    CHOICE_LIST,
    CHOICE_SEARCH,
    CHOICE_EXTRACT,
    CHOICE_UPDATE_STOP_WORDS,
    CHOICE_SET_MODEL,
    CHOICE_SHOW_DOCUMENT,
    CHOICE_EXIT,
) = (1, 2, 3, 4, 5, 6, 9)
MODEL_BOOL_LIN, MODEL_BOOL_INV, MODEL_BOOL_SIG, MODEL_FUZZY, MODEL_VECTOR = (
    1,
    2,
    3,
    4,
    5,
)
SW_METHOD_LIST, SW_METHOD_CROUCH = 1, 2


class InformationRetrievalSystem(object):
    def __init__(self):
        if not os.path.isdir(DATA_PATH):
            os.makedirs(DATA_PATH)

        # Collection of documents, initially empty.
        try:
            self.collection = extraction.load_collection_from_json(COLLECTION_PATH)
        except FileNotFoundError:
            print("No previous collection was found. Creating empty one.")
            self.collection = []

        # Stopword list, initially empty.
        try:
            with open(STOPWORD_FILE_PATH, "r") as f:
                self.stop_word_list = json.load(f)
        except FileNotFoundError:
            print("No stopword list was found.")
            self.stop_word_list = []

        self.model = None  # Saves the current IR model in use.
        self.output_k = 5  # Controls how many results should be shown for a query.

    def main_menu(self):
        """
        Provides the main loop of the CLI menu that the user interacts with.
        """
        while True:
            print(f"Current retrieval model: {self.model}")
            print(f"Current collection: {len(self.collection)} documents")
            print()
            print("Please choose an option:")
            print(f"{CHOICE_LIST} - List documents")
            print(f"{CHOICE_SEARCH} - Search for term")
            print(f"{CHOICE_EXTRACT} - Build collection")
            print(f"{CHOICE_UPDATE_STOP_WORDS} - Rebuild stopword list")
            print(f"{CHOICE_SET_MODEL} - Set model")
            print(f"{CHOICE_SHOW_DOCUMENT} - Show a specific document")
            print(f"{CHOICE_EXIT} - Exit")
            
            try:
                action_choice = int(input("Enter choice: "))
            except ValueError:
                print("Invalid choice. Please enter a number.")
                continue

            if action_choice == CHOICE_LIST:
                # List documents in CLI.
                if self.collection:
                    for document in self.collection:
                        print(document)
                else:
                    print("No documents.")
                print()

            elif action_choice == CHOICE_SEARCH:
                # Read a query string from the CLI and search for it.

                # Determine desired search parameters:
                SEARCH_NORMAL, SEARCH_SW, SEARCH_STEM, SEARCH_SW_STEM = 1, 2, 3, 4
                print("Search options:")
                print(f"{SEARCH_NORMAL} - Standard search (default)")
                print(f"{SEARCH_SW} - Search documents with removed stopwords")
                print(f"{SEARCH_STEM} - Search documents with stemmed terms")
                print(
                    f"{SEARCH_SW_STEM} - Search documents with removed stopwords AND stemmed terms"
                )
                search_mode = int(input("Enter choice: "))
                stop_word_filtering = (search_mode == SEARCH_SW) or (
                    search_mode == SEARCH_SW_STEM
                )
                stemming = (search_mode == SEARCH_STEM) or (
                    search_mode == SEARCH_SW_STEM
                )

                # Actual query processing begins here:
                query = input("Query: ")
                if stemming:
                    query = porter.stem_query_terms(query)
                start_time = time.time()  # Start measuring time

                if isinstance(self.model, models.InvertedListBooleanModel):
                    results = self.inverted_list_search(
                        query, stemming, stop_word_filtering
                    )
                elif isinstance(self.model, models.VectorSpaceModel):
                    results = self.buckley_lewit_search(
                        query, stemming, stop_word_filtering
                    )
                elif isinstance(self.model, models.SignatureBasedBooleanModel):
                    results = self.signature_search(
                        query, stemming, stop_word_filtering
                    )
                else:
                    results = self.basic_query_search(
                        query, stemming, stop_word_filtering
                    )
                end_time = time.time()  # End measuring time

                # Output of results:
                for score, document in results:
                    print(f"{score}: {document}")

                # Output of quality metrics:
                print()
                print(f'precision: {self.calculate_precision(results)}')
                print(f'recall: {self.calculate_recall(results)}')

                processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
                print(f'Query processing time: {processing_time:.2f} ms')

            elif action_choice == CHOICE_EXTRACT:
                # Extract document collection from text file.

                raw_collection_file = os.path.join(RAW_DATA_PATH, "aesopa10.txt")
                self.collection = extraction.extract_collection(raw_collection_file)
                assert isinstance(self.collection, list)
                assert all(isinstance(d, Document) for d in self.collection)

                if input("Should stopwords be filtered? [y/N]: ") == "y":
                    cleanup.filter_collection(self.collection)

                if input("Should stemming be performed? [y/N]: ") == "y":
                    porter.stem_all_documents(self.collection)

                extraction.save_collection_as_json(self.collection, COLLECTION_PATH)
                print("Done.\n")

            elif action_choice == CHOICE_UPDATE_STOP_WORDS:
                # Rebuild the stop word list, using one out of two methods.

                print("Available options:")
                print(f"{SW_METHOD_LIST} - Load stopword list from file")
                print(
                    f"{SW_METHOD_CROUCH} - Generate stopword list using Crouch's method"
                )

                method_choice = int(input("Enter choice: "))
                if method_choice in (SW_METHOD_LIST, SW_METHOD_CROUCH):
                    # Load stop words using the desired method:
                    if method_choice == SW_METHOD_LIST:
                        self.stop_word_list = cleanup.load_stop_word_list(
                            os.path.join(RAW_DATA_PATH, "englishST.txt")
                        )
                        print("Done.\n")
                    elif method_choice == SW_METHOD_CROUCH:
                        self.stop_word_list = (
                            cleanup.create_stop_word_list_by_frequency(self.collection)
                        )
                        print("Done.\n")

                    # Save new stopword list into file:
                    with open(STOPWORD_FILE_PATH, "w") as f:
                        json.dump(self.stop_word_list, f)
                else:
                    print("Invalid choice.")

            elif action_choice == CHOICE_SET_MODEL:
                # Choose and set the retrieval model to use for searches.

                print()
                print("Available models:")
                print(f"{MODEL_BOOL_LIN} - Boolean model with linear search")
                print(f"{MODEL_BOOL_INV} - Boolean model with inverted lists")
                print(f"{MODEL_BOOL_SIG} - Boolean model with signature-based search")
                print(f"{MODEL_FUZZY} - Fuzzy set model")
                print(f"{MODEL_VECTOR} - Vector space model")
                model_choice = int(input("Enter choice: "))
                if model_choice == MODEL_BOOL_LIN:
                    self.model = models.LinearBooleanModel()
                elif model_choice == MODEL_BOOL_INV:
                    self.model = models.InvertedListBooleanModel()
                elif model_choice == MODEL_BOOL_SIG:
                    self.model = models.SignatureBasedBooleanModel(self.collection)
                elif model_choice == MODEL_FUZZY:
                    self.model = models.FuzzySetModel()
                elif model_choice == MODEL_VECTOR:
                    self.model = models.VectorSpaceModel(self.collection)
                else:
                    print("Invalid choice.")

            elif action_choice == CHOICE_SHOW_DOCUMENT:
                target_id = int(input("ID of the desired document:"))
                found = False
                for document in self.collection:
                    if document.document_id == target_id:
                        print(document.title)
                        print("-" * len(document.title))
                        print(document.raw_text)
                        found = True

                if not found:
                    print(f"Document #{target_id} not found!")

            elif action_choice == CHOICE_EXIT:
                break
            else:
                print("Invalid choice.")

            print()
            input("Press ENTER to continue...")
            print()

    def basic_query_search(
        self, query: str, stemming: bool, stop_word_filtering: bool
    ) -> list:
        """
        Searches the collection for a query string. This method is "basic" in that it does not use any special algorithm
        to accelerate the search. It simply calculates all representations and matches them, returning a sorted list of
        the k most relevant documents and their scores.
        :param query: Query string
        :param stemming: Controls, whether stemming is used
        :param stop_word_filtering: Controls, whether stop-words are ignored in the search
        :return: List of tuples, where the first element is the relevance score and the second the corresponding
        document
        """
        query_representation = self.model.query_to_representation(query)
        document_representations = [
            self.model.document_to_representation(d, stop_word_filtering, stemming)
            for d in self.collection
        ]
        scores = [
            self.model.match(dr, query_representation)
            for dr in document_representations
        ]
        ranked_collection = sorted(
            zip(scores, self.collection), key=lambda x: x[0], reverse=True
        )
        results = ranked_collection[: self.output_k]
        return results

    def inverted_list_search(
        self, query: str, stemming: bool, stop_word_filtering: bool
    ) -> list:
        """
        Fast Boolean query search for inverted lists.
        :param query: Query string
        :param stemming: Controls, whether stemming is used
        :param stop_word_filtering: Controls, whether stop-words are ignored in the search
        :return: List of tuples, where the first element is the relevance score and the second the corresponding
        document
        """
        if not self.model.is_ready:

            self.model.build_inverted_list(self.collection, stop_word_filtering, stemming)

        query_terms = self.model.query_to_representation(query, stemming)
        operand_stack = []
        operator_stack = []

        def execute_operator(operator, left_set, right_set):
            if operator == '&':
                return left_set & right_set
            elif operator == '|':
                return left_set | right_set
            elif operator == '-':
                return left_set - right_set
            else:
                raise ValueError(f"Invalid operator: {operator}")

        for token in query_terms:
            if token in {'&', '|', '-'}:
                operator_stack.append(token)
            elif token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    right_set = operand_stack.pop()
                    left_set = operand_stack.pop()
                    operator = operator_stack.pop()
                    operand_stack.append(execute_operator(operator, left_set, right_set))
                if operator_stack and operator_stack[-1] == '(':
                    operator_stack.pop()  # Remove '('
            else:
                operand_stack.append(self.model.inverted_index.get(token, set()))

            while len(operand_stack) >= 2 and operator_stack and operator_stack[-1] not in {'(', ')'}:
                right_set = operand_stack.pop()
                left_set = operand_stack.pop()
                operator = operator_stack.pop()
                operand_stack.append(execute_operator(operator, left_set, right_set))

        while len(operand_stack) >= 2 and operator_stack:
            right_set = operand_stack.pop()
            left_set = operand_stack.pop()
            operator = operator_stack.pop()
            operand_stack.append(execute_operator(operator, left_set, right_set))

        if len(operand_stack) != 1:
            raise ValueError("Malformed query: Mismatch between operators and operands")

        final_result_set = operand_stack[0] if operand_stack else set()
        search_results = [(1, self.collection[doc_id]) for doc_id in final_result_set]
        return search_results

    def buckley_lewit_search(self, query: str, stemming: bool, stop_word_filtering: bool) -> list:
        """
        Fast query search for the Vector Space Model using the algorithm by Buckley & Lewit.
        :param query: Query string
        :param stemming: Controls, whether stemming is used
        :param stop_word_filtering: Controls, whether stop-words are ignored in the search
        :return: List of tuples, where the first element is the relevance score and the second the corresponding document
        """
        # Preprocess query terms
        query_terms = query.lower().split()
        N = len(self.collection)  # Number of documents

        # Initialize the query vector with zeros
        query_vector = np.zeros(len(self.model.inverted_index))

        # Calculate TF-IDF for query terms and construct the query vector
        term_frequencies = {term: query_terms.count(term) for term in set(query_terms)}
        max_tf = max(term_frequencies.values())

        for term, tf in term_frequencies.items():
            if term in self.model.inverted_index:
                df = len(self.model.inverted_index[term])  # Document frequency
                idf = math.log(N / (df + 1))  # Add 1 to avoid division by zero
                term_index = list(self.model.inverted_index.keys()).index(term)
                query_vector[term_index] = (0.5 + 0.5 * tf / max_tf) * idf

        # Normalize the query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector /= query_norm

        # Calculate cosine similarity scores for each document
        scores = np.zeros(N)
        for idx, doc_vector in self.model.document_vectors.items():
            doc_norm = np.linalg.norm(doc_vector)
            if doc_norm > 0:
                scores[idx] = np.dot(doc_vector, query_vector) / doc_norm

        # Sort the documents by score in descending order and select the top-k results
        top_k = self.output_k
        top_k_indices = scores.argsort()[::-1][:top_k]

        # Only return results where the score is greater than 0
        results = [(scores[idx], self.collection[idx]) for idx in top_k_indices if scores[idx] > 0]

        return results
    
    def signature_search(self, query: str, stemming: bool, stop_word_filtering: bool) -> list:
        """
        Performs a signature-based search for documents that match the query using logical AND on the signatures.
        :param query: The search query string
        :param stemming: Whether stemming should be applied
        :param stop_word_filtering: Whether stop words should be removed
        :return: List of tuples where the first element is the relevance score and the second is the document
        """

    # Search for documents by matching signatures with the query signature
        def search_documents(model, query):
            query_signature = model.query_to_representation(query)  # Create query signature

            matching_documents = []  # To hold document IDs that match the query
            for doc_id, doc_signatures in model.signatures.items():
                # Check if any document signature block matches the query signature using AND logic
                for doc_signature in doc_signatures:
                    if np.all((doc_signature & query_signature) == query_signature):
                        matching_documents.append(doc_id)
                        break  # Stop after first match for the document
            return matching_documents

        # Retrieve document IDs that match the query
        matched_document_ids = search_documents(self.model, query)

        # Filter and collect matched documents from the collection
        matched_documents = [
            (self.calculate_relevance(query, doc), doc)  # Adjust relevance score based on query
            for doc in self.collection if doc.document_id in matched_document_ids
        ]

        # Sort matched documents by relevance score in descending order
        matched_documents.sort(reverse=True, key=lambda x: x[0])

        # Return the top-k relevant documents
        return matched_documents[:self.output_k]
    def calculate_relevance(self, query, document):
        """
        Calculate the relevance of a document based on the number of matching terms between the query and document.
        :param query: The search query string
        :param document: The document object being evaluated
        :return: Relevance score based on the number of matched terms
        """
        query_terms = set(query.lower().split())
        doc_terms = set(document.terms)

        # Calculate the relevance as the ratio of matched terms to the total number of terms in the query
        matched_terms = query_terms.intersection(doc_terms)
        relevance_score = len(matched_terms) / len(query_terms) if query_terms else 0

        return relevance_score
    
    def calculate_precision(self, result_list: list[tuple]) -> float:
        try:
            with open(os.path.join(RAW_DATA_PATH, "ground_truth.txt"), "r") as f:
                ground_truth = f.read().splitlines()

            relevant_docs = set()
            for line in ground_truth:
                if not line.strip() or line.startswith("#"):
                    # Skip empty lines and comments
                    continue

                parts = line.split(' - ')
                if len(parts) == 2:
                    term, doc_ids = parts
                    relevant_docs.update(map(int, doc_ids.split(', ')))
                else:
                    print(f"Skipping malformed line in ground_truth.txt: {line}")

            retrieved_docs = {doc.document_id for _, doc in result_list}
            true_positives = len(relevant_docs.intersection(retrieved_docs))

            return true_positives / len(retrieved_docs) if retrieved_docs else 0

        except FileNotFoundError:
            return 0
        except Exception as e:
            print(f"An error occurred while calculating precision: {e}")
            return 0

    def calculate_recall(self, result_list: list[tuple]) -> float:
        try:
            with open(os.path.join(RAW_DATA_PATH, "ground_truth.txt"), "r") as f:
                ground_truth = f.read().splitlines()

            relevant_docs = set()
            for line in ground_truth:
                if not line.strip() or line.startswith("#"):
                    # Skip empty lines and comments
                    continue

                parts = line.split(' - ')
                if len(parts) == 2:
                    term, doc_ids = parts
                    relevant_docs.update(map(int, doc_ids.split(', ')))
                else:
                    print(f"Skipping malformed line in ground_truth.txt: {line}")

            retrieved_docs = {doc.document_id for _, doc in result_list}
            true_positives = len(relevant_docs.intersection(retrieved_docs))

            return true_positives / len(relevant_docs) if relevant_docs else 0

        except FileNotFoundError:
            return 0
        except Exception as e:
            print(f"An error occurred while calculating recall: {e}")
            return 0
    

if __name__ == "__main__":
    irs = InformationRetrievalSystem()
    irs.main_menu()
    exit(0)
