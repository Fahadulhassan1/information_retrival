# Contains functions that deal with the extraction of documents from a text file (see PR01)

import json

from document import Document


def extract_collection(source_file_path: str) -> list[Document]:
    """
    Loads a text file (aesopa10.txt) and extracts each of the listed fables/stories from the file.
    :param source_file_name: File name of the file that contains the fables
    :return: List of Document objects
    """
    catalog = []  # List to store the document objects
    document_id = 0  # Unique document ID counter

    with open(source_file_path, "r", encoding="utf-8") as file:
        # Skip lines until reaching line 308
        for _ in range(307):
            next(file)

        title = ""  # Temporary variable to store current title
        current_data = ""  # String to collect raw text for a document
        is_reading_text = False  # Flag to indicate if reading document content
        empty_line_counter = 0  # Counter to detect empty lines and document boundaries

        for line in file:
            line = line.strip()  # Remove leading and trailing whitespaces

            # Detect empty lines and handle transitions between documents
            if line == "":
                empty_line_counter += 1
                if is_reading_text and empty_line_counter >= 2:
                    # End of current document
                    doc = Document()
                    doc.title = title
                    doc.document_id = document_id
                    doc.terms = current_data.split()  # Split the raw data into terms
                    doc.raw_text = current_data.strip()
                    catalog.append(doc)

                    # Reset for next document
                    document_id += 1
                    title = ""
                    current_data = ""
                    is_reading_text = False
                continue

            # Title section
            if not is_reading_text and line:
                title = line
                is_reading_text = True
                empty_line_counter = 0  # Reset empty line counter after the title

            # Content section
            if is_reading_text and line:
                current_data += line + " "
                empty_line_counter = 0  # Reset empty line counter when reading content

        # Handle the last document if the file ends without enough blank lines
        if current_data:
            doc = Document()
            doc.title = title
            doc.document_id = document_id
            doc.terms = current_data.split()
            doc.raw_text = current_data.strip()
            catalog.append(doc)

    return catalog


def save_collection_as_json(collection: list[Document], file_path: str) -> None:
    """
    Saves the collection to a JSON file.
    :param collection: The collection to store (list of Document objects)
    :param file_path: Path of the JSON file
    """
    serializable_collection = []
    for document in collection:
        serializable_collection.append({
            "document_id": document.document_id,
            "title": document.title,
            "raw_text": document.raw_text,
            "terms": document.terms,
            "filtered_terms": getattr(document, "filtered_terms", None),  # Handle missing fields
            "stemmed_terms": getattr(document, "stemmed_terms", None),  # Handle missing fields
        })

    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(serializable_collection, json_file, ensure_ascii=False, indent=4)


def load_collection_from_json(file_path: str) -> list[Document]:
    """
    Loads the collection from a JSON file.
    :param file_path: Path of the JSON file
    :return: list of Document objects
    """
    try:
        with open(file_path, "r", encoding="utf-8") as json_file:
            json_collection = json.load(json_file)

        collection = []
        for doc_dict in json_collection:
            document = Document()
            document.document_id = doc_dict.get("document_id")
            document.title = doc_dict.get("title")
            document.raw_text = doc_dict.get("raw_text")
            document.terms = doc_dict.get("terms")
            document.filtered_terms = doc_dict.get("filtered_terms")
            document.stemmed_terms = doc_dict.get("stemmed_terms")
            collection.append(document)

        return collection
    except FileNotFoundError:
        print("No collection was found. Creating empty one.")
        return []
