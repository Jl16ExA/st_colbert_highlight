# Jina Document Highlighter

## Overview

The **Jina Document Highlighter** is a Streamlit-based web application that leverages machine learning to highlight relevant sections of a document based on a user's query. This tool uses the `jina-colbert-v2` model from Jina AI to compute relevance scores between the input document and query, providing both a visual highlight of the relevant text and a heatmap representing the relevance of the document tokens to the query.

## Features

- **Document and Query Embedding:** Generates embeddings for both the document and query using the `jina-colbert-v2` model.
- **Relevance Scoring:** Computes relevance scores between query and document embeddings to determine the most relevant sections of the document.
- **Text Highlighting:** Highlights the relevant portions of the document text based on a configurable relevance threshold.
- **Relevance Heatmap:** Generates a heatmap visualization that shows the relevance of each document token to the query.

## Installation

To run the Jina Document Highlighter, you need to install the required dependencies. You can do this using `pip`:

\`\`\`bash
pip install streamlit torch transformers matplotlib seaborn annotated_text pillow requests
\`\`\`

## Usage

### Running the Application

To start the application, navigate to the directory containing the `JinaDocumentHighlighter` script and run:

\`\`\`bash
streamlit run main.py
\`\`\`

### Application Interface

1. **API Key Configuration:**

   - Enter your Jina API key in the sidebar to enable the embedding functionality.

2. **Document and Query Input:**

   - Input the text of the document you wish to analyze in the "Document Input" section.
   - Enter your search query in the "Query Input" section.

3. **Highlighting Threshold:**

   - Adjust the threshold slider to control the sensitivity of the highlighting.

4. **Analyze Document:**

   - Click the "Analyze Document" button to process the input document and query.

5. **Highlighted Document:**

   - View the document with the relevant portions highlighted based on the query.

6. **Relevance Heatmap:**
   - A heatmap visualization of the relevance scores will be displayed, showing the relationship between the query and document tokens.

### Example

\`\`\`python
document_text = "The quick brown fox jumps over the lazy dog. The fox is known for its agility and speed."
query_text = "fox speed"

highlighted_doc = highlighter.get_highlighted_document(document_text, query_text)
heatmap_image = highlighter.create_heatmap(document_text, query_text)

# Displaying results

print("Highlighted Document:")
print(highlighted_doc)
heatmap_image.show()
\`\`\`

## Code Structure

- **JinaDocumentHighlighter Class:**

  - **\`**init**(self, jina_api_key: str)\`**: Initializes the class with the Jina API key and loads the tokenizer.
  - **\`preprocess_doc(self, doc)\`**: Pads document embeddings to a consistent length for processing.
  - **\`compute_relevance_scores(self, query_embeddings, document_embeddings)\`**: Computes relevance scores between query and document embeddings.
  - **\`tokenize(self, text, is_query=True)\`**: Tokenizes the input text for query or document processing.
  - **\`highlight_relevant_parts(self, document, query, threshold=0.5)\`**: Highlights relevant parts of the document based on the computed relevance scores.
  - **\`embed(self, text: str, is_query: bool = True)\`**: Calls the Jina API to embed the text.
  - **\`get_highlighted_document(self, document_text, query_text, threshold=0.5)\`**: Returns the highlighted document based on the query.
  - **\`create_heatmap(self, document_text, query_text)\`**: Generates a heatmap visualization of query-document relevance.

- **Streamlit Application (main function):**
  - Handles user inputs, configures the JinaDocumentHighlighter, and displays the results.

## Requirements

- Python 3.6 or higher
- Jina API Key (required for embedding functionality)

## License

This project is licensed under the MIT License.

## Contact

For any inquiries or issues, please contact the project maintainer at [juandlopezb160@gmail.com].
