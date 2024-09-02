import streamlit as st
import torch
import torch.nn.functional as F
import requests
from transformers import AutoTokenizer
from annotated_text import annotated_text
import matplotlib.pyplot as plt
import seaborn as sns
import io
import numpy as np
from PIL import Image

class JinaDocumentHighlighter:
    def __init__(self, jina_api_key: str):
        self.auto_tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-colbert-v2")
        self.jina_api_key = jina_api_key

    def preprocess_doc(self, doc):
        embeddings = [torch.Tensor(doc["embeddings"])]
        max_length = max(embd.size() for embd in embeddings)[0]
        padded_embeddings = [
            F.pad(embd, (0, 0, 0, max_length - embd.size()[0]), "constant", 0)
            for embd in embeddings
        ]
        return torch.stack(padded_embeddings)

    def compute_relevance_scores(self, query_embeddings, document_embeddings):
        scores = torch.matmul(
            query_embeddings.unsqueeze(0), document_embeddings.transpose(1, 2)
        )
        max_scores_per_query_term = scores.max(dim=2).values
        total_scores = max_scores_per_query_term.sum(dim=1)
        return scores[0], total_scores

    def tokenize(self, text, is_query=True):
        auto_tokens = self.auto_tokenizer.tokenize(
            ". " + text,
            padding=False,
            truncation=True,
            return_token_type_ids=False,
        )
        if is_query:
            auto_tokens.insert(0, '[CLS]')
            auto_tokens[1] = '<Q>'
            auto_tokens.append('[SEP]')
        else:
            auto_tokens.insert(0, '[CLS]')
            auto_tokens.insert(1, '<D>')
            auto_tokens.append('[SEP]')
        return auto_tokens

    def highlight_relevant_parts(self, document, query, threshold=0.5):
        document_embeddings = self.preprocess_doc(document)
        query_embeddings = torch.Tensor(query["embeddings"])
        
        scores, _ = self.compute_relevance_scores(query_embeddings, document_embeddings)
        
        document_tokens = self.tokenize(document["text"], is_query=False)
        
        # Normalize scores
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        # Get the maximum score for each token
        max_scores, _ = normalized_scores.max(dim=0)
        
        highlighted_text = []
        for i, token in enumerate(document_tokens):
            if i < len(max_scores):
                score = max_scores[i].item()  # Convert to Python scalar
                if score > threshold:
                    highlighted_text.append((token, "", f"rgba(255, 255, 0, {score})"))
                else:
                    highlighted_text.append(token)
            else:
                highlighted_text.append(token)
        
        return highlighted_text

    def embed(self, text: str, is_query: bool = True):
        input_type = "query" if is_query else "document"
        trimmed_text = text[:10000]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.jina_api_key}",
        }
        data = {
            "input": trimmed_text,
            "model": "jina-colbert-v2",
            "input_type": input_type,
        }
        response = requests.post(
            url="https://api.jina.ai/v1/multi-embeddings", headers=headers, json=data
        )
        if response.status_code == 200:
            embeddings = response.json()["data"]
            if embeddings:
                embedded_docs = [{"text": text, "embeddings": embeddings[0]["embeddings"]}]
            return embedded_docs
        else:
            st.error(f"API request failed with status code {response.status_code}")
            st.error(f"Response: {response.text}")
            return [{"text": text}]

    def get_highlighted_document(self, document_text, query_text, threshold=0.5):
        embedded_query = self.embed(query_text, is_query=True)[0]
        embedded_doc = self.embed(document_text, is_query=False)[0]
        
        return self.highlight_relevant_parts(embedded_doc, embedded_query, threshold)

    def create_heatmap(self, document_text, query_text):
        embedded_query = self.embed(query_text, is_query=True)[0]
        embedded_doc = self.embed(document_text, is_query=False)[0]

        document_embeddings = self.preprocess_doc(embedded_doc)
        query_embeddings = torch.Tensor(embedded_query["embeddings"])
        
        scores, _ = self.compute_relevance_scores(query_embeddings, document_embeddings)
        
        query_tokens = self.tokenize(query_text, is_query=True)
        document_tokens = self.tokenize(document_text, is_query=False)

        fig, ax = plt.subplots(figsize=(12, len(query_tokens) * 0.5))
        scores_np = scores.cpu().detach().numpy()
    
        im = ax.imshow(scores_np, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(document_tokens)))
        ax.set_yticks(range(len(query_tokens)))
        ax.set_xticklabels(document_tokens, rotation=90)
        ax.set_yticklabels(query_tokens)
        plt.colorbar(im)
        plt.title("Query-Document Relevance Heatmap")
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return Image.open(buf)
def main():
    st.set_page_config(page_title="Jina Document Highlighter", layout="wide")
    st.title("Jina Document Highlighter")

    # Sidebar for API key input
    st.sidebar.title("Configuration")
    jina_api_key = st.sidebar.text_input("Enter your Jina API key", type="password")

    if not jina_api_key:
        st.warning("Please enter your Jina API key in the sidebar to proceed.")
        return
    highlighter = JinaDocumentHighlighter(jina_api_key)

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Document Input")
        document_text = st.text_area("Enter your document text here:", height=200)

    with col2:
        st.subheader("Query Input")
        query_text = st.text_input("Enter your query here:")

    threshold = st.slider("Highlighting Threshold", 0.0, 1.0, 0.5, 0.01)

    if st.button("Analyze Document"):
        if document_text and query_text:
            with st.spinner("Analyzing document..."):
                highlighted_doc = highlighter.get_highlighted_document(document_text, query_text, threshold)
                heatmap_image = highlighter.create_heatmap(document_text, query_text)

            st.subheader("Highlighted Document")
            annotated_text(*highlighted_doc)

            st.subheader("Relevance Heatmap")
            st.image(heatmap_image, use_column_width=True)
        else:
            st.warning("Please enter both document text and query to proceed with analysis.")

if __name__ == "__main__":
    main()