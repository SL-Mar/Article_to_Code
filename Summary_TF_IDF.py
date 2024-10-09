### Info extraction based on Tf-Idf

import re
import pdfplumber
import spacy
import networkx as nx
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple
from itertools import combinations
import numpy as np

class PDFLoader:
    """Handles loading and extracting text from PDF files."""
    def load_pdf(self, pdf_path: str) -> str:
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except FileNotFoundError:
            print(f"PDF file not found: {pdf_path}")
        except Exception as e:
            print(f"An error occurred while loading the PDF: {e}")
        return text

class TextPreprocessor:
    """Handles preprocessing of extracted text."""
    def __init__(self):
        # Precompile regex patterns for performance
        self.multinew_pattern = re.compile(r'\n+')
        self.url_pattern = re.compile(r'https?://\S+')
        self.phrase_pattern = re.compile(r'Electronic copy available at: .*', re.IGNORECASE)
        self.number_pattern = re.compile(r'^\d+\s*$', re.MULTILINE)

        self.nlp = spacy.load("en_core_web_sm")

    def preprocess_text(self, text: str) -> str:
        text = self.url_pattern.sub('', text)  # Remove URLs
        text = self.phrase_pattern.sub('', text)  # Remove undesired phrases
        text = self.number_pattern.sub('', text)  # Remove standalone numbers
        text = self.multinew_pattern.sub(' ', text)  # Replace multiple newlines with a space
        text = text.strip()
        
        # Lemmatize and remove stopwords
        doc = self.nlp(text)
        lemmatized_text = " ".join([token.lemma_ for token in doc if not token.is_stop])
        
        # Post-process to improve readability by keeping only alphabetic tokens
        refined_text = " ".join([token.text for token in self.nlp(lemmatized_text) if token.is_alpha])
        return refined_text

class TradingTextRank:
    """Performs extractive summarization using TextRank."""
    def __init__(self, model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except Exception as e:
            raise Exception(f"SpaCy model '{model}' could not be loaded: {e}")

    def rank_sentences(self, text: str, top_n: int = 5) -> List[str]:
        """
        Rank sentences using TextRank to generate an extractive summary.
        """
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]

        if len(sentences) == 0:
            return []

        # Generate sentence embeddings using TF-IDF
        vectorizer = TfidfVectorizer().fit_transform(sentences)
        vectors = vectorizer.toarray()

        # Create a similarity matrix using cosine similarity
        similarity_matrix = nx.Graph()
        for i, j in combinations(range(len(sentences)), 2):
            norm_i = np.linalg.norm(vectors[i])
            norm_j = np.linalg.norm(vectors[j])
            if norm_i == 0 or norm_j == 0:
                cosine_similarity = 0
            else:
                cosine_similarity = vectors[i].dot(vectors[j]) / (norm_i * norm_j)
            if cosine_similarity > 0:
                similarity_matrix.add_edge(i, j, weight=cosine_similarity)

        # Rank sentences using PageRank
        if len(similarity_matrix) == 0:
            return []
        
        scores = nx.pagerank(similarity_matrix)
        ranked_sentences = sorted(((scores.get(i, 0), sent) for i, sent in enumerate(sentences)), reverse=True)

        # Extract the top-ranked sentences as summary
        summary = [sent for _, sent in ranked_sentences[:top_n]]
        return summary

class StrategyExtractor:
    """Extracts sentences that are related to trading strategy design."""
    def __init__(self):
        self.trading_keywords = {
            "strategy", "indicator", "trend", "moving average", "rsi", "bollinger bands", "entry", "exit",
            "signal", "stop-loss", "take profit", "risk management", "momentum", "position sizing",
            "double bottom", "double top", "support", "resistance", "market timing", "swing trading"
        }

    def filter_sentences(self, sentences: List[str]) -> List[str]:
        """
        Filter sentences based on the presence of trading-related keywords.
        """
        return [sentence for sentence in sentences if any(kw in sentence.lower() for kw in self.trading_keywords)]

    def categorize_sentences(self, sentences: List[str]) -> dict:
        """
        Categorize sentences into key components of a trading strategy.
        """
        categories = {
            'Entry Conditions': [],
            'Exit Conditions': [],
            'Indicators Used': [],
            'Risk Management': [],
            'Trade Frequency and Universe': []
        }

        for sentence in sentences:
            lower_sentence = sentence.lower()
            if any(keyword in lower_sentence for keyword in ["entry", "buy", "enter", "double bottom", "double top", "support"]):
                categories['Entry Conditions'].append(f"Entry Signal: {sentence}")
            elif any(keyword in lower_sentence for keyword in ["exit", "sell", "stop-loss", "take profit", "resistance"]):
                categories['Exit Conditions'].append(f"Exit Signal: {sentence}")
            elif any(keyword in lower_sentence for keyword in ["indicator", "moving average", "rsi", "bollinger bands", "sentiment"]):
                categories['Indicators Used'].append(f"Indicator Used: {sentence}")
            elif any(keyword in lower_sentence for keyword in ["risk management", "leverage", "position sizing", "drawdown", "portfolio"]):
                categories['Risk Management'].append(f"Risk Management Rule: {sentence}")
            elif any(keyword in lower_sentence for keyword in ["frequency", "monthly", "daily", "etf", "universe"]):
                categories['Trade Frequency and Universe'].append(f"Trade Frequency/Universe: {sentence}")

        return categories

    def refine_summary(self, categorized_sentences: dict) -> str:
        """
        Refine the categorized sentences into a cohesive summary for a developer.
        """
        summary = []
        for category, sentences in categorized_sentences.items():
            if sentences:
                summary.append(f"{category}:")
                for sentence in sentences:
                    summary.append(f"- {sentence}")
                summary.append("")
        return "\n".join(summary)

    def enhance_summary(self, categorized_sentences: dict) -> str:
        """
        Enhance the categorized sentences by adding clarifications and inferred details for a developer.
        """
        enhanced_summary = []
        for category, sentences in categorized_sentences.items():
            if sentences:
                enhanced_summary.append(f"{category}:")
                for sentence in sentences:
                    if category == 'Entry Conditions':
                        enhanced_summary.append(f"- {sentence} (e.g., look for confirmation with RSI above 30 or a breakout above resistance)")
                    elif category == 'Exit Conditions':
                        enhanced_summary.append(f"- {sentence} (e.g., set a stop-loss at 5% below entry price or take profit at a 10% gain)")
                    elif category == 'Indicators Used':
                        enhanced_summary.append(f"- {sentence} (e.g., use a 50-day moving average for trend confirmation)")
                    elif category == 'Risk Management':
                        enhanced_summary.append(f"- {sentence} (e.g., maintain leverage below 2:1 and limit drawdown to 10%)")
                    else:
                        enhanced_summary.append(f"- {sentence}")
                enhanced_summary.append("")
        return "\n".join(enhanced_summary)

# Main function to process the PDF and extract a summary
def main(pdf_path: str, top_n: int = 10):
    pdf_loader = PDFLoader()
    preprocessor = TextPreprocessor()
    text_rank = TradingTextRank()
    strategy_extractor = StrategyExtractor()

    # Load and preprocess PDF text
    raw_text = pdf_loader.load_pdf(pdf_path)
    if not raw_text:
        print("No text extracted from the PDF.")
        return

    preprocessed_text = preprocessor.preprocess_text(raw_text)
    
    # Extractive summarization
    ranked_sentences = text_rank.rank_sentences(preprocessed_text, top_n=top_n * 2)  # Get more sentences to filter
    if not ranked_sentences:
        print("No sentences ranked for summarization.")
        return
    
    trading_summary = strategy_extractor.filter_sentences(ranked_sentences)[:top_n]  # Further refine based on keywords

    if not trading_summary:
        print("No trading strategy-related sentences found in the summary.")
        return

    # Categorize and enhance the summary
    categorized_summary = strategy_extractor.categorize_sentences(trading_summary)
    enhanced_summary = strategy_extractor.enhance_summary(categorized_summary)
    print("Enhanced Extractive Summary (Trading Strategy Related):")
    print(enhanced_summary)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate extractive summary for a quant finance article.")
    parser.add_argument('pdf_path', type=str, help='Path to the PDF file to process.')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top sentences to include in the summary.')
    args = parser.parse_args()

    main(args.pdf_path, top_n=args.top_n)
