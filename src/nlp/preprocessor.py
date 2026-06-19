"""
Text Preprocessor — AI-Powered Lie Detection System
Full NLP preprocessing pipeline: cleaning, tokenization, feature preparation.
"""
from __future__ import annotations

import re
import string
from typing import List, Optional, Tuple

import numpy as np

from src.utils.logger import logger


# ──────────────────────────────────────────────────────────────
# Lazy imports for optional heavy deps
# ──────────────────────────────────────────────────────────────

def _get_nltk():
    import nltk
    for resource in ["stopwords", "wordnet", "punkt", "averaged_perceptron_tagger"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
    return nltk


def _get_spacy():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")


# ──────────────────────────────────────────────────────────────
# Text Cleaner
# ──────────────────────────────────────────────────────────────

class TextCleaner:
    """
    Rule-based text cleaning pipeline.
    Handles HTML, URLs, special characters, whitespace, and casing.
    """

    HTML_PATTERN = re.compile(r"<[^>]+>")
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    EMAIL_PATTERN = re.compile(r"\S+@\S+")
    MENTION_PATTERN = re.compile(r"@\w+")
    HASHTAG_PATTERN = re.compile(r"#\w+")
    NUMBER_PATTERN = re.compile(r"\b\d+\b")
    MULTI_SPACE_PATTERN = re.compile(r"\s+")
    PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)

    def __init__(
        self,
        lowercase: bool = True,
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_mentions: bool = False,
        remove_hashtags: bool = False,
        replace_numbers: bool = False,
        remove_punctuation: bool = False,
    ):
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.replace_numbers = replace_numbers
        self.remove_punctuation = remove_punctuation

    def clean(self, text: str) -> str:
        """Apply full cleaning pipeline to a single text."""
        if not isinstance(text, str):
            return ""

        if self.remove_html:
            text = self.HTML_PATTERN.sub(" ", text)
        if self.remove_urls:
            text = self.URL_PATTERN.sub(" ", text)
        if self.remove_emails:
            text = self.EMAIL_PATTERN.sub(" ", text)
        if self.remove_mentions:
            text = self.MENTION_PATTERN.sub(" ", text)
        if self.remove_hashtags:
            text = self.HASHTAG_PATTERN.sub(" ", text)
        if self.replace_numbers:
            text = self.NUMBER_PATTERN.sub("[NUM]", text)
        if self.remove_punctuation:
            text = text.translate(self.PUNCTUATION_TABLE)
        if self.lowercase:
            text = text.lower()

        text = self.MULTI_SPACE_PATTERN.sub(" ", text).strip()
        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a list of texts."""
        return [self.clean(t) for t in texts]


# ──────────────────────────────────────────────────────────────
# NLP Preprocessor (Tokenization + Stopword + Lemmatization)
# ──────────────────────────────────────────────────────────────

class NLPPreprocessor:
    """
    Full NLP preprocessing pipeline using NLTK.
    Steps: clean → tokenize → stopword removal → lemmatization.
    """

    def __init__(
        self,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        min_word_length: int = 2,
    ):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_word_length = min_word_length
        self.cleaner = TextCleaner()
        self._nltk = None
        self._stop_words = None
        self._lemmatizer = None

    def _init_nltk(self) -> None:
        if self._nltk is None:
            self._nltk = _get_nltk()
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            self._stop_words = set(stopwords.words("english"))
            self._lemmatizer = WordNetLemmatizer()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into word tokens."""
        self._init_nltk()
        return self._nltk.word_tokenize(text)

    def process(self, text: str) -> Tuple[str, List[str]]:
        """
        Full preprocessing pipeline.
        Returns: (processed_text_string, token_list)
        """
        self._init_nltk()
        cleaned = self.cleaner.clean(text)
        tokens = self.tokenize(cleaned)

        processed_tokens = []
        for token in tokens:
            if len(token) < self.min_word_length:
                continue
            if self.remove_stopwords and token in self._stop_words:
                continue
            if self.lemmatize:
                token = self._lemmatizer.lemmatize(token)
            processed_tokens.append(token)

        return " ".join(processed_tokens), processed_tokens

    def process_batch(self, texts: List[str]) -> Tuple[List[str], List[List[str]]]:
        """Process a list of texts."""
        results = [self.process(t) for t in texts]
        processed_texts = [r[0] for r in results]
        token_lists = [r[1] for r in results]
        return processed_texts, token_lists


# ──────────────────────────────────────────────────────────────
# Linguistic Feature Extractor (deception-specific)
# ──────────────────────────────────────────────────────────────

class LinguisticFeatureExtractor:
    """
    Extracts handcrafted linguistic features known to correlate with deception.

    Features based on:
    - LIWC-style categories (negations, certainty, sensory words)
    - Verb tense analysis
    - Lexical richness
    - Statement complexity
    """

    NEGATION_WORDS = {
        "no", "not", "never", "nothing", "nowhere", "neither",
        "nobody", "none", "cannot", "won't", "don't", "didn't",
        "doesn't", "hadn't", "haven't", "isn't", "aren't", "wasn't",
    }

    CERTAINTY_WORDS = {
        "always", "never", "definitely", "absolutely", "certainly",
        "completely", "totally", "surely", "undoubtedly", "obviously",
    }

    HEDGE_WORDS = {
        "maybe", "perhaps", "probably", "possibly", "might", "could",
        "somewhat", "kind of", "sort of", "I think", "I guess",
    }

    FIRST_PERSON_SINGULAR = {"i", "me", "my", "myself", "mine"}
    FIRST_PERSON_PLURAL = {"we", "us", "our", "ourselves", "ours"}

    def extract(self, text: str, tokens: Optional[List[str]] = None) -> np.ndarray:
        """
        Extract linguistic features as a fixed-size numpy vector.

        Returns array of shape (15,) with features:
        [word_count, avg_word_len, unique_ratio, negation_count,
         certainty_count, hedge_count, fp_singular_ratio, fp_plural_ratio,
         exclamation_count, question_count, comma_count, digit_count,
         sentence_count, avg_sentence_len, punctuation_ratio]
        """
        if tokens is None:
            tokens = text.lower().split()

        word_count = len(tokens)
        if word_count == 0:
            return np.zeros(15, dtype=np.float32)

        lower_tokens = [t.lower() for t in tokens]
        lower_text = text.lower()

        avg_word_len = np.mean([len(t) for t in tokens]) if tokens else 0.0
        unique_ratio = len(set(lower_tokens)) / word_count
        negation_count = sum(1 for t in lower_tokens if t in self.NEGATION_WORDS)
        certainty_count = sum(1 for t in lower_tokens if t in self.CERTAINTY_WORDS)
        hedge_count = sum(1 for t in lower_tokens if t in self.HEDGE_WORDS)
        fp_singular = sum(1 for t in lower_tokens if t in self.FIRST_PERSON_SINGULAR)
        fp_plural = sum(1 for t in lower_tokens if t in self.FIRST_PERSON_PLURAL)
        exclamation_count = text.count("!")
        question_count = text.count("?")
        comma_count = text.count(",")
        digit_count = sum(c.isdigit() for c in text)
        sentences = re.split(r"[.!?]+", text.strip())
        sentence_count = max(len([s for s in sentences if s.strip()]), 1)
        avg_sentence_len = word_count / sentence_count
        punctuation_ratio = sum(c in string.punctuation for c in text) / max(len(text), 1)

        return np.array([
            word_count,
            avg_word_len,
            unique_ratio,
            negation_count / word_count,
            certainty_count / word_count,
            hedge_count / word_count,
            fp_singular / word_count,
            fp_plural / word_count,
            exclamation_count,
            question_count,
            comma_count,
            digit_count,
            sentence_count,
            avg_sentence_len,
            punctuation_ratio,
        ], dtype=np.float32)

    def extract_batch(self, texts: List[str]) -> np.ndarray:
        """Extract features for a list of texts, returns (N, 15) array."""
        return np.vstack([self.extract(t) for t in texts])

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "word_count", "avg_word_len", "unique_ratio",
            "negation_ratio", "certainty_ratio", "hedge_ratio",
            "fp_singular_ratio", "fp_plural_ratio",
            "exclamation_count", "question_count", "comma_count",
            "digit_count", "sentence_count", "avg_sentence_len",
            "punctuation_ratio",
        ]
