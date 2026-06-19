"""
NLP Training Orchestrator — AI-Powered Lie Detection System
Runs the full training pipeline: baseline → intermediate → transformer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.nlp.preprocessor import NLPPreprocessor, LinguisticFeatureExtractor
from src.nlp.feature_extractor import TFIDFExtractor, Word2VecExtractor, TransformerEmbeddingExtractor
from src.nlp.baseline_model import TFIDFLogisticRegression, XGBoostTextClassifier, SVMTextClassifier
from src.nlp.transformer_model import TransformerDeceptionClassifier
from src.utils.logger import logger
from src.utils.config import config
from src.utils.helpers import set_seed, save_json


class NLPTrainer:
    """
    Orchestrates the complete NLP training pipeline across all model tiers.

    Pipeline:
    1. Preprocess text (clean + tokenize + lemmatize)
    2. Extract features (TF-IDF, Word2Vec, Linguistic)
    3. Train baseline models (LR, XGBoost, SVM)
    4. Fine-tune transformer (BERT/RoBERTa/DeBERTa)
    5. Save best models and evaluation results
    """

    def __init__(
        self,
        output_dir: str = "models/nlp",
        results_dir: str = "reports",
        seed: int = 42,
    ):
        set_seed(seed)
        self.output_dir = Path(output_dir)
        self.results_dir = Path(results_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.preprocessor = NLPPreprocessor(remove_stopwords=False, lemmatize=True)
        self.ling_extractor = LinguisticFeatureExtractor()
        self.results: Dict = {}

    def prepare_data(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        label_col: str = "label",
        val_size: float = 0.15,
        test_size: float = 0.15,
    ) -> Tuple:
        """
        Preprocess and split data into train/val/test.

        Returns:
            (X_train_texts, X_val_texts, X_test_texts,
             y_train, y_val, y_test,
             train_tokens, val_tokens, test_tokens)
        """
        df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)
        texts = df[text_col].tolist()
        labels = df[label_col].astype(int).tolist()

        logger.info(f"Dataset size: {len(texts)}, Label dist: {pd.Series(labels).value_counts().to_dict()}")

        # Preprocess
        logger.info("Preprocessing texts...")
        processed_texts, token_lists = self.preprocessor.process_batch(texts)

        # Split
        X_trainval, X_test, y_trainval, y_test, tok_trainval, tok_test = train_test_split(
            processed_texts, labels, token_lists, test_size=test_size, stratify=labels, random_state=42
        )
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val, tok_train, tok_val = train_test_split(
            X_trainval, y_trainval, tok_trainval, test_size=val_ratio, stratify=y_trainval, random_state=42
        )

        logger.info(f"Splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        return (X_train, X_val, X_test,
                np.array(y_train), np.array(y_val), np.array(y_test),
                tok_train, tok_val, tok_test)

    def train_baseline(
        self,
        X_train: List[str], y_train: np.ndarray,
        X_val: List[str], y_val: np.ndarray,
        X_test: List[str], y_test: np.ndarray,
    ) -> Dict:
        """Train TF-IDF + LR and TF-IDF + XGBoost baselines."""
        results = {}

        # ── TF-IDF Extraction ─────────────────────────────────
        logger.info("Fitting TF-IDF vectorizer...")
        tfidf = TFIDFExtractor()
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_val_tfidf = tfidf.transform(X_val)
        X_test_tfidf = tfidf.transform(X_test)
        tfidf.save(str(self.output_dir / "tfidf_vectorizer.pkl"))

        # ── Linguistic Features ───────────────────────────────
        logger.info("Extracting linguistic features...")
        X_train_ling = self.ling_extractor.extract_batch(X_train)
        X_val_ling = self.ling_extractor.extract_batch(X_val)
        X_test_ling = self.ling_extractor.extract_batch(X_test)

        # ── LR ────────────────────────────────────────────────
        lr_model = TFIDFLogisticRegression()
        lr_model.fit(X_train_tfidf, y_train)
        results["tfidf_lr"] = lr_model.evaluate(X_test_tfidf, y_test)
        lr_model.save(str(self.output_dir / "lr_model.pkl"))
        logger.info(f"LR ROC-AUC: {results['tfidf_lr']['roc_auc']:.4f}")

        # ── XGBoost on TF-IDF ─────────────────────────────────
        xgb_model = XGBoostTextClassifier()
        xgb_model.fit(X_train_tfidf, y_train, X_val_tfidf, y_val)
        results["tfidf_xgb"] = xgb_model.evaluate(X_test_tfidf, y_test)
        xgb_model.save(str(self.output_dir / "xgb_tfidf_model.json"))
        logger.info(f"XGBoost (TF-IDF) ROC-AUC: {results['tfidf_xgb']['roc_auc']:.4f}")

        return results

    def train_word2vec(
        self,
        tok_train: List[List[str]], y_train: np.ndarray,
        tok_val: List[List[str]], y_val: np.ndarray,
        tok_test: List[List[str]], y_test: np.ndarray,
    ) -> Dict:
        """Train Word2Vec + XGBoost."""
        results = {}

        logger.info("Training Word2Vec embeddings...")
        w2v = Word2VecExtractor(vector_size=300, pooling="concat")
        w2v.fit(tok_train)
        X_train_w2v = w2v.transform(tok_train)
        X_val_w2v = w2v.transform(tok_val)
        X_test_w2v = w2v.transform(tok_test)
        w2v.save(str(self.output_dir / "word2vec.model"))

        xgb_model = XGBoostTextClassifier()
        xgb_model.fit(X_train_w2v, y_train, X_val_w2v, y_val)
        results["w2v_xgb"] = xgb_model.evaluate(X_test_w2v, y_test)
        xgb_model.save(str(self.output_dir / "xgb_w2v_model.json"))
        logger.info(f"Word2Vec + XGBoost ROC-AUC: {results['w2v_xgb']['roc_auc']:.4f}")

        return results

    def train_transformer(
        self,
        X_train: List[str], y_train: np.ndarray,
        X_val: List[str], y_val: np.ndarray,
        X_test: List[str], y_test: np.ndarray,
        model_name: str = "roberta-base",
        num_epochs: int = 5,
        batch_size: int = 16,
    ) -> Dict:
        """Fine-tune a transformer model."""
        logger.info(f"Fine-tuning transformer: {model_name}")
        safe_name = model_name.replace("/", "_")
        output_path = str(self.output_dir / safe_name)

        clf = TransformerDeceptionClassifier(model_name=model_name)
        history = clf.train(
            X_train, list(y_train),
            X_val, list(y_val),
            num_epochs=num_epochs,
            batch_size=batch_size,
            output_dir=output_path,
        )

        _, probas = clf.predict(X_test, batch_size=batch_size)
        preds = np.argmax(probas, axis=1)
        from sklearn.metrics import classification_report, roc_auc_score
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        auc = roc_auc_score(y_test, probas[:, 1])

        results = {
            "model_name": model_name,
            "history": history,
            "test_report": report,
            "test_roc_auc": auc,
        }
        logger.info(f"Transformer {model_name} — Test AUC: {auc:.4f}")
        return results

    def run_full_pipeline(
        self,
        df: pd.DataFrame,
        train_transformers: bool = True,
        transformer_models: Optional[List[str]] = None,
    ) -> Dict:
        """
        Execute the full NLP training pipeline.

        Args:
            df: DataFrame with 'text' and 'label' columns
            train_transformers: Whether to fine-tune transformers (requires GPU)
            transformer_models: List of model names to fine-tune

        Returns:
            Dictionary of all evaluation results
        """
        if transformer_models is None:
            transformer_models = ["roberta-base"]

        # Prepare data
        (X_train, X_val, X_test,
         y_train, y_val, y_test,
         tok_train, tok_val, tok_test) = self.prepare_data(df)

        # Baseline models
        logger.info("=== Phase 1: Baseline Models ===")
        baseline_results = self.train_baseline(X_train, y_train, X_val, y_val, X_test, y_test)
        self.results.update(baseline_results)

        # Word2Vec intermediate
        logger.info("=== Phase 2: Word2Vec + XGBoost ===")
        w2v_results = self.train_word2vec(tok_train, y_train, tok_val, y_val, tok_test, y_test)
        self.results.update(w2v_results)

        # Transformers
        if train_transformers:
            logger.info("=== Phase 3: Transformer Fine-tuning ===")
            for model_name in transformer_models:
                transformer_results = self.train_transformer(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    model_name=model_name,
                )
                self.results[model_name] = transformer_results

        # Save results
        save_json(self.results, str(self.results_dir / "nlp_results.json"))
        logger.info(f"All NLP results saved to {self.results_dir / 'nlp_results.json'}")

        return self.results


def main():
    """Entry point for CLI training."""
    import argparse
    parser = argparse.ArgumentParser(description="Train NLP deception detection models")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV/TSV dataset")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--transformers", action="store_true", help="Fine-tune transformers")
    parser.add_argument("--model", type=str, default="roberta-base")
    parser.add_argument("--output-dir", type=str, default="models/nlp")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    trainer = NLPTrainer(output_dir=args.output_dir)
    results = trainer.run_full_pipeline(
        df,
        train_transformers=args.transformers,
        transformer_models=[args.model],
    )
    print(json.dumps(
        {k: v.get("roc_auc", "N/A") if isinstance(v, dict) else "N/A"
         for k, v in results.items()},
        indent=2
    ))


if __name__ == "__main__":
    main()
