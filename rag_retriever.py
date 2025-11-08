
import os
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import faiss

SentenceTransformer = None

class RAGRetriever:
    def __init__(self, model_name: str = "paraphrase-MiniLM-L3-v2", use_merged_index: bool = True):
        print("Initializing RAGRetriever (CPU mode)...")

        global SentenceTransformer
        if SentenceTransformer is None:
            try:
                from sentence_transformers import SentenceTransformer as _ST
                SentenceTransformer = _ST
            except Exception as e:
                raise RuntimeError("sentence_transformers is required for RAGRetriever. Please install it.") from e

        self.model = SentenceTransformer(model_name)
        try:
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        except Exception:
            self.embedding_dim = None

        self.index: Optional[faiss.Index] = None
        self.df: Optional[pd.DataFrame] = None
        self.use_merged_index = use_merged_index
        self.index_source = "unknown"
        self._load_resources()

    def _load_resources(self):
        merged_idx_path = "data/medquad_faiss_merged.idx"
        merged_meta_csv = "data/medquad_faiss_merged_meta.csv"
        default_idx_path = "data/medquad_faiss.idx"
        default_meta_csv = "data/medquad_faiss_meta.csv"

        idx_path = None
        meta_csv = None

        if self.use_merged_index and os.path.exists(merged_idx_path) and os.path.exists(merged_meta_csv):
            idx_path = merged_idx_path
            meta_csv = merged_meta_csv
            self.index_source = "merged (MEDQUAD + MIMIC-IV)"
        elif os.path.exists(default_idx_path) and os.path.exists(default_meta_csv):
            idx_path = default_idx_path
            meta_csv = default_meta_csv
            self.index_source = "default (MEDQUAD only)"

        if idx_path and meta_csv:
            try:
                print(f"Loading FAISS index from {idx_path} ({self.index_source})...")
                self.index = faiss.read_index(idx_path)
                print(f"Loading metadata from {meta_csv} ...")
                self.df = pd.read_csv(meta_csv)

                if 'text' not in self.df.columns:
                    q_col = self.df.columns[self.df.columns.str.lower().str.contains('question')].tolist()
                    a_col = self.df.columns[self.df.columns.str.lower().str.contains('answer')].tolist()
                    q_col = q_col[0] if q_col else None
                    a_col = a_col[0] if a_col else None

                    if q_col and a_col:
                        self.df['text'] = "Question: " + self.df[q_col].astype(str) + " Answer: " + self.df[a_col].astype(str)
                    else:
                        self.df['text'] = self.df.astype(str).agg(' '.join, axis=1)

                try:
                    idx_d = getattr(self.index, 'd', None)
                    if idx_d and (self.embedding_dim is None):
                        self.embedding_dim = int(idx_d)
                except Exception:
                    pass

                print(f"RAG resources loaded. Total documents: {len(self.df)}")
            except Exception as e:
                print(f"Failed to load index/data: {e}")
                self.index = None
                self.df = None
        else:
            print("No FAISS index or metadata CSV found. Retrieval disabled.")
            if self.use_merged_index:
                print(f"Checked merged: {merged_idx_path}, {merged_meta_csv}")
            print(f"Checked default: {default_idx_path}, {default_meta_csv}")

    def is_ready(self) -> bool:
        """Return True if index and metadata are loaded."""
        return (self.index is not None) and (self.df is not None) and len(self.df) > 0

    def switch_to_merged_index(self) -> bool:
        """Switch to merged MIMIC-IV + MEDQUAD index if available."""
        merged_idx = "data/medquad_faiss_merged.idx"
        merged_meta = "data/medquad_faiss_merged_meta.csv"
        
        if not os.path.exists(merged_idx) or not os.path.exists(merged_meta):
            print("[WARN] Merged index not found")
            return False
        
        try:
            self.index = faiss.read_index(merged_idx)
            self.df = pd.read_csv(merged_meta)
            self.use_merged_index = True
            self.index_source = "merged (MEDQUAD + MIMIC-IV)"
            print(f"[OK] Switched to merged index. Total docs: {len(self.df)}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to switch to merged index: {e}")
            return False

    def switch_to_default_index(self) -> bool:
        """Switch back to default MEDQUAD-only index."""
        default_idx = "data/medquad_faiss.idx"
        default_meta = "data/medquad_faiss_meta.csv"
        
        if not os.path.exists(default_idx) or not os.path.exists(default_meta):
            print("[WARN] Default index not found")
            return False
        
        try:
            self.index = faiss.read_index(default_idx)
            self.df = pd.read_csv(default_meta)
            self.use_merged_index = False
            self.index_source = "default (MEDQUAD only)"
            print(f"[OK] Switched to default index. Total docs: {len(self.df)}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to switch to default index: {e}")
            return False

    def get_stats(self) -> Dict[str, Optional[int]]:
        """Return basic stats for tests and logging."""
        total_docs = 0
        if self.index is not None and hasattr(self.index, "ntotal"):
            try:
                total_docs = int(self.index.ntotal)
            except Exception:
                total_docs = len(self.df) if self.df is not None else 0
        elif self.df is not None:
            total_docs = len(self.df)

        vector_dim = self.embedding_dim if self.embedding_dim is not None else (
            getattr(self.index, 'd', None) if self.index is not None else None
        )

        return {
            "total_documents": int(total_docs),
            "vector_dimension": int(vector_dim) if vector_dim is not None else None
        }

    def _prepare_query_embedding(self, query: str) -> np.ndarray:
        emb = self.model.encode([query], convert_to_tensor=False)
        emb = np.array(emb, dtype='float32')
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        try:
            metric = getattr(self.index, 'metric_type', None)
            if metric is None and self.index is not None:
                if 'IndexFlatIP' in type(self.index).__name__ or 'IndexIVFFlat' in type(self.index).__name__:
                    metric = faiss.METRIC_INNER_PRODUCT
        except Exception:
            metric = None

        if metric == faiss.METRIC_INNER_PRODUCT:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb = emb / norms

        return emb

    def retrieve(self, query: str, k: int = 3, debug: bool = False) -> List[str]:
        """
        Retrieve top-k documents for the given query.
        If debug=True, prints detailed FAISS/index info (index class, ntotal, d, returned D and I).
        """
        if not query:
            if debug: print("Empty query passed to retrieve()")
            return []

        if not self.is_ready():
            if debug: print("RAGRetriever not ready: index or metadata missing.")
            return []

        try:
            ntotal = self.get_stats().get("total_documents", 0)
            if ntotal == 0:
                if debug: print("Index has zero documents.")
                return []

            kk = min(k, ntotal)
            q_emb = self._prepare_query_embedding(query)

            if debug:
                print("DEBUG: Index class:", type(self.index))
                print("DEBUG: ntotal:", ntotal, "requested k:", k, "using k:", kk)
                print("DEBUG: query embedding shape:", q_emb.shape, "dtype:", q_emb.dtype)
                try:
                    print("DEBUG: index.ntotal:", getattr(self.index, 'ntotal', None))
                    print("DEBUG: index.d:", getattr(self.index, 'd', None))
                    print("DEBUG: index metric:", getattr(self.index, 'metric_type', None))
                except Exception:
                    pass

            D, I = self.index.search(q_emb, kk)

            if debug:
                print("DEBUG: D (distances):", D)
                print("DEBUG: I (indices):", I)

            results = []
            for idx in I[0]:
                if idx != -1 and self.df is not None and idx < len(self.df):
                    results.append(str(self.df.iloc[int(idx)]['text']))
                else:
                    if debug:
                        print("DEBUG: skipping index", idx, "len(df)=", len(self.df) if self.df is not None else None)

            if debug: print(f"Retrieved {len(results)} docs for query: {query!r}")
            return results

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

    def search_by_symptoms(self, symptoms: List[str], k: int = 5) -> List[str]:
        """Convenience method used by tests: combine symptoms and search. Increased k for better context."""
        if not symptoms:
            return []
        symptom_query = " ".join(symptoms)
        return self.retrieve(symptom_query, k=k)

    def find_metadata_rows(self, text_snippet: str) -> Optional[pd.DataFrame]:
        """Return DataFrame rows matching the snippet (case-insensitive)."""
        if self.df is None:
            return None
        return self.df[self.df['text'].str.contains(text_snippet, case=False, na=False)]
