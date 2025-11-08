#!/usr/bin/env python3
"""
MIMIC-IV FAISS Integration Script

This script:
1. Loads prepared MIMIC-IV clinical notes
2. Generates embeddings using SentenceTransformer
3. Creates/updates FAISS indices
4. Merges MIMIC-IV with existing MEDQUAD data
5. Saves updated indices and metadata
"""

import os
import sys
import json
import pickle
import gzip
import csv
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

print("[*] Loading libraries...")

try:
    from sentence_transformers import SentenceTransformer
    print("    [OK] sentence_transformers loaded")
except Exception as e:
    print(f"    [ERROR] Failed to load sentence_transformers: {e}")
    sys.exit(1)

try:
    import faiss
    print("    [OK] faiss loaded")
except Exception as e:
    print(f"    [ERROR] Failed to load faiss: {e}")
    sys.exit(1)


def load_prepared_mimic_data(json_path: str = "data/mimic_iv_prepared.json") -> Dict:
    """Load prepared MIMIC-IV data from JSON."""
    if not os.path.exists(json_path):
        print(f"[WARN] Prepared MIMIC data not found: {json_path}")
        return {"documents": [], "metadata": []}
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"[OK] Loaded {len(data.get('documents', []))} MIMIC-IV document chunks")
    return data


def extract_additional_mimic_notes(limit: int = 500) -> Tuple[List[str], List[Dict]]:
    """
    Extract additional MIMIC-IV notes directly from CSV files for large-scale indexing.
    """
    print("\n[*] Extracting additional MIMIC-IV notes...")
    
    mimic_path = Path("data/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note")
    if not mimic_path.exists():
        print(f"[WARN] MIMIC-IV path not found")
        return [], []
    
    documents = []
    metadata = []
    
    gz_files = list(mimic_path.glob("*.csv.gz"))
    print(f"    Found {len(gz_files)} MIMIC-IV files")
    
    for gz_file in gz_files:
        if len(documents) >= limit:
            break
        
        print(f"    Processing {gz_file.name}...")
        
        try:
            with gzip.open(gz_file, 'rt', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                count = 0
                
                for row in reader:
                    if len(documents) >= limit or count >= 200:
                        break
                    
                    # Extract text
                    text = row.get('text', '')
                    if not text or len(str(text).strip()) < 50:
                        continue
                    
                    # Chunk if needed
                    words = str(text).split()
                    chunk_size = 300
                    
                    for chunk_idx in range(0, len(words), chunk_size):
                        if len(documents) >= limit:
                            break
                        
                        chunk_words = words[chunk_idx:chunk_idx + chunk_size]
                        if not chunk_words:
                            continue
                        
                        chunk_text = " ".join(chunk_words)
                        documents.append(chunk_text)
                        
                        metadata.append({
                            "source": "mimic-iv",
                            "file": gz_file.name,
                            "note_id": row.get('note_id', 'unknown'),
                            "note_type": row.get('note_type', 'unknown'),
                            "chunk": chunk_idx // chunk_size
                        })
                    
                    count += 1
        
        except Exception as e:
            print(f"        [WARN] Error processing {gz_file.name}: {e}")
    
    print(f"    [OK] Extracted {len(documents)} MIMIC-IV chunks")
    return documents, metadata


def load_existing_medquad_data(csv_path: str = "data/medquad_faiss_meta.csv") -> Tuple[List[str], List[Dict]]:
    """Load existing MEDQUAD data for merging."""
    if not os.path.exists(csv_path):
        print(f"[WARN] MEDQUAD metadata not found: {csv_path}")
        return [], []
    
    df = pd.read_csv(csv_path)
    print(f"[OK] Loaded {len(df)} MEDQUAD documents")
    
    documents = []
    metadata = []
    
    for idx, row in df.iterrows():
        text = row.get('text', '')
        if text:
            documents.append(str(text))
            metadata.append({
                "source": "medquad",
                "index": idx,
            })
    
    return documents, metadata


def generate_embeddings(documents: List[str], model_name: str = "paraphrase-MiniLM-L3-v2", batch_size: int = 32) -> np.ndarray:
    """Generate embeddings for documents using SentenceTransformer."""
    print(f"\n[*] Generating embeddings for {len(documents)} documents...")
    
    model = SentenceTransformer(model_name)
    print(f"    Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    embeddings = model.encode(documents, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=False)
    embeddings = np.array(embeddings, dtype='float32')
    
    print(f"    [OK] Generated embeddings shape: {embeddings.shape}")
    return embeddings


def create_merged_index(mimic_docs: List[str], mimic_meta: List[Dict], 
                       medquad_docs: List[str], medquad_meta: List[Dict]) -> Tuple[faiss.Index, pd.DataFrame]:
    """Create merged FAISS index with both MIMIC-IV and MEDQUAD data."""
    print("\n[*] Creating merged FAISS index...")
    
    # Combine documents and metadata
    all_documents = medquad_docs + mimic_docs
    all_metadata = medquad_meta + mimic_meta
    
    print(f"    Total documents: {len(all_documents)}")
    print(f"    - MEDQUAD: {len(medquad_docs)}")
    print(f"    - MIMIC-IV: {len(mimic_docs)}")
    
    # Generate embeddings
    embeddings = generate_embeddings(all_documents)
    
    # Create FAISS index
    print("\n    Creating FAISS index...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    
    # Normalize embeddings for inner product search
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index.add(embeddings_norm)
    
    print(f"    [OK] FAISS index created with {index.ntotal} vectors")
    
    # Create metadata dataframe
    df_meta = pd.DataFrame(all_metadata)
    df_meta['text'] = all_documents
    
    return index, df_meta


def save_index_and_metadata(index: faiss.Index, df_meta: pd.DataFrame, 
                           index_path: str = "data/medquad_faiss_merged.idx",
                           meta_path: str = "data/medquad_faiss_merged_meta.csv"):
    """Save FAISS index and metadata to disk."""
    print(f"\n[*] Saving index and metadata...")
    
    # Save index
    faiss.write_index(index, index_path)
    print(f"    [OK] Saved FAISS index to {index_path} ({os.path.getsize(index_path) / 1024 / 1024:.1f} MB)")
    
    # Save metadata
    df_meta.to_csv(meta_path, index=False)
    print(f"    [OK] Saved metadata to {meta_path} ({os.path.getsize(meta_path) / 1024:.1f} KB)")
    
    return index_path, meta_path


def update_rag_retriever_config(index_path: str, meta_path: str):
    """Create configuration file for updated RAG retriever."""
    config = {
        "default_index": "data/medquad_faiss.idx",
        "default_metadata": "data/medquad_faiss_meta.csv",
        "merged_index": index_path,
        "merged_metadata": meta_path,
        "description": "Merged index containing MEDQUAD + MIMIC-IV clinical notes",
        "switching_enabled": True,
        "switch_command": "use_merged_index()"
    }
    
    config_path = "data/rag_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n[OK] Created RAG config: {config_path}")
    return config_path


def main():
    """Main integration workflow."""
    print("\n" + "=" * 70)
    print("MIMIC-IV FAISS Integration Pipeline")
    print("=" * 70)
    
    # Step 1: Load MEDQUAD
    print("\n[STEP 1] Load existing MEDQUAD data")
    medquad_docs, medquad_meta = load_existing_medquad_data()
    
    if not medquad_docs:
        print("[WARN] No MEDQUAD data found, will create new index from MIMIC-IV only")
    
    # Step 2: Extract MIMIC-IV data
    print("\n[STEP 2] Extract MIMIC-IV clinical notes")
    mimic_docs, mimic_meta = extract_additional_mimic_notes(limit=1000)
    
    if not mimic_docs:
        print("[ERROR] No MIMIC-IV data extracted")
        return False
    
    # Step 3: Create merged index
    print("\n[STEP 3] Create merged FAISS index")
    index, df_meta = create_merged_index(mimic_docs, mimic_meta, medquad_docs, medquad_meta)
    
    # Step 4: Save index and metadata
    print("\n[STEP 4] Save index and metadata")
    index_path, meta_path = save_index_and_metadata(index, df_meta)
    
    # Step 5: Update configuration
    print("\n[STEP 5] Update RAG configuration")
    update_rag_retriever_config(index_path, meta_path)
    
    print("\n" + "=" * 70)
    print("INTEGRATION COMPLETE")
    print("=" * 70)
    print(f"Merged index: {index_path}")
    print(f"Metadata: {meta_path}")
    print(f"Total documents: {index.ntotal}")
    print("\nNext steps:")
    print("1. Update modules/rag_retriever.py to use new indices")
    print("2. Test RAG retrieval with MIMIC-IV data")
    print("3. Run end-to-end diagnosis test")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Integration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
