# WikiDT Phase 7: OCR Embedding Storage and Integration with Other QA Models

**Implementation reference:** `WikiDT_Phase7_Modular_OCR_Embeddings_PlugAndPlay_QA.ipynb`  
**Status:** Architecture and code documentation; this document does **not** claim a measured Phase 7 accuracy improvement.

## 1. Purpose

Phase 7 separates **OCR extraction and passage representation** from the **downstream question-answering (QA) reader**. Amazon Textract output is supplied by the WikiDT dataset, not rerun during this notebook. The notebook reconstructs readable text from Textract JSON, splits it into passages, generates reusable MiniLM embeddings, stores both vectors and their source text, and retrieves relevant evidence for a swappable QA model.

**Critical distinction:** Textract produces text and positional annotations; **MiniLM** produces the stored embeddings. A BERT extractive QA reader consumes the **question plus retrieved passage text**, not the MiniLM vectors as BERT `inputs_embeds`.

```text
WikiDT screenshot (processed upstream)
        |
        v
Precomputed Amazon Textract JSON (text + bounding boxes)
        |
        v
OCR parser / approximate row grouping
        |
        +------------------> OCR text cache (Parquet)
        |
        v
140-word passages, step 95 words (45-word overlap)
        |
        +------------------> Passage index (Parquet: source + text + row ID)
        |
        v
all-MiniLM-L6-v2 semantic encoder
        |
        +------------------> Passage vectors (NumPy float32, N x 384)
        |
        +------------------> Embedding manifest (JSON)
        |
Question --> MiniLM query vector + TF-IDF query representation
        |
        v
Hybrid semantic / lexical ranking --> top 6 passage texts
        |
        +--> BERT extractive QA
        +--> Other extractive QA checkpoint
        +--> Generative QA / LLM (custom adapter)
        +--> Classifier / clustering / similarity using vectors
```

## 2. Where artifacts are stored

The notebook uses Google Drive when mounted, otherwise `/mnt/data/WikiDT_Persistent`:

```text
ROOT/
  phase7_modular_ocr_embeddings/
    semantic_hybrid_v1/
      pilot_300_60_60/                       # RUN
        ocr_text_cache_layout.parquet
        embeddings/
          <16-character-cache-key>/         # embed_dir
            ocr_passage_vectors.npy
            ocr_passage_index.parquet
            ocr_embedding_manifest.json
        retrieval_audit.parquet
        best_reader_qa/                     # Hugging Face QA model + tokenizer
          config.json
          model.safetensors (or equivalent)
          ...
          phase7_signature.json
        training_history.csv
        val_predictions.parquet
        test_predictions.parquet
        phase7_metrics.csv
```

In Colab, `ROOT` is normally `/content/drive/MyDrive/WikiDT_Persistent`. The `<cache-key>` directory name is computed from the embedding configuration and passage content; **do not hard-code it**. In the notebook, `embed_dir` is the authoritative location.

### Artifact contract

| Artifact | Contents | Producer | Consumer |
|---|---|---|---|
| `ocr_text_cache_layout.parquet` | `ocr_path`, `text` | Textract parser | Passage chunker or another text pipeline |
| `ocr_passage_vectors.npy` | NumPy `float32` matrix, shape **`[N, 384]`** | `sentence-transformers/all-MiniLM-L6-v2` | Semantic search, classifiers, clustering, reranking |
| `ocr_passage_index.parquet` | `ocr_path`, `chunk_id`, `context`, `row_id` | Passage chunker | Maps a vector back to its source OCR file and original passage text |
| `ocr_embedding_manifest.json` | `encoder`, `hash`, `chunk_words`, `chunk_step`, `layout`, `normalized` | Embedding export | Cache identity and reproducibility |
| `retrieval_audit.parquet` | Question, OCR file references, ranked evidence JSON, answer-coverage diagnostics | Retriever | Analysis and debugging |
| `best_reader_qa/` | Fine-tuned extractive QA reader and tokenizer | QA training | BERT or compatible Hugging Face inference |
| `phase7_metrics.csv` | Validation/test EM, F1, OCR/retrieval coverage | Evaluation | Phase comparison |

**Alignment guarantee:** row `i` of `ocr_passage_vectors.npy` corresponds to row `i` of `ocr_passage_index.parquet` **in its saved row order**. The `row_id` field records the original sequential index. Do not sort or filter the index independently of the vector matrix.

## 3. How OCR text becomes a vector

1. Read the OCR JSON files referenced by each WikiDT sample's `all_ocr_files_textract` metadata.
2. Extract text. For the compact WikiDT annotation format, use bounding-box `x`/`y` positions to group approximately horizontal rows; for standard Textract `Blocks`, prefer `LINE` text, falling back to `WORD` text. The layout grouping is **heuristic**, not a guaranteed table-cell reconstruction.
3. Save extracted text in `ocr_text_cache_layout.parquet` for reuse.
4. Split each OCR text into passages of **140 words**, advancing **95 words** per passage.
5. Encode each passage independently with `sentence-transformers/all-MiniLM-L6-v2`.
6. L2-normalize each embedding, cast to `float32`, and save the vector matrix and aligned passage index.

A passage might contain 140 words but is represented by **one 384-dimensional MiniLM vector**. This is a *passage-level retrieval representation*, not BERT's per-token `384 x 768` hidden-state matrix. These two matrices serve different purposes.

## 4. Loading stored OCR embeddings without running OCR or BERT

The following standalone Python snippet assumes that Phase 7's embedding-export cell has already run and that you know the generated `embed_dir` path:

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

embed_dir = Path('/content/drive/MyDrive/WikiDT_Persistent/'
                 'phase7_modular_ocr_embeddings/semantic_hybrid_v1/'
                 'pilot_300_60_60/embeddings/<cache-key>')

vectors = np.load(embed_dir / 'ocr_passage_vectors.npy')
index = pd.read_parquet(embed_dir / 'ocr_passage_index.parquet')
manifest = json.loads((embed_dir / 'ocr_embedding_manifest.json').read_text())

assert vectors.dtype == np.float32
assert vectors.shape[0] == len(index)
assert vectors.shape[1] == 384
assert (index['row_id'].to_numpy() == np.arange(len(index))).all()

print('Embeddings:', vectors.shape)
print('Encoder:', manifest['encoder'])
print(index[['row_id', 'ocr_path', 'chunk_id', 'context']].head())

# Passage row i and embedding row i always refer to the same passage.
i = 0
passage_text = index.iloc[i]['context']
passage_embedding = vectors[i]
```

Replace `<cache-key>` with the actual directory name printed by the notebook. **Loading these files does not require the original OCR archive or the BERT reader checkpoint.**

## 5. How the question selects passages

Phase 7 performs **question-only retrieval**, without looking at the ground-truth answer:

1. Restrict candidate passages to OCR files associated with the current WikiDT sample.
2. Encode the question using the **same MiniLM encoder** used for passage vectors.
3. Compute semantic similarity using the dot product of L2-normalized vectors (equivalent to cosine similarity).
4. Independently calculate TF-IDF cosine similarity between the question and candidate passage texts.
5. Min-max normalize semantic and lexical scores **within the candidate set**.
6. Combine them as `0.55 × normalized semantic + 0.45 × normalized lexical`.
7. Return the **top 6** passages with their text and scores.

The notebook's `retrieve(question, ocr_paths, top_k=6)` returns a list of dictionaries with these exact keys:

```python
{
    'path': 'source/ocr/file.json',
    'context': 'recognized passage text ...',
    'score': 0.83,            # example value; fused ranking score
    'semantic_score': 0.76,   # example value; raw semantic cosine
    'lexical_score': 0.41     # example value; raw TF-IDF cosine
}
```

The example values above are illustrative. Score magnitudes are not calibrated answer probabilities.

## 6. Connecting another extractive QA model

A compatible Hugging Face extractive QA checkpoint can replace the Phase 7 reader. The easiest approach is to retain the existing `get_retrieved_evidence()` function and pass its returned **passage text** to the new reader.

```python
from transformers import pipeline

# Execute inside the Phase 7 notebook, after the retrieval cells.
reader = pipeline(
    'question-answering',
    model='deepset/bert-base-cased-squad2',
    tokenizer='deepset/bert-base-cased-squad2',
    device=0 if DEVICE == 'cuda' else -1,
)

question = 'Which team plays at Glenferrie Oval?'
ocr_paths = example_row.ocr_files  # source files belonging to this WikiDT sample
hits = get_retrieved_evidence(question, ocr_paths, k=6)

answers = []
for hit in hits:
    result = reader(question=question, context=hit['context'])
    answers.append({
        'answer': result['answer'],
        'reader_score': float(result['score']),
        'retrieval_score': hit['score'],
        'ocr_path': hit['path'],
    })
print(answers)
```

**Important:** This example is an integration demonstration, not the notebook's exact multi-window prediction algorithm. The pipeline may truncate long contexts or apply different no-answer handling; its scores are not necessarily directly comparable across passages. For rigorous evaluation, implement sliding-window tokenization and consistent span/no-answer scoring, as in the notebook's custom reader.

To swap extractive checkpoints within the notebook itself, change `MODEL_ID` to another compatible `AutoModelForQuestionAnswering` checkpoint, then retrain/evaluate. The notebook saves the selected reader separately from the OCR embedding artifacts.

## 7. Connecting a generative QA model or LLM

Generative models consume a **question and retrieved passage text** in a prompt; they do not directly understand MiniLM's 384-dimensional vectors without a separately trained adapter.

```python
question = 'Which team plays at Glenferrie Oval?'
hits = get_retrieved_evidence(question, example_row.ocr_files, k=6)
context = '\n\n'.join(f'Passage {i+1}: {h["context"]}' for i, h in enumerate(hits))
prompt = (
    'Answer using only the supplied OCR passages. '
    'If the answer is missing, say "Not found".\n\n'
    f'Context:\n{context}\n\nQuestion: {question}\nAnswer:'
)
# Send `prompt` to your chosen generative model using its own tokenizer/API.
```

For long retrieved passages, apply the target model's context-length and prompt-size limits. Generative answers may paraphrase or hallucinate, so evaluate against the same held-out WikiDT questions.

## 8. Using the vectors directly in other models

The saved `N x 384` vectors can be inputs to vector-native tasks:

- **Nearest-neighbor retrieval:** compare a MiniLM-encoded question with the saved vectors.
- **Classification:** train a classifier with each 384-dimensional passage vector as its feature input, using task-specific labels.
- **Clustering:** group similar OCR passages.
- **Reranking:** use vector similarity as one feature in a learned reranker.

However, **you cannot feed these vectors directly into BERT's `inputs_embeds`**: BERT-base expects token-level representations in a different learned embedding space (hidden size 768). Even a linear projection from 384 to 768 would need task-specific training and would not automatically recover the original token sequence required for span extraction.

If the goal is *one embedding output that any downstream model can consume*, define an **adapter contract** for each model: vector-native models take `vectors`; text-based extractive or generative QA models take `context` and `question`; multimodal architectures need an explicitly trained cross-modal adapter.

## 9. Portable interface and recommended API

The current notebook already defines:

```python
get_ocr_embeddings(ocr_paths)
# Returns (passage_vectors_subset, passage_df_subset)
# passage_df_subset includes ocr_path, chunk_id, context, row_id.

get_retrieved_evidence(question, ocr_paths, k=TOP_K)
# Returns list of {path, context, score, semantic_score, lexical_score}.
```

For use outside the notebook, package these functions with their dependencies (MiniLM encoder, TF-IDF ranking logic, passage index, vector matrix, and path-to-row mapping) in a Python module. A portable API should expose:

```python
class OCRStore:
    def get_passages(self, ocr_paths): ...  # metadata + text
    def get_vectors(self, ocr_paths): ...   # float32 [M, 384]
    def retrieve(self, question, ocr_paths, top_k=6): ...

class QAReader:
    def answer(self, question, evidence): ...
```

These classes are a **recommended interface design**, not classes already implemented by the Phase 7 notebook. Save the OCR store and QA reader independently so retraining a reader does not regenerate OCR embeddings.

## 10. Reproducibility, limits, and evaluation

- **Cache integrity:** the manifest records the embedding model, text hash, chunk parameters, layout flag, and normalization setting; the notebook reloads vectors only when the manifest matches. Preserve the index and vectors together.
- **Model compatibility:** a new embedding model requires regenerating both passage vectors and question embeddings. Never compare vectors from different embedding spaces directly.
- **Table layout:** approximate row grouping can help but does not guarantee accurate row/column semantics; a true table-cell reconstruction is a possible future improvement.
- **Retrieval scope:** retrieval searches only OCR paths associated with the sample, not an unrestricted Wikipedia corpus.
- **Evaluation:** report OCR answer coverage, top-6 retrieval coverage, reader Exact Match, token F1, and conditional EM. Use the same held-out questions when comparing reader architectures.
- **No guaranteed gain:** Phase 7 is a new architecture; performance improvement over Phase 6 must be established by executing the notebook and examining the resulting metrics.

## 11. Short summary for a paper

> Phase 7 decouples OCR representation from question answering by converting precomputed WikiDT Amazon Textract annotations into passage text and independent, L2-normalized 384-dimensional MiniLM embeddings. Each embedding is persisted in a NumPy matrix with an aligned Parquet index that retains the source OCR file and passage text. A hybrid semantic–lexical retriever selects question-relevant evidence, which can be consumed by interchangeable extractive or generative QA readers. The embeddings are reusable for retrieval and other vector-based tasks, while text-based readers continue to consume the original passage text. This design allows QA model changes without rerunning OCR or recomputing passage embeddings, provided the OCR corpus and embedding configuration remain unchanged.
