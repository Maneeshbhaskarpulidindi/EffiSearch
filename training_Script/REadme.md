What ICT really does

ICT (Inverse Cloze Task) was first proposed by Facebook AI Research (from REALM / ICT-BERT papers).
It’s a self-supervised pretraining method for dense retrievers.


⚙️ How ICT works conceptually

Given a single passage (say, a paragraph):
Sample one sentence from it → treat it as the query.
Use the rest of the passage as the positive document.
Randomly sample other passages from the corpus → treat them as negatives.
(query_sentence, positive_passage, random_negatives)



| Concept              | True ICT                     | Your Code                             |
| -------------------- | ---------------------------- | ------------------------------------- |
| **Data Source**      | Unlabeled corpus (Wikipedia) | MS MARCO (supervised retrieval data)  |
| **Query Creation**   | Sentence from same passage   | ✅ Same                                |
| **Positive Passage** | Remaining sentences          | ✅ Same                                |
| **Encoder Weights**  | Shared                       | ❌ Separate                            |
| **Negatives**        | In-batch only                | Explicit + in-batch                   |
| **Loss**             | InfoNCE (no explicit neg)    | Modified contrastive                  |
| **Purpose**          | Self-supervised pretraining  | Semi-supervised retriever pretraining |
| **Closer To**        | REALM / ICT                  | DPR with ICT-style positives          |
