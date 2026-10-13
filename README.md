# Two-Tower Semantic Search from Scratch

[![Kaggle competition](https://img.shields.io/badge/Kaggle-Learning%20Equality-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations)
[![Course on Udemy](https://img.shields.io/badge/Full%20course-Udemy-A435F0?logo=udemy&logoColor=white)](https://www.udemy.com/course/master-transformers-from-scratch-bert-vit-ai-search/)

Build a complete semantic search system (retrieve, then re-rank) for the Kaggle
[Learning Equality: Curriculum Recommendations](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations) competition:
given a curriculum **topic**, find the **content** items that belong with it.

No `sentence-transformers`. Both towers are built by hand from a plain transformer encoder, so every step is visible.

## The idea

```
            OFFLINE                                   ONLINE
content ──▶ content tower ──▶ vector ─┐
content ──▶ content tower ──▶ vector ─┤  index    topic ──▶ topic tower ──▶ vector
content ──▶ content tower ──▶ vector ─┘    │                                  │
                                           └──── similarity (one matmul) ◀────┘
                                                        │
                                                   top-50 shortlist
                                                        │
                                       cross-encoder re-ranks [CLS] topic [SEP] content [SEP]
                                                        │
                                                      top-5
```

- **Two towers** turn topics and content into vectors in the same space. Content is encoded once, ahead of time.
- **Retrieval** is a single matrix multiply against the precomputed index, so it's fast.
- **A cross-encoder** reads topic and content *together* with full attention. It's slower but more accurate, so it only re-ranks the shortlist.

## What the notebook does

| Step | What happens |
|---|---|
| 1 | Setup and constants (`DEMO_MODE` keeps the run short) |
| 2 | Load the competition CSVs |
| 3 | Build one text string per topic and per content item |
| 4 | Build labeled `(topic, content, label)` pairs |
| 5 | Tokenizer, Dataset and collate function |
| 6 | The `TwoTower` model: two encoders, mean pooling, L2 normalization |
| 7 | Three losses: contrastive, triplet and in-batch cross-entropy |
| 8 | Train the towers |
| 9 | Encode the content library into an index |
| 10 | Retrieve top-k per topic and measure recall@k |
| 11 | Cross-encoder over `[CLS] topic [SEP] content [SEP]` |
| 12 | Re-rank the shortlist and compare precision@5 |
| 13 | Write `submission.csv` |

## Run it on Kaggle

Upload [`two_tower_semantic_search.ipynb`](two_tower_semantic_search.ipynb) to a Kaggle notebook:

1. Open the [competition](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations) and accept the rules.
2. From the **Code** tab, create a notebook and import this file (**File → Import Notebook**).
3. Turn on a **GPU** and **Internet** so the pretrained weights can download.
4. **Run All**. Set `DEMO_MODE = False` for a full run.

## Ideas to go further

- Mine hard negatives instead of random ones
- Share weights between the two towers and compare
- Try a different base encoder

## Learn the full transformer

This is Session 16 of **[Master Transformers from Scratch: BERT, ViT & AI Search](https://www.udemy.com/course/master-transformers-from-scratch-bert-vit-ai-search/)**.
The course builds up to this from the math of attention through BERT, so you understand what each tower is doing inside.

More from the course:
- [transformer-encoder-pytorch](https://github.com/seyedvahidmirnezami/transformer-encoder-pytorch): the full encoder block, and why positional encoding matters
- [self-attention-from-scratch](https://github.com/seyedvahidmirnezami/self-attention-from-scratch): attention with real numbers, in NumPy and PyTorch
- [vision-transformer-cassava](https://github.com/seyedvahidmirnezami/vision-transformer-cassava): fine-tune a Vision Transformer on Kaggle

If this helped you, a ⭐ helps other learners find it.

## License

[MIT](LICENSE). Competition data is subject to the Kaggle competition rules.
