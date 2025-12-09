# LLM-Powered Recipe Recommender

## Team Members

- **Novikov Ilya** – LLM Lead
- **Ternov Mikhail** – Data & EDA Lead
- **Lepin Mikhail** – UI/UX Lead

## Problem Definition

Many users struggle to find recipes using ingredients they already have. Traditional recommender systems rely on keyword or tag matching and fail to capture semantic meaning (e.g., "tomato + cheese" relates to "pizza" or "pasta").

**RecSys gap with LLMs**: LLMs can understand ingredient combinations and cooking contexts through embeddings and natural-language reasoning, offering explainable recommendations.

**User**: Home cooks or food app users seeking quick, personalized recipe ideas.

**Business Value**: Improves engagement and reduces food waste by suggesting creative recipes based on available ingredients.

**"Why LLM?"**: Unlike classic content-based filters, LLMs provide deeper semantic understanding and generate natural explanations ("This recipe fits because it uses your ingredients and is similar to X cuisine").

## Data & EDA

**Dataset**: Kaggle "Recipe Ingredients Dataset" (recipe title, ingredients, steps, cuisine).

**Explore**: Ingredient frequency, cuisine distribution, and recipe diversity.

**Goal**: Prepare clean, structured data for embedding and similarity comparison.

## Core Feature & UI/UX

### Workflow

```
User enters ingredients → LLM generates embeddings → Similarity search → Rank recipes → Explain choice
```

**UI**: Simple web/Colab interface displaying top-3 recipe cards with names, ingredients, and LLM-generated explanations ("We recommend this because it uses X and Y").

## Milestones (Module-Aligned)

- **Checkpoint 1 (Week 3)**: Data cleaning + embedding pipeline ready.
- **Checkpoint 2 (Week 4)**: Model integrated with basic UI (input + top-3 recommendations).
- **Final (Week 5)**: Full demo, presentation, and final defense submission (PDF).

## Modeling & Deployment

**Training**: Use Sentence-BERT or OpenAI text-embedding models for semantic similarity.

**Deployment**: Web app hosted on GitHub Pages.

**Architecture**: 
```
Input → Embedding model → Cosine similarity search → Output recipes + explanation
```

## Metrics & Trade-offs

**Primary Metrics**: 
- Cosine similarity accuracy
- User satisfaction survey (qualitative feedback)

**Trade-off**: Speed (LLM inference time) vs. accuracy (semantic depth).

**Evaluation**: Human evaluation on recipe relevance and explanation clarity.

## Risks & Mitigation

**Risk**: LLM hallucinations (inventing recipes or wrong matches).

**Mitigation**: Limit generation to dataset-based results; use structured prompts and similarity thresholds.

