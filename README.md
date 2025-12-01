# Depression Detection from Clinical Interviews (DAIC-WOZ)

NLP-based depression detection system using the DAIC-WOZ dataset, implementing multiple ML/DL approaches including classical machine learning, transformer models, and ensemble methods.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Pipeline Architecture](#pipeline-architecture)
- [Preprocessing](#preprocessing)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [GCN-Keyword Temporal Heatmaps](#gcn-keyword-temporal-heatmaps)
- [Results & Outputs](#results--outputs)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)

## Overview

This project implements a comprehensive pipeline for binary depression classification from clinical interview transcripts. The system processes the Distress Analysis Interview Corpus - Wizard of Oz (DAIC-WOZ) dataset and applies various feature engineering techniques and classification models to detect depression indicators from conversational text.

### Key Features
- Multi-speaker analysis (Participant and Ellie/Interviewer)
- Multiple feature extraction approaches (TF-IDF, SBERT, behavioral features)
- Classical ML models (SVM, XGBoost) with extensive hyperparameter tuning
- Transformer-based models (BERT, DistilBERT, RoBERTa) with focal loss
- Stratified K-Fold ensemble methods
- GCN-style temporal keyword heatmap visualizations
- Comprehensive evaluation with PNG outputs

## Dataset

**DAIC-WOZ (Distress Analysis Interview Corpus - Wizard of Oz)**

The dataset contains audio, video, and text transcripts of clinical interviews conducted by an animated virtual interviewer named Ellie. Depression labels are derived from PHQ-8 questionnaire scores.

- **Source**: [USC Institute for Creative Technologies](https://dcapswoz.ict.usc.edu/)
- **Access**: Requires license agreement from USC ICT
- **Format**: Tab-separated transcript files with timestamps
- **Labels**: Binary classification (PHQ8_Binary: 0 = non-depressed, 1 = depressed)
- **Splits**: Train, Development, Test (following AVEC 2016/2017 protocol)

### Data Citation
```
Gratch, J., Artstein, R., Lucas, G., Stratou, G., Scherer, S., Nazarian, A., ... & Morency, L. P. (2014). 
The Distress Analysis Interview Corpus of human and computer interviews. 
In Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC'14).
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for transformer models)
- Kaggle account (if running on Kaggle)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/depression-detection-daic-woz.git
cd depression-detection-daic-woz

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Pipeline Architecture
```
Raw DAIC-WOZ Transcripts
         │
         ▼
┌─────────────────────────────────────┐
│     1. TRANSCRIPT SEPARATION        │
│  - Extract per-participant CSVs     │
│  - Convert TSV → CSV format         │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     2. TEXT PREPROCESSING           │
│  - Speaker separation (P/E)         │
│  - Lowercasing, punctuation removal │
│  - Stopword removal, lemmatization  │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     3. FEATURE EXTRACTION           │
│  - TF-IDF (word + char n-grams)     │
│  - SBERT embeddings                 │
│  - Behavioral/linguistic features   │
│  - Sentiment analysis               │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     4. MODEL TRAINING               │
│  - Classical ML (SVM, XGBoost)      │
│  - Transformers (BERT variants)     │
│  - Ensemble methods                 │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     5. EVALUATION & VISUALIZATION   │
│  - Classification reports           │
│  - Confusion matrices, ROC curves   │
│  - GCN-keyword temporal heatmaps    │
└─────────────────────────────────────┘
```

## Preprocessing

### 1. Transcript Separation
- Extracts individual participant transcripts from raw DAIC-WOZ nested directories
- Converts tab-separated format to comma-separated CSV
- Organizes files by participant ID (e.g., `300.csv`, `301.csv`)

### 2. Text Cleaning Pipeline
```python
# Preprocessing steps applied:
1. Lowercasing
2. Punctuation removal (regex-based)
3. Tokenization (NLTK word_tokenize)
4. Stopword removal (English stopwords)
5. Lemmatization (WordNet lemmatizer)
6. Speaker-level text aggregation
```

### 3. Visualization
- Word clouds (before/after preprocessing)
- Word length distribution histograms
- Storage space analysis

## Feature Engineering

### TF-IDF Features
| Parameter | Word TF-IDF | Character TF-IDF |
|-----------|-------------|------------------|
| Max Features | 5,000-8,000 | 2,000-3,000 |
| N-gram Range | (1,2) or (1,3) | (3,5) |
| Analyzer | word | char_wb |
| Min DF | 2 | - |

Feature selection using Chi-squared test (top 3,000-4,000 features).

### Semantic Embeddings (SBERT)
- **Model**: `all-MiniLM-L6-v2`
- **Dimensions**: 384 → reduced to 200 via TruncatedSVD
- **Purpose**: Capture semantic meaning beyond bag-of-words

### Behavioral/Linguistic Features
| Feature | Description | Source |
|---------|-------------|--------|
| Polarity | Sentiment score (-1 to 1) | TextBlob |
| Subjectivity | Opinion vs fact (0 to 1) | TextBlob |
| Word Count | Total words in transcript | Text |
| Char Count | Total characters | Text |
| Avg Word Length | Mean word length | Text |
| Speech Speed | Words per second | Timestamps |
| Avg Response Time | Mean utterance duration | Timestamps |
| Adjective Frequency | Adjective ratio | Text patterns |

### Combined Feature Vector
```
Final Features = [TF-IDF (SVD-reduced)] + [SBERT embeddings] + [Behavioral features]
Typical dimensions: ~200 (TF-IDF) + 200 (SBERT) + 4-5 (behavioral) ≈ 405 features
```

## Models

### 1. SVM (Pure TF-IDF)
- **Kernel**: RBF
- **Optimization**: Two-stage grid search (coarse → refined)
- **Class Balancing**: Minority upsampling + balanced class weights
- **Calibration**: Platt scaling (sigmoid)
- **Features**: TF-IDF + sentiment/linguistic extras

### 2. SVM + SBERT
- Same architecture as pure TF-IDF SVM
- **Additional Features**: SBERT sentence embeddings
- **Combined Dimensions**: ~3,400 features

### 3. XGBoost
- **Optimization**: RandomizedSearchCV (60 iterations)
- **Early Stopping**: 25-30 rounds on validation set
- **Key Hyperparameters**:
  - `n_estimators`: [300, 500, 800, 1200]
  - `max_depth`: [3, 15]
  - `learning_rate`: [1e-3, 0.2] (log-uniform)
  - `scale_pos_weight`: Computed from class ratio

### 4. BERT with Focal Loss
- **Base Model**: `bert-base-uncased`
- **Loss Function**: Focal loss (γ=2.0) for class imbalance
- **Training**:
  - Epochs: 30
  - Batch Size: 8 (effective 16 with accumulation)
  - Learning Rate: 2e-5
  - Max Sequence Length: 512
  - Warmup: 6% of total steps
- **Sampling**: WeightedRandomSampler for balanced batches

### 5. Stratified K-Fold Ensemble
- **Models**: DistilBERT, BERT, RoBERTa
- **Strategy**: 5-fold cross-validation per model
- **Aggregation**: Average probability fusion across all folds
- **Epochs per Fold**: 50
- **Checkpointing**: Best fold per model retained

## GCN-Keyword Temporal Heatmaps

Visualizes the temporal distribution of depression-indicative keywords across interview progression.

### Methodology
1. **Word Selection**: TF-IDF differential scoring (depressed - control)
2. **Top-K Selection**: 100 most discriminative words
3. **Position Tracking**: Normalized positions (0-1) of keyword occurrences
4. **Density Estimation**: Kernel Density Estimation (Gaussian, bandwidth=0.05)
5. **Visualization**: 2D heatmap (interview index × temporal position)

### Output
- 2×2 subplot: [Ellie × Depressed/Control] and [Participant × Depressed/Control]
- Word lists with discrimination scores saved as CSV

## Results & Outputs

Each model generates the following artifacts:

| File | Description |
|------|-------------|
| `classification_report.png` | Precision, recall, F1 per class |
| `confusion_matrix.png` | TP/TN/FP/FN visualization |
| `roc_curve.png` | ROC curve with AUC score |
| `learning_curve.png` | Train/val accuracy vs samples |
| `summary.txt` | Hyperparameters and final metrics |
| `*.joblib` | Serialized model and processors |

### Transformer Models Additionally Save:
- `test_predictions.csv` - Predictions with true labels
- `test_metrics.json` - JSON-formatted metrics
- `history.json` - Per-epoch training history
- HuggingFace model checkpoint directory

## Project Structure
```
depression-detection-daic-woz/
├── data/
│   └── DAIC-WOZ/                    # Raw dataset (obtain from USC ICT)
├── transcripts/                      # Processed per-participant CSVs
├── outputs/
│   ├── svm_pure_tfidf_outputs/
│   │   ├── classification_report.png
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   ├── learning_curve.png
│   │   ├── summary.txt
│   │   └── models_and_processors.joblib
│   ├── sbert_tfidf_svm_outputs/
│   ├── xgb_outputs/
│   ├── xgb_enhanced/
│   ├── bert-focal-model/
│   └── ensemble-bert-final/
├── processed_data_main.csv           # Preprocessed text + labels
├── features.npy                      # Extracted feature matrices
├── labels.csv                        # Participant IDs and labels
├── temporal_keyword_heatmap.png      # GCN-style visualization
├── participant_words.csv             # Discriminative words (participant)
├── ellie_words.csv                   # Discriminative words (interviewer)
├── requirements.txt
└── README.md
```

## Usage

### 1. Preprocess Transcripts
```python
# Separate transcripts by participant
transcripts = separate_transcripts(input_dir='path/to/daicwoz')
print(f'Separated {len(transcripts)} transcripts')

# Clean and preprocess text
for csv in os.listdir(output_dir):
    # Process each participant...
    cleaned_text = preprocess_text(participant_text)
```

### 2. Extract Features
```python
# TF-IDF features
word_vect = TfidfVectorizer(max_features=6000, ngram_range=(1,2))
char_vect = TfidfVectorizer(max_features=2000, ngram_range=(3,5), analyzer='char_wb')

# SBERT embeddings
from sentence_transformers import SentenceTransformer
sbert = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = sbert.encode(texts)

# Combine features
X_combined = np.hstack([tfidf_reduced, sbert_reduced, behavioral_features])
```

### 3. Train Models
```python
# SVM with two-stage grid search
grid1 = GridSearchCV(svc, param_grid_stage1, cv=5, scoring='f1')
grid1.fit(X_train, y_train)
# ... refine around best params

# XGBoost with RandomizedSearch
rnd = RandomizedSearchCV(xgb, param_dist, n_iter=60, scoring='f1')
rnd.fit(X_train, y_train)

# BERT with focal loss
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
# ... training loop with focal_loss()
```

### 4. Generate Heatmaps
```python
# Load interviews by speaker
dep_interviews, ctrl_interviews, _, _ = load_interviews_by_speaker('Participant')

# Get discriminative words
keywords, scores = get_discriminative_words(dep_interviews, ctrl_interviews, top_k=100)

# Create temporal heatmap
create_heatmap(ax, positions, title, ylabel, kde_bw=0.05)
```

## Evaluation Metrics

| Metric | Description | Usage |
|--------|-------------|-------|
| **F1-Score** | Harmonic mean of precision/recall | Primary metric (handles imbalance) |
| **Accuracy** | Overall correct predictions | Secondary metric |
| **Precision** | TP / (TP + FP) | Measures false positive rate |
| **Recall** | TP / (TP + FN) | Measures false negative rate |
| **ROC-AUC** | Area under ROC curve | Threshold-independent performance |

### Class Imbalance Handling
- **Upsampling**: Minority class resampled to match majority
- **WeightedRandomSampler**: Balanced batch composition
- **Focal Loss**: Down-weights easy examples (γ=2.0)
- **scale_pos_weight**: XGBoost class weighting
- **class_weight='balanced'**: SVM automatic weighting

## Limitations

1. **Dataset Size**: ~189 participants limits deep learning potential
2. **Binary Labels**: PHQ-8 threshold may not capture depression severity spectrum
3. **Text-Only**: Audio and video modalities not utilized
4. **Interview Setting**: May not generalize to naturalistic speech
5. **Language**: English-only transcripts
6. **Temporal Context**: Turn-level analysis loses some conversational context

<!-- ## Citation

If you use this code, please cite:
```bibtex
@misc{depression-detection-daic-woz,
  author = {Your Name},
  title = {Depression Detection from Clinical Interviews using NLP},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/depression-detection-daic-woz}
}
``` -->

### Dataset Citation
```bibtex
@inproceedings{gratch2014distress,
  title={The distress analysis interview corpus of human and computer interviews},
  author={Gratch, Jonathan and Artstein, Ron and Lucas, Gale and Stratou, Giota and Scherer, Stefan and Nazarian, Angela and Wood, Rachel and Boberg, Jill and DeVault, David and Marsella, Stacy and others},
  booktitle={Proceedings of LREC},
  year={2014}
}
```

## References

- AVEC 2016/2017 Depression Sub-challenge
- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers
- Lin et al. (2017). Focal Loss for Dense Object Detection
- Reimers & Gurevych (2019). Sentence-BERT

## License

This project is for research purposes only. The DAIC-WOZ dataset requires a separate license agreement from [USC Institute for Creative Technologies](https://dcapswoz.ict.usc.edu/).

---

**Note**: Ensure you have obtained proper authorization to use the DAIC-WOZ dataset before running this code.
```

And here's the `requirements.txt`:
```
# Core Scientific Computing
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Machine Learning
scikit-learn>=1.0.0
xgboost>=1.5.0
joblib>=1.1.0

# Deep Learning
torch>=1.10.0
transformers>=4.20.0
sentence-transformers>=2.2.0

# NLP
nltk>=3.6.0
textblob>=0.17.1

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
wordcloud>=1.8.0

# Utilities
tqdm>=4.62.0

# Optional (for Kaggle/Colab)
# accelerate>=0.12.0
# datasets>=2.0.0
# evaluate>=0.3.0
# sentencepiece>=0.1.96