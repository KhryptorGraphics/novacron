"""Text data processing and tokenization for sentiment analysis."""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """Comprehensive text preprocessing pipeline."""
    
    def __init__(self, lowercase: bool = True, remove_stopwords: bool = True,
                 lemmatize: bool = True, remove_punctuation: bool = True):
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.remove_punctuation = remove_punctuation
        
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation if specified
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def preprocess_text(self, text: str) -> str:
        """Full preprocessing pipeline."""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Join back to string
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts."""
        return [self.preprocess_text(text) for text in texts]

class SentimentDataset(Dataset):
    """Dataset class for sentiment analysis."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer,
                 max_length: int = 512, preprocessor: Optional[TextPreprocessor] = None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessor = preprocessor
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Preprocess text if preprocessor is provided
        if self.preprocessor:
            text = self.preprocessor.preprocess_text(text)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class SentimentDataLoader:
    """Data loader for sentiment analysis with BERT tokenization."""
    
    def __init__(self, model_name: str = 'bert-base-uncased', max_length: int = 512,
                 batch_size: int = 16, preprocessor_config: Optional[Dict] = None):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize preprocessor
        if preprocessor_config:
            self.preprocessor = TextPreprocessor(**preprocessor_config)
        else:
            self.preprocessor = TextPreprocessor()
        
        # Label encoder
        self.label_encoder = LabelEncoder()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_imdb_data(self, data_path: str = None) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """Load IMDB movie review dataset."""
        
        if data_path and Path(data_path).exists():
            # Load from provided path
            df = pd.read_csv(data_path)
        else:
            # Create synthetic dataset for demonstration
            self.logger.warning("Creating synthetic dataset for demonstration")
            df = self._create_synthetic_dataset()
        
        # Basic data analysis
        data_stats = {
            'total_samples': len(df),
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'avg_text_length': df['review'].str.len().mean(),
            'max_text_length': df['review'].str.len().max(),
            'min_text_length': df['review'].str.len().min()
        }
        
        return df, data_stats
    
    def _create_synthetic_dataset(self) -> pd.DataFrame:
        """Create synthetic sentiment dataset for demonstration."""
        
        positive_reviews = [
            "This movie was absolutely fantastic! Great acting and storyline.",
            "I loved every minute of it. Highly recommended!",
            "Outstanding performance by all actors. A masterpiece!",
            "Brilliant direction and cinematography. Must watch!",
            "Excellent movie with great character development.",
            "Amazing plot twists and engaging storyline.",
            "Superb acting and beautiful visuals throughout.",
            "One of the best movies I've seen this year!",
            "Incredible emotional depth and powerful performances.",
            "Perfect blend of action and drama. Loved it!"
        ] * 100  # Repeat for larger dataset
        
        negative_reviews = [
            "Terrible movie with poor acting and weak plot.",
            "Complete waste of time. Very disappointing.",
            "Boring storyline and unconvincing performances.",
            "Poor direction and confusing narrative structure.",
            "Awful script and terrible character development.",
            "Extremely disappointing and poorly executed.",
            "Weak plot and mediocre acting throughout.",
            "One of the worst movies I've ever watched.",
            "Lacks depth and substance. Very boring.",
            "Poor quality production and uninspiring story."
        ] * 100  # Repeat for larger dataset
        
        # Combine datasets
        reviews = positive_reviews + negative_reviews
        sentiments = ['positive'] * len(positive_reviews) + ['negative'] * len(negative_reviews)
        
        df = pd.DataFrame({
            'review': reviews,
            'sentiment': sentiments
        })
        
        # Shuffle dataset
        df = df.sample(frac=1).reset_index(drop=True)
        
        return df
    
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                    val_size: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare train, validation, and test data loaders."""
        
        # Encode labels
        df['label'] = self.label_encoder.fit_transform(df['sentiment'])
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            df['review'].tolist(), df['label'].tolist(),
            test_size=test_size, random_state=42, stratify=df['label']
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
        )
        
        # Create datasets
        train_dataset = SentimentDataset(
            X_train, y_train, self.tokenizer, self.max_length, self.preprocessor
        )
        val_dataset = SentimentDataset(
            X_val, y_val, self.tokenizer, self.max_length, self.preprocessor
        )
        test_dataset = SentimentDataset(
            X_test, y_test, self.tokenizer, self.max_length, self.preprocessor
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )
        
        self.logger.info(f"Data prepared: Train={len(train_dataset)}, "
                        f"Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def get_label_mapping(self) -> Dict[int, str]:
        """Get label to sentiment mapping."""
        return {i: label for i, label in enumerate(self.label_encoder.classes_)}
    
    def analyze_text_lengths(self, texts: List[str]) -> Dict[str, any]:
        """Analyze text length distribution for optimal max_length setting."""
        
        # Tokenize texts to get token counts
        token_lengths = []
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            token_lengths.append(len(tokens))
        
        token_lengths = np.array(token_lengths)
        
        return {
            'mean_length': np.mean(token_lengths),
            'median_length': np.median(token_lengths),
            'max_length': np.max(token_lengths),
            'min_length': np.min(token_lengths),
            'percentile_95': np.percentile(token_lengths, 95),
            'percentile_99': np.percentile(token_lengths, 99),
            'recommended_max_length': int(np.percentile(token_lengths, 95))
        }

# MLE-Star Stage 1: Situation Analysis for NLP
def analyze_nlp_situation(data_path: Optional[str] = None) -> Dict[str, any]:
    """Analyze the NLP task situation for sentiment analysis."""
    
    # Initialize data loader
    data_loader = SentimentDataLoader()
    
    # Load and analyze data
    df, data_stats = data_loader.load_imdb_data(data_path)
    
    # Analyze text lengths
    text_analysis = data_loader.analyze_text_lengths(df['review'].tolist()[:1000])  # Sample for speed
    
    situation_analysis = {
        'problem_type': 'binary sentiment classification',
        'dataset': 'IMDB Movie Reviews (or synthetic)',
        'data_characteristics': {
            **data_stats,
            'text_length_analysis': text_analysis
        },
        'challenges': [
            'Variable text length and complexity',
            'Contextual understanding requirements',
            'Handling negations and sarcasm',
            'Domain-specific vocabulary',
            'Class imbalance potential'
        ],
        'recommended_approaches': [
            'Pre-trained transformer models (BERT, RoBERTa)',
            'Fine-tuning on domain-specific data',
            'Appropriate text preprocessing pipeline',
            'Attention mechanism for context understanding',
            'Ensemble methods for robustness'
        ],
        'technical_considerations': {
            'model_architecture': 'BERT-based transformer',
            'tokenizer': 'bert-base-uncased',
            'max_sequence_length': text_analysis.get('recommended_max_length', 512),
            'batch_size': 16,
            'learning_rate': 2e-5
        }
    }
    
    return situation_analysis

if __name__ == '__main__':
    # Test data loading and preprocessing
    print("Testing NLP data processing...")
    
    # Initialize components
    preprocessor = TextPreprocessor()
    data_loader = SentimentDataLoader()
    
    # Load data
    df, stats = data_loader.load_imdb_data()
    print(f"Loaded {len(df)} samples")
    print(f"Data statistics: {stats}")
    
    # Test preprocessing
    sample_text = "This movie was ABSOLUTELY amazing!!! I loved it so much! @user #movie"
    processed_text = preprocessor.preprocess_text(sample_text)
    print(f"\nOriginal: {sample_text}")
    print(f"Processed: {processed_text}")
    
    # Prepare data loaders
    train_loader, val_loader, test_loader = data_loader.prepare_data(df)
    print(f"\nData loaders created:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test batch loading
    sample_batch = next(iter(train_loader))
    print(f"\nSample batch shapes:")
    print(f"Input IDs: {sample_batch['input_ids'].shape}")
    print(f"Attention Mask: {sample_batch['attention_mask'].shape}")
    print(f"Labels: {sample_batch['label'].shape}")
    
    # Situation analysis
    analysis = analyze_nlp_situation()
    print("\n=== NLP Situation Analysis ===")
    for key, value in analysis.items():
        print(f"{key}: {value}")
