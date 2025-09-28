import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Ensure required NLTK resources are available (downloads if missing)
# -------------------------------------------------------------------
def safe_download(resource):
    try:
        nltk.data.find(resource)
    except LookupError:
        logger.info(f"Downloading NLTK resource: {resource}")
        nltk.download(resource.split("/")[-1], quiet=True)

# Download required resources
required_resources = [
    "tokenizers/punkt",
    "tokenizers/punkt_tab",
    "corpora/stopwords",
    "corpora/wordnet"
]

for resource in required_resources:
    safe_download(resource)

# -------------------------------------------------------------------
# Preprocessing setup
# -------------------------------------------------------------------
STOP = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """Clean text by removing URLs, special characters, and extra spaces"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # Remove special characters
    return text.strip()

def tokenize(text: str):
    """Tokenize text into words"""
    return nltk.word_tokenize(text)

def preprocess(text: str):
    """Full preprocessing pipeline: clean, tokenize, remove stopwords, lemmatize"""
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = [t for t in tokens if t not in STOP and len(t) > 1]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return ' '.join(tokens)