from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
import networkx as nx
from ml.preprocessing import preprocess
from sklearn.metrics.pairwise import cosine_similarity
import yake
from keybert import KeyBERT
from collections import Counter
import re

MODEL_NAME = "all-MiniLM-L6-v2"

class HybridKeywordExtractor:
    def __init__(self, model_name=MODEL_NAME, top_k=10):
        self.model = SentenceTransformer(model_name)
        self.top_k = top_k
        self.tfidf = TfidfVectorizer(ngram_range=(1,2), max_df=0.85, min_df=1)
        self.yake_extractor = yake.KeywordExtractor(top=top_k, stopwords=None)
        self.keybert_model = KeyBERT(model_name)

    def candidate_terms(self, doc):
        cleaned = preprocess(doc)
        tfidf = TfidfVectorizer(ngram_range=(1,2))
        try:
            tfidf_matrix = tfidf.fit_transform([cleaned])
            features = np.array(tfidf.get_feature_names_out())
            scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
            idx = np.where(scores > 0)[0]
            candidates = features[idx].tolist()
            return candidates
        except:
            return []

    def embed_terms(self, terms):
        return self.model.encode(terms, convert_to_numpy=True, show_progress_bar=False)

    def embed_text(self, text):
        return self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]

    def graph_ranking(self, terms, term_embeds, text_embed):
        if len(terms) <= 1:
            return [(terms[0], 1.0)] if terms else []
            
        sim = cosine_similarity(term_embeds)
        G = nx.Graph()
        for i, term in enumerate(terms):
            G.add_node(i, term=term)
        for i in range(len(terms)):
            for j in range(i+1, len(terms)):
                w = float(sim[i, j])
                if w > 0.2:  # add edge only if similarity is meaningful
                    G.add_edge(i, j, weight=w)
        if len(G.nodes) == 0:
            return []
        pr = nx.pagerank(G, weight='weight')
        doc_sims = cosine_similarity(term_embeds, text_embed.reshape(1, -1)).ravel()
        combined = []
        for idx, term in enumerate(terms):
            score = pr.get(idx, 0.0) * 0.6 + float(doc_sims[idx]) * 0.4
            combined.append((term, score))
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined

    def extract_with_yake(self, doc):
        keywords = self.yake_extractor.extract_keywords(doc)
        return [kw[0] for kw in keywords]

    def extract_with_keybert(self, doc):
        keywords = self.keybert_model.extract_keywords(
            doc, keyphrase_ngram_range=(1, 2), stop_words='english'
        )
        return [kw[0] for kw in keywords]

    def ensemble_extraction(self, doc, top_k=None):
        top_k = top_k or self.top_k
        
        methods = [
            self.extract_keywords(doc, top_k*2),
            self.extract_with_yake(doc),
            self.extract_with_keybert(doc)
        ]
        
        # Combine results with voting
        all_keywords = []
        for method in methods:
            all_keywords.extend(method)
        
        # Count occurrences and return most frequent
        keyword_counts = Counter(all_keywords)
        return [kw for kw, count in keyword_counts.most_common(top_k)]

    def extract_keywords(self, doc, top_k=None):
        top_k = top_k or self.top_k
        terms = self.candidate_terms(doc)
        if len(terms) == 0:
            return self.extract_with_yake(doc)[:top_k]
        
        term_embeds = self.embed_terms(terms)
        text_embed = self.embed_text(doc)
        scored = self.graph_ranking(terms, term_embeds, text_embed)
        keywords = [t for t, _ in scored[:top_k*2]]  # Get more keywords for filtering
        
        uniq = []
        seen = set()
        for kw in keywords:
            if kw not in seen:
                uniq.append(kw)
                seen.add(kw)
        return uniq[:top_k]