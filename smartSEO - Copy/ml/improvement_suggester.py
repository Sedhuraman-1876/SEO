from ml.keyword_extractor import HybridKeywordExtractor
from ml.preprocessing import preprocess
import nltk
import re
from collections import Counter
import numpy as np

class ContentScorer:
    def __init__(self):
        self.weights = {
            'length_score': 0.20,
            'readability_score': 0.15,
            'keyword_score': 0.25,
            'structure_score': 0.15,
            'intent_alignment': 0.15,
            'technical_seo': 0.10
        }
    
    def calculate_score(self, text, keywords, suggestions):
        base_score = 100
        
        # Calculate individual scores
        length_score = self._calculate_length_score(text)
        readability_score = self._calculate_readability_score(text)
        keyword_score = self._calculate_keyword_score(text, keywords)
        structure_score = self._calculate_structure_score(text)
        technical_score = self._calculate_technical_score(text)
        
        # Apply weights
        final_score = (
            length_score * self.weights['length_score'] +
            readability_score * self.weights['readability_score'] +
            keyword_score * self.weights['keyword_score'] +
            structure_score * self.weights['structure_score'] +
            technical_score * self.weights['technical_seo']
        ) * 100
        
        # Deductions based on critical suggestions
        critical_issues = [
            "Content is too short", 
            "No headings detected",
            "Missing target keywords",
            "Very poor readability"
        ]
        
        for suggestion in suggestions:
            for issue in critical_issues:
                if issue.lower() in suggestion.lower():
                    final_score -= 5
        
        return max(0, min(100, int(final_score)))
    
    def _calculate_length_score(self, text):
        word_count = len(text.split())
        if word_count >= 800:
            return 1.0  # Excellent
        elif word_count >= 500:
            return 0.8  # Good
        elif word_count >= 300:
            return 0.6  # Fair
        else:
            return 0.3  # Poor
    
    def _calculate_readability_score(self, text):
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return 0.5
        
        words_per_sentence = [len(nltk.word_tokenize(sent)) for sent in sentences]
        avg_sentence_length = np.mean(words_per_sentence)
        
        if 15 <= avg_sentence_length <= 25:
            return 1.0  # Ideal
        elif 10 <= avg_sentence_length <= 30:
            return 0.7  # Acceptable
        else:
            return 0.4  # Needs improvement
    
    def _calculate_keyword_score(self, text, keywords):
        if not keywords:
            return 0.3
        
        text_lower = text.lower()
        keyword_presence = sum(1 for kw in keywords if kw.lower() in text_lower)
        coverage_ratio = keyword_presence / len(keywords)
        
        return min(1.0, coverage_ratio * 1.2)  # Bonus for full coverage
    
    def _calculate_structure_score(self, text):
        score = 0.5  # Base score
        
        # Check for headings
        if re.search(r'#+|<\/?h[1-6]|^[A-Z][A-Z\s]{10,}:', text, re.MULTILINE):
            score += 0.3
        
        # Check for paragraphs (multiple sentences)
        sentences = nltk.sent_tokenize(text)
        if len(sentences) >= 5:
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_technical_score(self, text):
        score = 0.5
        
        # Check for basic SEO elements
        if len(text) > 100:  # Has substantial content
            score += 0.3
        
        # Check for variety in sentence starters
        sentences = nltk.sent_tokenize(text)
        if len(sentences) > 3:
            starters = [sent.strip()[0] for sent in sentences if sent.strip()]
            unique_starters = len(set(starters))
            if unique_starters / len(starters) > 0.6:
                score += 0.2
        
        return score

class ContentImprover:
    def __init__(self, keyword_extractor=None):
        self.keyword_extractor = keyword_extractor or HybridKeywordExtractor()
        self.scorer = ContentScorer()

    def suggest_improvements(self, text, predicted_intent, predicted_rank, target_keywords=None):
        suggestions = []
        
        if not text or len(text.strip()) < 50:
            return ["Content is too short for meaningful analysis. Please provide at least 100 characters."]

        # Preprocess and analyze
        processed = preprocess(text)
        tokens = nltk.word_tokenize(processed)
        word_count = len(tokens)
        sentences = nltk.sent_tokenize(text)
        char_count = len(text)

        # 1. CONTENT LENGTH ANALYSIS
        length_suggestions = self._analyze_length(word_count, char_count)
        suggestions.extend(length_suggestions)

        # 2. READABILITY ANALYSIS
        readability_suggestions = self._analyze_readability(text, sentences, word_count)
        suggestions.extend(readability_suggestions)

        # 3. KEYWORD OPTIMIZATION
        keyword_suggestions = self._analyze_keywords(text, target_keywords)
        suggestions.extend(keyword_suggestions)

        # 4. STRUCTURE & FORMATTING
        structure_suggestions = self._analyze_structure(text, sentences)
        suggestions.extend(structure_suggestions)

        # 5. INTENT-SPECIFIC OPTIMIZATIONS
        intent_suggestions = self._analyze_intent_alignment(text, predicted_intent, word_count)
        suggestions.extend(intent_suggestions)

        # 6. TECHNICAL SEO SUGGESTIONS
        technical_suggestions = self._analyze_technical_seo(text, predicted_rank)
        suggestions.extend(technical_suggestions)

        # Remove duplicates and limit suggestions
        unique_suggestions = []
        seen = set()
        for suggestion in suggestions:
            if suggestion not in seen:
                unique_suggestions.append(suggestion)
                seen.add(suggestion)
        
        return unique_suggestions[:10]  # Return top 10 most important suggestions

    def _analyze_length(self, word_count, char_count):
        suggestions = []
        
        if word_count < 300:
            suggestions.extend([
                f"📏 **Content is short ({word_count} words)**: Aim for 800+ words for comprehensive coverage.",
                "💡 **Add value**: Include examples, case studies, statistics, or expert quotes.",
                "🔍 **Expand sections**: Add FAQs, step-by-step guides, or comparative analysis."
            ])
        elif word_count < 500:
            suggestions.append(f"📏 **Content length is adequate ({word_count} words)**: Consider expanding to 800+ words for better depth.")
        elif word_count >= 1500:
            suggestions.append(f"📏 **Content is very long ({word_count} words)**: Consider breaking into multiple articles or adding a table of contents.")
        
        return suggestions

    def _analyze_readability(self, text, sentences, word_count):
        suggestions = []
        
        if not sentences:
            return ["📖 **Readability issue**: Content appears to be unstructured. Add proper sentences and paragraphs."]
        
        # Calculate average sentence length
        avg_sentence_length = word_count / len(sentences)
        
        if avg_sentence_length > 25:
            suggestions.extend([
                f"📖 **Long sentences detected (avg: {avg_sentence_length:.1f} words)**: Break complex sentences into simpler ones.",
                "💡 **Improve flow**: Use transition words and vary sentence structure."
            ])
        elif avg_sentence_length < 12:
            suggestions.append(f"📖 **Short sentences (avg: {avg_sentence_length:.1f} words)**: Combine some sentences for better flow.")
        
        # Check sentence variety
        if len(sentences) > 5:
            starters = [sent.strip()[:20] for sent in sentences]
            if len(set(starters)) / len(starters) < 0.6:
                suggestions.append("🔄 **Sentence variety**: Vary sentence beginnings to improve readability.")
        
        return suggestions

    def _analyze_keywords(self, text, target_keywords):
        suggestions = []
        text_lower = text.lower()
        
        if target_keywords:
            # Check keyword presence
            missing_keywords = [kw for kw in target_keywords if kw.lower() not in text_lower]
            if missing_keywords:
                suggestions.append(f"🔑 **Missing keywords**: Incorporate these important terms: {', '.join(missing_keywords[:3])}")
            
            # Check keyword frequency and placement
            for kw in target_keywords[:3]:  # Check top 3 keywords
                count = text_lower.count(kw.lower())
                if count == 0:
                    continue
                elif count == 1:
                    suggestions.append(f"🔑 **Keyword frequency**: '{kw}' appears only once. Use it 2-3 times naturally.")
                elif count > 5:
                    suggestions.append(f"🔑 **Keyword optimization**: '{kw}' appears {count} times. Ensure natural usage.")
            
            # Check keyword placement
            first_100 = text_lower[:100]
            important_keywords_in_opening = any(kw.lower() in first_100 for kw in target_keywords[:2])
            if not important_keywords_in_opening:
                suggestions.append("🔑 **Keyword placement**: Include main keywords in the first paragraph.")
        
        return suggestions

    def _analyze_structure(self, text, sentences):
        suggestions = []
        
        # Check for headings
        has_headings = bool(re.search(r'#+|<\/?h[1-6]', text, re.IGNORECASE))
        if not has_headings and len(sentences) > 8:
            suggestions.extend([
                "📐 **Structure improvement**: Add headings (H2, H3) to organize content.",
                "💡 **Formatting tips**: Use bullet points, numbered lists, and short paragraphs."
            ])
        
        # Check paragraph length
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        if paragraphs:
            avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs)
            if avg_paragraph_length > 150:
                suggestions.append("📐 **Paragraph length**: Break long paragraphs into smaller ones (50-100 words ideal).")
        
        return suggestions

    def _analyze_intent_alignment(self, text, predicted_intent, word_count):
        suggestions = []
        
        intent_suggestions = {
            "informational": [
                "🎯 **Informational intent**: Focus on educating and providing comprehensive information.",
                "💡 **Enhance with**: Definitions, step-by-step guides, examples, and expert insights.",
                "📊 **Add value**: Include statistics, research findings, and practical applications."
            ],
            "transactional": [
                "🎯 **Transactional intent**: Optimize for conversions and action-taking.",
                "💡 **Include**: Clear CTAs, benefits, pricing, testimonials, and guarantees.",
                "🛒 **Conversion tips**: Add urgency, scarcity, and risk-reversal elements."
            ],
            "navigational": [
                "🎯 **Navigational intent**: Help users find specific information quickly.",
                "💡 **Improve with**: Clear navigation, internal links, and organized content structure.",
                "🔍 **User experience**: Ensure fast loading and mobile-friendly design."
            ]
        }
        
        suggestions.extend(intent_suggestions.get(predicted_intent, [
            "🎯 **General SEO**: Focus on user intent and providing clear value."
        ]))
        
        # Intent-specific length recommendations
        if predicted_intent == "informational" and word_count < 800:
            suggestions.append("📏 **Depth needed**: Informational content benefits from comprehensive coverage (1000+ words ideal).")
        elif predicted_intent == "transactional" and word_count > 1000:
            suggestions.append("📏 **Conciseness**: Transactional content should be clear and action-oriented (500-800 words ideal).")
        
        return suggestions

    def _analyze_technical_seo(self, text, predicted_rank):
        suggestions = []
        
        if predicted_rank and len(predicted_rank) > 0:
            rank_score = predicted_rank[0]
            
            if rank_score > 15:
                suggestions.extend([
                    "📈 **Ranking potential**: Focus on content quality and relevance.",
                    "🔗 **Backlink strategy**: Consider building authoritative backlinks.",
                    "⚡ **Technical SEO**: Improve page speed and mobile responsiveness."
                ])
            elif rank_score <= 5:
                suggestions.append("📈 **Excellent ranking potential**: Maintain quality and freshness.")
        
        # Meta content suggestions
        if len(text) > 200:
            first_para = text[:200]
            if not any(word in first_para.lower() for word in ['the', 'this', 'we', 'our']):
                suggestions.append("⚡ **Opening optimization**: Start with a compelling hook that includes key topics.")
        
        return suggestions

    def get_content_score(self, text, suggestions):
        # Extract keywords for scoring
        keywords = self.keyword_extractor.ensemble_extraction(text, top_k=5) if text else []
        return self.scorer.calculate_score(text, keywords, suggestions)