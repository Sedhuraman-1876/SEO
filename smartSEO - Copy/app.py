import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime

# Add current directory to path so imports work
sys.path.append(os.path.dirname(__file__))

from ml.keyword_extractor import HybridKeywordExtractor
from ml.improvement_suggester import ContentImprover
from ml.intent_classifier import IntentModels
from ml.ranking_predictor import RankingPredictor

# Page configuration
st.set_page_config(
    page_title="SmartSEO Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def train_or_load_models(df):
    """Train new models or load existing ones if available"""
    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    intent_model_path = os.path.join(MODEL_DIR, "intent_model.joblib")
    ranking_model_path = os.path.join(MODEL_DIR, "ranking_model.joblib")
    
    try:
        # Try to load existing models
        intent_model = IntentModels.load_model(intent_model_path)
        ranking_model = RankingPredictor.load_model(ranking_model_path)
        st.sidebar.success("✅ Loaded pre-trained models")
    except (FileNotFoundError, Exception) as e:
        st.sidebar.warning(f"🔄 Training new models: {e}")
        # Train new models
        intent_model = IntentModels().train(df)
        ranking_model = RankingPredictor().train(df)
        
        # Save the trained models
        intent_model.save_model(intent_model_path)
        ranking_model.save_model(ranking_model_path)
        st.sidebar.success("✅ Models trained and saved")
    
    return intent_model, ranking_model

def main():
    st.title("🔍 SmartSEO Content Analyzer")
    st.markdown("Analyze your content for SEO optimization and get actionable recommendations")
    
    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.markdown("Upload or paste your content for analysis")
        
    # Content input
    input_method = st.radio("Choose input method:", 
                           ["📝 Paste Text", "📁 Upload File"])
    
    user_text = ""
    
    if input_method == "📝 Paste Text":
        user_text = st.text_area("Paste your content here:", 
                                height=300,
                                placeholder="Enter your article, blog post, or webpage content...")
        
    
            
    else:
        uploaded_file = st.file_uploader("Upload text file", type=['txt', 'md'])
        if uploaded_file:
            user_text = uploaded_file.read().decode("utf-8")
    
    # Analysis button
    if st.button("🔍 Analyze SEO", type="primary", use_container_width=True) and user_text.strip():
        analyze_content(user_text)

def analyze_content(text):
    with st.spinner("🔄 Analyzing your content. This may take a few seconds..."):
        try:
            # Load dataset
            df = pd.read_csv("data/sample_dataset.csv")
            
            # Load or train models
            intent_model, ranking_model = train_or_load_models(df)
            
            # Initialize tools
            keyword_extractor = HybridKeywordExtractor()
            improver = ContentImprover(keyword_extractor)
            
            # Run analysis
            keywords = keyword_extractor.ensemble_extraction(text, top_k=5)
            intents = intent_model.predict(text)
            rank = ranking_model.predict(text)
            dominant_intent = intents["nb"][0]
            suggestions = improver.suggest_improvements(text, dominant_intent, rank, keywords)
            score = improver.get_content_score(text, suggestions)
            
            # Display results
            display_results(text, keywords, intents, rank, suggestions, score)
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.info("Make sure all required files are in the correct locations")

def display_results(text, keywords, intents, rank, suggestions, score):
    # Overview metrics
    st.success(f"✅ Analysis Complete! Content Score: **{score}/100**")
    
    # Create tabs for organized results
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔑 Keywords", "🎯 Intent & Ranking", "💡 Suggestions"])
    
    with tab1:
        show_overview(text, score)
    
    with tab2:
        show_keyword_analysis(keywords)
    
    with tab3:
        show_intent_ranking(intents, rank)
    
    with tab4:
        show_suggestions(suggestions, score)

def show_overview(text, score):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        word_count = len(text.split())
        st.metric("Word Count", word_count)
    
    with col2:
        st.metric("Character Count", len(text))
    
    with col3:
        sentences = len([s for s in text.split('.') if s.strip()])
        st.metric("Sentences", sentences)
    
    with col4:
        # Color code based on score
        if score >= 80:
            st.metric("SEO Score", f"{score}/100", delta="Excellent", delta_color="off")
        elif score >= 60:
            st.metric("SEO Score", f"{score}/100", delta="Good", delta_color="off")
        else:
            st.metric("SEO Score", f"{score}/100", delta="Needs Improvement", delta_color="inverse")
    
    # Content preview
    with st.expander("📝 Content Preview"):
        st.text_area("Preview", text[:500] + "..." if len(text) > 500 else text, height=150)

def show_keyword_analysis(keywords):
    st.subheader("Top 5 Keywords")
    
    if keywords:
        cols = st.columns(5)
        for i, keyword in enumerate(keywords[:5]):
            with cols[i]:
                st.success(f"**{keyword}**")
        
        st.write("These keywords represent the main topics of your content.")
    else:
        st.warning("No keywords could be extracted from the content.")

def show_intent_ranking(intents, rank):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Content Intent")
        intent = intents["nb"][0]
        
        intent_info = {
            "informational": "🔍 Users seeking information",
            "transactional": "💰 Users ready to take action", 
            "navigational": "🧭 Users looking for specific pages"
        }
        
        st.info(f"**Primary Intent:** {intent}")
        st.write(intent_info.get(intent, "General content"))
        
        # Show both model results
        with st.expander("View Both Models"):
            st.write(f"**Naive Bayes:** {intents['nb'][0]}")
            st.write(f"**Decision Tree:** {intents['dt'][0]}")
    
    with col2:
        st.subheader("Ranking Potential")
        rank_score = rank[0] if rank else 0
        
        if rank_score <= 5:
            st.success(f"**Score:** {rank_score}")
            st.write("🎯 **Excellent ranking potential**")
        elif rank_score <= 10:
            st.warning(f"**Score:** {rank_score}")
            st.write("📈 **Good ranking potential**")
        else:
            st.error(f"**Score:** {rank_score}")
            st.write("⚠️ **Needs improvement for better ranking**")
        
        st.caption("Lower scores indicate better ranking potential")

def show_suggestions(suggestions, score):
    st.subheader("💡 Improvement Suggestions")
    
    if suggestions:
        # Categorize suggestions by type
        critical = [s for s in suggestions if any(word in s.lower() for word in ['missing', 'short', 'no headings', 'poor'])]
        important = [s for s in suggestions if s not in critical]
        
        if critical:
            st.error("🚨 **Critical Improvements Needed:**")
            for i, suggestion in enumerate(critical, 1):
                st.write(f"**{i}.** {suggestion}")
        
        if important:
            st.warning("📋 **Recommended Optimizations:**")
            for i, suggestion in enumerate(important, 1):
                st.write(f"**{i}.** {suggestion}")
    else:
        st.success("✅ **Excellent!** Your content is well-optimized for SEO.")
    
    # Progress visualization
    st.subheader("📊 Content Score Breakdown")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Score", f"{score}/100")
    
    with col2:
        status = "Excellent" if score >= 80 else "Good" if score >= 60 else "Needs Work"
        st.metric("Status", status)
    
    with col3:
        st.metric("Improvement Potential", f"{100-score} points")
    
    # Download results
    if st.button("📥 Save Analysis Results"):
        save_results(text, keywords, intents, rank, suggestions, score)

def save_results(text, keywords, intents, rank, suggestions, score):
    import json
    from datetime import datetime
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "content_preview": text[:200] + "..." if len(text) > 200 else text,
        "content_score": score,
        "keywords": keywords,
        "intent_naive_bayes": intents["nb"][0],
        "intent_decision_tree": intents["dt"][0],
        "ranking_score": float(rank[0]) if rank else None,
        "suggestions": suggestions,
        "word_count": len(text.split()),
        "character_count": len(text)
    }
    
    with open("seo_analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    st.success("✅ Results saved to 'seo_analysis_results.json'")

if __name__ == "__main__":
    main()