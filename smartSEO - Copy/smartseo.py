from ml.keyword_extractor import HybridKeywordExtractor
from ml.intent_classifier import IntentModels
from ml.ranking_predictor import RankingPredictor
from ml.improvement_suggester import ContentImprover
import pandas as pd
import os
import json
from datetime import datetime

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_or_load_models(df):
    """Train new models or load existing ones if available"""
    intent_model_path = os.path.join(MODEL_DIR, "intent_model.joblib")
    ranking_model_path = os.path.join(MODEL_DIR, "ranking_model.joblib")
    
    try:
        # Try to load existing models
        intent_model = IntentModels.load_model(intent_model_path)
        ranking_model = RankingPredictor.load_model(ranking_model_path)
        print("Loaded pre-trained models")
    except (FileNotFoundError, Exception) as e:
        print(f"Training new models: {e}")
        # Train new models
        intent_model = IntentModels().train(df)
        ranking_model = RankingPredictor().train(df)
        
        # Save the trained models
        intent_model.save_model(intent_model_path)
        ranking_model.save_model(ranking_model_path)
    
    return intent_model, ranking_model

def generate_html_report(text, keywords, intents, rank, suggestions, score, filename="seo_report.html"):
    """Generate an HTML report of the SEO analysis"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SEO Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .section { margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }
            .keyword { display: inline-block; background: #f0f7ff; padding: 5px 10px; margin: 5px; border-radius: 3px; }
            .suggestion { background: #fff4e6; padding: 10px; margin: 10px 0; border-left: 4px solid #ffa94d; }
            .score { font-size: 24px; font-weight: bold; color: #2c5aa0; }
            .positive { color: green; }
            .negative { color: red; }
            .neutral { color: orange; }
        </style>
    </head>
    <body>
        <h1>SEO Content Analysis Report</h1>
        <p>Generated on {date}</p>
        
        <div class="section">
            <h2>Content Score: <span class="score">{score}/100</span></h2>
        </div>
        
        <div class="section">
            <h2>Analyzed Content</h2>
            <div style="background: #f9f9f9; padding: 15px; border-radius: 5px;">
                <pre>{text}</pre>
            </div>
        </div>
        
        <div class="section">
            <h2>Extracted Keywords</h2>
            {keywords_html}
        </div>
        
        <div class="section">
            <h2>Predicted Intent</h2>
            <p><strong>Naive Bayes:</strong> {nb_intent}</p>
            <p><strong>Decision Tree:</strong> {dt_intent}</p>
        </div>
        
        <div class="section">
            <h2>Predicted Ranking Score</h2>
            <p>{rank_score}</p>
            <p><em>Lower scores indicate better ranking potential</em></p>
        </div>
        
        <div class="section">
            <h2>Suggested Improvements</h2>
            {suggestions_html}
        </div>
    </body>
    </html>
    """
    
    # Prepare keywords HTML
    keywords_html = "".join([f'<span class="keyword">{kw}</span>' for kw in keywords])
    
    # Prepare suggestions HTML
    if suggestions:
        suggestions_html = "".join([f'<div class="suggestion">{s}</div>' for s in suggestions])
    else:
        suggestions_html = '<div class="suggestion positive">✅ Your content looks good! No major improvements needed.</div>'
    
    # Fill the template
    html_content = html_template.format(
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        score=score,
        text=text[:1000] + "..." if len(text) > 1000 else text,
        keywords_html=keywords_html,
        nb_intent=intents["nb"][0],
        dt_intent=intents["dt"][0],
        rank_score=rank[0] if rank else "N/A",
        suggestions_html=suggestions_html
    )
    
    # Save to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Report saved to {filename}")

def main():
    # -------- Step 1: Load dataset --------
    try:
        df = pd.read_csv("data/sample_dataset.csv")
        print("Dataset loaded successfully")
    except FileNotFoundError:
        print("Warning: sample_dataset.csv not found. Using empty dataset for training.")
        df = pd.DataFrame({
            'text': ['buy product', 'how to use', 'contact us'],
            'intent': ['transactional', 'informational', 'navigational'],
            'rank_score': [5.0, 12.0, 30.0]
        })
    
    # -------- Step 2: Train or load models --------
    intent_model, ranking_model = train_or_load_models(df)
    
    # -------- Step 3: Initialize tools --------
    keyword_extractor = HybridKeywordExtractor()
    improver = ContentImprover(keyword_extractor)
    
    # -------- Step 4: Take multi-line input --------
    print("\nPaste your content below (type END on a new line when finished):\n")
    
    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        except EOFError:
            break
    
    user_text = "\n".join(lines)
    
    # -------- Step 5: Run analysis --------
    if not user_text.strip():
        print("\n⚠️ No input given. Exiting.")
        return
    
    print("\n🔹 Analyzing content...")
    
    # Keywords
    keywords = keyword_extractor.ensemble_extraction(user_text, top_k=5)
    print("\n🔹 Extracted Keywords:")
    print(keywords)
    
    # Intent classification
    intents = intent_model.predict(user_text)
    print("\n🔹 Predicted Intent:")
    print(f"Naive Bayes: {intents['nb'][0]}")
    print(f"Decision Tree: {intents['dt'][0]}")
    
    # Ranking prediction
    rank = ranking_model.predict(user_text)
    print("\n🔹 Predicted Ranking Score:")
    print(rank[0])
    
    # Suggestions
    dominant_intent = intents["nb"][0]  # pick Naive Bayes output
    suggestions = improver.suggest_improvements(
        user_text,
        predicted_intent=dominant_intent,
        predicted_rank=rank,
        target_keywords=keywords
    )
    
    # Calculate content score
    score = improver.get_content_score(user_text, suggestions)
    
    print("\n🔹 Content Score:", f"{score}/100")
    print("\n🔹 Suggested Improvements:")
    if suggestions:
        for i, s in enumerate(suggestions, 1):
            print(f"{i}. {s}")
    else:
        print("✅ Your content looks good! No major improvements needed.")
    
    # Generate HTML report
    generate_html_report(user_text, keywords, intents, rank, suggestions, score)
    
    # Save results to JSON
    results = {
        "timestamp": datetime.now().isoformat(),
        "content_preview": user_text[:200] + "..." if len(user_text) > 200 else user_text,
        "keywords": keywords,
        "intents": intents,
        "ranking_score": float(rank[0]) if rank else None,
        "content_score": score,
        "suggestions": suggestions
    }
    
    with open("seo_analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Analysis complete. Results saved to seo_analysis_results.json and seo_report.html")

if __name__ == "__main__":
    main()