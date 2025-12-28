"""
🔬 Advanced Analytics Part 2: Sentiment, Network Influence & Predictions
Building on research methods with NLP and graph analytics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
OUTPUT_DIR = Path("data/processed/analysis/advanced")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_data():
    """Load all datasets"""
    data = {}
    
    try:
        data['global'] = pd.read_csv('data/raw/global_stats/Global YouTube Statistics.csv', encoding='latin-1')
        print(f"✓ Global: {len(data['global'])} channels")
    except: pass
    
    try:
        data['top_channels'] = pd.read_csv('data/raw/top_channels/most_subscribed_youtube_channels.csv')
        print(f"✓ Top channels: {len(data['top_channels'])}")
    except: pass
    
    trending_frames = []
    for f in Path('data/raw/trending_historical').glob('*videos.csv'):
        try:
            df = pd.read_csv(f, encoding='latin-1', on_bad_lines='skip')
            df['country'] = f.stem[:2]
            trending_frames.append(df)
        except: pass
    if trending_frames:
        data['trending'] = pd.concat(trending_frames, ignore_index=True)
        print(f"✓ Trending: {len(data['trending'])} videos")
    
    influencer_frames = []
    inf_dir = Path('data/raw/influencers_2024/Top 100 Influencers')
    if inf_dir.exists():
        for country_dir in inf_dir.iterdir():
            if country_dir.is_dir():
                yt_file = country_dir / f"youtube_data_{country_dir.name}.csv"
                if yt_file.exists():
                    try:
                        df = pd.read_csv(yt_file)
                        df['country'] = country_dir.name
                        influencer_frames.append(df)
                    except: pass
    if influencer_frames:
        data['influencers'] = pd.concat(influencer_frames, ignore_index=True)
        print(f"✓ Influencers 2024: {len(data['influencers'])}")
    
    return data


# =============================================================================
# 1. SENTIMENT ANALYSIS (Lexicon-based)
# Using VADER-style sentiment from title words
# =============================================================================

def analyze_title_sentiment(df):
    """
    Analyze emotional sentiment in video titles
    Using a lexicon-based approach
    """
    print("\n" + "="*70)
    print("😊 SENTIMENT ANALYSIS ON VIDEO TITLES")
    print("   Emotional tone of trending content")
    print("="*70)
    
    if 'title' not in df.columns:
        print("   No title data")
        return
    
    # Simple sentiment lexicon
    positive_words = {
        'amazing', 'awesome', 'best', 'beautiful', 'brilliant', 'celebrate',
        'excited', 'fantastic', 'fun', 'good', 'great', 'happy', 'incredible',
        'inspiring', 'love', 'lovely', 'perfect', 'success', 'win', 'winner',
        'wonderful', 'wow', 'yes', 'hilarious', 'funny', 'epic', 'legendary'
    }
    
    negative_words = {
        'bad', 'worst', 'terrible', 'horrible', 'fail', 'failure', 'hate',
        'sad', 'angry', 'wrong', 'problem', 'disaster', 'crisis', 'dead',
        'killed', 'scary', 'fear', 'warning', 'danger', 'exposed', 'scam',
        'fake', 'lie', 'never', 'cant', 'wont', 'stop', 'destroy', 'broken'
    }
    
    intensity_words = {
        'very', 'extremely', 'super', 'ultra', 'mega', 'insane', 'crazy',
        'unbelievable', 'absolutely', 'totally', 'completely', 'literally'
    }
    
    def analyze_sentiment(title):
        if pd.isna(title):
            return 0
        words = str(title).lower().split()
        pos = sum(1 for w in words if w in positive_words)
        neg = sum(1 for w in words if w in negative_words)
        intensity = sum(1 for w in words if w in intensity_words)
        return pos - neg + (intensity * 0.5 * (1 if pos > neg else -1 if neg > pos else 0))
    
    df['sentiment'] = df['title'].apply(analyze_sentiment)
    
    # Stats
    print(f"\n   📊 SENTIMENT DISTRIBUTION:")
    positive = (df['sentiment'] > 0).sum()
    negative = (df['sentiment'] < 0).sum()
    neutral = (df['sentiment'] == 0).sum()
    
    total = len(df)
    print(f"      Positive: {positive:,} ({positive/total*100:.1f}%)")
    print(f"      Neutral:  {neutral:,} ({neutral/total*100:.1f}%)")
    print(f"      Negative: {negative:,} ({negative/total*100:.1f}%)")
    
    # Most positive/negative titles
    print(f"\n   😄 MOST POSITIVE TITLES:")
    for _, row in df.nlargest(5, 'sentiment').iterrows():
        print(f"      [{row['sentiment']:.1f}] {str(row['title'])[:60]}...")
    
    print(f"\n   😠 MOST NEGATIVE TITLES:")
    for _, row in df.nsmallest(5, 'sentiment').iterrows():
        print(f"      [{row['sentiment']:.1f}] {str(row['title'])[:60]}...")
    
    # Sentiment vs Views correlation
    if 'views' in df.columns or 'view_count' in df.columns:
        view_col = 'views' if 'views' in df.columns else 'view_count'
        corr = df['sentiment'].corr(df[view_col])
        print(f"\n   📈 SENTIMENT-VIEWS CORRELATION: {corr:.3f}")
        
        if abs(corr) > 0.1:
            direction = "Positive" if corr > 0 else "Negative"
            print(f"      {direction} sentiment titles tend to get {'more' if corr > 0 else 'fewer'} views")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df['sentiment'], bins=30, color='steelblue', edgecolor='white')
    axes[0].axvline(0, color='red', linestyle='--', label='Neutral')
    axes[0].set_xlabel('Sentiment Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Sentiment Distribution in Trending Titles')
    axes[0].legend()
    
    # Pie chart
    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [positive, neutral, negative]
    colors = ['#2ecc71', '#95a5a6', '#e74c3c']
    axes[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Sentiment Categories')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sentiment_analysis.png', dpi=150)
    plt.close()
    print(f"\n   📊 Saved: {OUTPUT_DIR / 'sentiment_analysis.png'}")
    
    return df


# =============================================================================
# 2. PAGERANK-STYLE INFLUENCE ANALYSIS
# Which channels are most "influential" in the trending ecosystem?
# =============================================================================

def pagerank_influence(df, damping=0.85, iterations=50):
    """
    Calculate PageRank-style influence scores for channels
    Based on co-occurrence in trending
    """
    print("\n" + "="*70)
    print("🏆 PAGERANK-STYLE INFLUENCE ANALYSIS")
    print("   Finding the most influential channels in trending")
    print("="*70)
    
    if 'channel_title' not in df.columns or 'trending_date' not in df.columns:
        print("   Need channel_title and trending_date")
        return
    
    df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m', errors='coerce')
    
    # Build co-occurrence adjacency
    channels = df['channel_title'].value_counts().head(200).index.tolist()
    n = len(channels)
    channel_to_idx = {ch: i for i, ch in enumerate(channels)}
    
    adjacency = np.zeros((n, n))
    
    for date in df['trending_date'].dropna().unique():
        day_channels = df[df['trending_date'] == date]['channel_title'].unique()
        day_channels = [ch for ch in day_channels if ch in channel_to_idx]
        
        for ch1 in day_channels:
            for ch2 in day_channels:
                if ch1 != ch2:
                    i, j = channel_to_idx[ch1], channel_to_idx[ch2]
                    adjacency[i, j] += 1
    
    # Normalize to transition matrix
    row_sums = adjacency.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition = adjacency / row_sums
    
    # PageRank iteration
    pagerank = np.ones(n) / n
    teleport = np.ones(n) / n
    
    for _ in range(iterations):
        pagerank = damping * transition.T @ pagerank + (1 - damping) * teleport
    
    # Results
    channel_scores = list(zip(channels, pagerank))
    channel_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n   🌟 TOP 20 MOST INFLUENTIAL CHANNELS (PageRank):")
    for i, (channel, score) in enumerate(channel_scores[:20]):
        bar = "█" * int(score * 500)
        print(f"      {i+1:2d}. {channel[:30]:<30} {bar} ({score:.4f})")
    
    # Power concentration
    top_10_share = sum(s for _, s in channel_scores[:10])
    print(f"\n   📊 INFLUENCE CONCENTRATION:")
    print(f"      Top 10 channels hold: {top_10_share*100:.1f}% of PageRank influence")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 10))
    
    top_30 = channel_scores[:30]
    names = [ch[:20] for ch, _ in top_30]
    scores = [s for _, s in top_30]
    
    bars = ax.barh(range(len(names)), scores, color=plt.cm.viridis(np.linspace(0.8, 0.2, 30)))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('PageRank Score')
    ax.set_title('Channel Influence Scores (PageRank Algorithm)')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pagerank_influence.png', dpi=150)
    plt.close()
    print(f"\n   📊 Saved: {OUTPUT_DIR / 'pagerank_influence.png'}")


# =============================================================================
# 3. GROWTH TRAJECTORY ANALYSIS
# Predicting success based on channel age
# =============================================================================

def growth_trajectory_analysis(df):
    """
    Analyze how channel age relates to subscriber count
    Find fast-growing vs slow-growing channels
    """
    print("\n" + "="*70)
    print("📈 GROWTH TRAJECTORY ANALYSIS")
    print("   How does age relate to success?")
    print("="*70)
    
    # Find year column
    year_col = None
    for col in ['created_year', 'started', 'Started']:
        if col in df.columns:
            year_col = col
            break
    
    if not year_col:
        print("   No year data")
        return
    
    df = df.copy()
    df['year'] = pd.to_numeric(df[year_col], errors='coerce')
    df = df[df['year'].between(2005, 2023)]
    
    current_year = 2024  # Our latest data year
    df['age'] = current_year - df['year']
    
    sub_col = 'subscribers' if 'subscribers' in df.columns else 'Subscribers'
    if sub_col not in df.columns:
        print("   No subscriber data")
        return
    
    df = df.dropna(subset=['age', sub_col])
    df = df[df[sub_col] > 0]
    
    # Growth rate (subs per year of existence)
    df['growth_rate'] = df[sub_col] / df['age']
    
    print(f"\n   📊 AGE DISTRIBUTION OF TOP CHANNELS:")
    age_groups = df.groupby('age').agg({
        sub_col: ['count', 'median']
    })
    age_groups.columns = ['count', 'median_subs']
    
    for age in sorted(df['age'].unique())[:15]:
        if age in age_groups.index:
            count = int(age_groups.loc[age, 'count'])
            median = age_groups.loc[age, 'median_subs']
            if median > 1e6:
                median_str = f"{median/1e6:.0f}M"
            else:
                median_str = f"{median/1e3:.0f}K"
            print(f"      {int(age):2d} years old: {count:3d} channels, median {median_str} subs")
    
    # Find fastest growing (young but big)
    young_big = df[(df['age'] <= 5) & (df[sub_col] > df[sub_col].quantile(0.5))]
    
    print(f"\n   🚀 FASTEST GROWING (≤5 years old, above median):")
    name_col = 'Youtuber' if 'Youtuber' in df.columns else 'Youtube Channel'
    if name_col in df.columns:
        for _, row in young_big.nlargest(10, 'growth_rate').iterrows():
            subs = row[sub_col]
            rate = row['growth_rate']
            if subs > 1e6:
                subs_str = f"{subs/1e6:.0f}M"
            else:
                subs_str = f"{subs/1e3:.0f}K"
            print(f"      {row[name_col]}: {subs_str} subs in {int(row['age'])} years ({rate/1e6:.1f}M/year)")
    
    # Regression: predict subs from age
    X = df[['age']].values
    y = np.log1p(df[sub_col].values)
    
    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)
    
    print(f"\n   📈 AGE-SUCCESS RELATIONSHIP:")
    print(f"      R² score: {r2:.3f}")
    print(f"      Coefficient: {model.coef_[0]:.3f} (log-subs per year)")
    
    if model.coef_[0] > 0:
        print(f"      ✓ Older channels tend to have more subscribers")
    else:
        print(f"      📌 Newer channels are breaking through")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter
    axes[0].scatter(df['age'], np.log10(df[sub_col] + 1), alpha=0.5, s=20)
    axes[0].set_xlabel('Channel Age (years)')
    axes[0].set_ylabel('Log10(Subscribers)')
    axes[0].set_title('Channel Age vs Subscribers')
    
    # Box plot by age group
    df['age_group'] = pd.cut(df['age'], bins=[0, 3, 6, 10, 15, 99], 
                             labels=['0-3y', '3-6y', '6-10y', '10-15y', '15+y'])
    df.boxplot(column=sub_col, by='age_group', ax=axes[1])
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Age Group')
    axes[1].set_ylabel('Subscribers (log scale)')
    axes[1].set_title('Subscriber Distribution by Age')
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'growth_trajectory.png', dpi=150)
    plt.close()
    print(f"\n   📊 Saved: {OUTPUT_DIR / 'growth_trajectory.png'}")


# =============================================================================
# 4. VIRAL POTENTIAL PREDICTION
# What features predict view count?
# =============================================================================

def viral_prediction_model(df):
    """
    Build a model to predict viral potential
    """
    print("\n" + "="*70)
    print("🔮 VIRAL POTENTIAL ANALYSIS")
    print("   What features predict high view counts?")
    print("="*70)
    
    if 'title' not in df.columns:
        print("   No title data")
        return
    
    view_col = 'views' if 'views' in df.columns else 'view_count'
    if view_col not in df.columns:
        print("   No view data")
        return
    
    df = df.copy()
    df = df.dropna(subset=['title', view_col])
    df = df[df[view_col] > 0]
    
    # Sample for speed
    if len(df) > 50000:
        df = df.sample(50000, random_state=42)
    
    # Engineer features
    df['title_length'] = df['title'].str.len()
    df['word_count'] = df['title'].str.split().str.len()
    df['has_caps'] = df['title'].str.contains(r'[A-Z]{3,}', regex=True).astype(int)
    df['has_number'] = df['title'].str.contains(r'\d', regex=True).astype(int)
    df['has_emoji'] = df['title'].str.contains(r'[\U0001F600-\U0001F64F]', regex=True).astype(int)
    df['has_question'] = df['title'].str.contains(r'\?', regex=True).astype(int)
    df['has_exclamation'] = df['title'].str.contains(r'!', regex=True).astype(int)
    
    # Keywords
    clickbait_words = ['you', 'new', 'how', 'why', 'best', 'top', 'first', 'official', 
                       'video', 'watch', 'must', 'now', 'free', 'full', 'live']
    for word in clickbait_words[:5]:  # Top 5
        df[f'has_{word}'] = df['title'].str.lower().str.contains(word).astype(int)
    
    # Feature matrix
    feature_cols = ['title_length', 'word_count', 'has_caps', 'has_number',
                   'has_question', 'has_exclamation'] + [f'has_{w}' for w in clickbait_words[:5]]
    
    X = df[feature_cols].fillna(0)
    y = np.log1p(df[view_col])
    
    # Random Forest model
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    
    # Cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    
    print(f"\n   📊 MODEL PERFORMANCE (5-fold CV):")
    print(f"      Mean R²: {scores.mean():.3f} ± {scores.std():.3f}")
    
    # Fit full model for feature importance
    model.fit(X, y)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n   🎯 FEATURE IMPORTANCE (What predicts views?):")
    for _, row in importance.head(10).iterrows():
        bar = "█" * int(row['importance'] * 100)
        print(f"      {row['feature']:<20} {bar} {row['importance']:.3f}")
    
    # Insights
    top_feature = importance.iloc[0]['feature']
    print(f"\n   💡 KEY INSIGHT: '{top_feature}' is the strongest predictor of views")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.barh(importance['feature'], importance['importance'], color='steelblue')
    ax.set_xlabel('Feature Importance')
    ax.set_title('What Title Features Predict Views?')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'viral_features.png', dpi=150)
    plt.close()
    print(f"\n   📊 Saved: {OUTPUT_DIR / 'viral_features.png'}")


# =============================================================================
# 5. INFLUENCER TIER ANALYSIS (2024 data)
# =============================================================================

def influencer_tier_analysis(df):
    """
    Analyze influencer tiers and engagement patterns
    """
    print("\n" + "="*70)
    print("👑 INFLUENCER TIER ANALYSIS (2024)")
    print("   Understanding the influencer ecosystem")
    print("="*70)
    
    if df is None or len(df) == 0:
        print("   No influencer data")
        return
    
    # Parse followers
    def parse_followers(val):
        if pd.isna(val):
            return 0
        val = str(val).upper().replace(',', '')
        if 'M' in val:
            return float(val.replace('M', '')) * 1_000_000
        elif 'K' in val:
            return float(val.replace('K', '')) * 1_000
        try:
            return float(val)
        except:
            return 0
    
    df['followers_num'] = df['FOLLOWERS'].apply(parse_followers)
    
    # Define tiers
    def get_tier(followers):
        if followers >= 10_000_000:
            return 'Mega (10M+)'
        elif followers >= 1_000_000:
            return 'Macro (1M-10M)'
        elif followers >= 100_000:
            return 'Mid (100K-1M)'
        elif followers >= 10_000:
            return 'Micro (10K-100K)'
        else:
            return 'Nano (<10K)'
    
    df['tier'] = df['followers_num'].apply(get_tier)
    
    tier_counts = df['tier'].value_counts()
    tier_order = ['Mega (10M+)', 'Macro (1M-10M)', 'Mid (100K-1M)', 'Micro (10K-100K)', 'Nano (<10K)']
    
    print(f"\n   📊 INFLUENCER TIER DISTRIBUTION:")
    for tier in tier_order:
        if tier in tier_counts.index:
            count = tier_counts[tier]
            pct = count / len(df) * 100
            print(f"      {tier:<20} {count:>5} ({pct:.1f}%)")
    
    # Top countries by mega influencers
    if 'country' in df.columns:
        mega = df[df['tier'] == 'Mega (10M+)']
        
        print(f"\n   🌍 TOP COUNTRIES FOR MEGA INFLUENCERS:")
        country_mega = mega['country'].value_counts().head(10)
        for country, count in country_mega.items():
            print(f"      {country}: {count} mega influencers")
    
    # Topic distribution
    if 'TOPIC OF INFLUENCE' in df.columns:
        print(f"\n   📑 TOP TOPICS AMONG INFLUENCERS:")
        topic_counts = df['TOPIC OF INFLUENCE'].value_counts().head(10)
        for topic, count in topic_counts.items():
            pct = count / len(df) * 100
            print(f"      {topic}: {count} ({pct:.1f}%)")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Tier distribution
    tier_data = [tier_counts.get(t, 0) for t in tier_order if t in tier_counts.index]
    tier_labels = [t for t in tier_order if t in tier_counts.index]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(tier_labels)))
    axes[0].pie(tier_data, labels=tier_labels, colors=colors, autopct='%1.1f%%')
    axes[0].set_title('Influencer Tier Distribution (2024)')
    
    # Country distribution
    if 'country' in df.columns:
        top_countries = df['country'].value_counts().head(10)
        axes[1].barh(range(len(top_countries)), top_countries.values, color='steelblue')
        axes[1].set_yticks(range(len(top_countries)))
        axes[1].set_yticklabels(top_countries.index)
        axes[1].set_xlabel('Number of Influencers')
        axes[1].set_title('Top Countries by YouTube Influencers')
        axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'influencer_tiers.png', dpi=150)
    plt.close()
    print(f"\n   📊 Saved: {OUTPUT_DIR / 'influencer_tiers.png'}")


# =============================================================================
# 6. CATEGORY EVOLUTION ANALYSIS
# =============================================================================

def category_success_factors(df):
    """
    What makes certain categories more successful?
    """
    print("\n" + "="*70)
    print("🎬 CATEGORY SUCCESS FACTORS")
    print("   Deep dive into what makes categories thrive")
    print("="*70)
    
    cat_col = 'category' if 'category' in df.columns else 'Category'
    if cat_col not in df.columns:
        print("   No category data")
        return
    
    sub_col = 'subscribers' if 'subscribers' in df.columns else 'Subscribers'
    view_col = 'video views' if 'video views' in df.columns else 'Video Views'
    
    df = df.dropna(subset=[cat_col])
    
    # Calculate metrics per category
    category_stats = df.groupby(cat_col).agg({
        sub_col: ['sum', 'mean', 'count', 'std'] if sub_col in df.columns else ['count'],
        view_col: ['sum', 'mean'] if view_col in df.columns else ['count']
    }).fillna(0)
    
    # Flatten column names
    category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns]
    
    if f'{sub_col}_sum' in category_stats.columns:
        category_stats = category_stats.sort_values(f'{sub_col}_sum', ascending=False)
        
        # Calculate efficiency
        if f'{view_col}_sum' in category_stats.columns:
            category_stats['efficiency'] = category_stats[f'{view_col}_sum'] / category_stats[f'{sub_col}_sum']
        
        print(f"\n   📊 CATEGORY RANKINGS:")
        print(f"\n   By Total Subscribers:")
        for cat in category_stats.head(10).index:
            total = category_stats.loc[cat, f'{sub_col}_sum']
            if total > 1e9:
                print(f"      {str(cat)[:20]:<20}: {total/1e9:.1f}B subs")
            else:
                print(f"      {str(cat)[:20]:<20}: {total/1e6:.0f}M subs")
        
        if 'efficiency' in category_stats.columns:
            print(f"\n   By Efficiency (Views per Subscriber):")
            efficient = category_stats.sort_values('efficiency', ascending=False)
            for cat in efficient.head(10).index:
                eff = efficient.loc[cat, 'efficiency']
                print(f"      {str(cat)[:20]:<20}: {eff:.0f}x efficiency")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "🔬"*35)
    print("   ADVANCED ANALYTICS PART 2")
    print("   Sentiment, Influence & Predictions")
    print("🔬"*35)
    
    # Load data
    print("\n📂 Loading datasets...")
    data = load_all_data()
    
    if not data:
        print("No data!")
        return
    
    # Run analyses
    
    # 1. Sentiment Analysis
    if 'trending' in data:
        analyze_title_sentiment(data['trending'])
    
    # 2. PageRank Influence
    if 'trending' in data:
        pagerank_influence(data['trending'])
    
    # 3. Growth Trajectory
    if 'global' in data:
        growth_trajectory_analysis(data['global'])
    
    # 4. Viral Prediction
    if 'trending' in data:
        viral_prediction_model(data['trending'])
    
    # 5. Influencer Tiers
    if 'influencers' in data:
        influencer_tier_analysis(data['influencers'])
    
    # 6. Category Success
    if 'global' in data:
        category_success_factors(data['global'])
    
    print("\n" + "="*70)
    print("✅ PART 2 ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n   Outputs saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
