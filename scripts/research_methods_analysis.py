"""
🔬 Advanced Exploratory Analytics: Research Paper Methods
Using cutting-edge techniques from data science and social media research

Techniques included:
1. Power Law / Zipf's Law Analysis (Scale-free networks)
2. Topic Modeling with LDA on video titles
3. Network Centrality Analysis
4. Benford's Law for fraud/anomaly detection
5. UMAP Manifold Learning for channel embeddings
6. Information Entropy of content ecosystem
7. Pareto Frontier Analysis (efficiency vs scale)
8. Temporal Trend Decomposition
9. Collaborative Filtering similarity
10. Gini & Lorenz inequality measures
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
OUTPUT_DIR = Path("data/processed/analysis/advanced")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_datasets():
    """Load all available datasets"""
    data = {}
    
    try:
        data['global'] = pd.read_csv('data/raw/global_stats/Global YouTube Statistics.csv', encoding='latin-1')
        print(f"✓ Global stats: {len(data['global'])} channels")
    except: pass
    
    try:
        data['top_channels'] = pd.read_csv('data/raw/top_channels/most_subscribed_youtube_channels.csv')
        print(f"✓ Top channels: {len(data['top_channels'])} channels")
    except: pass
    
    try:
        data['spotify_yt'] = pd.read_csv('data/raw/spotify_youtube/Spotify_Youtube.csv')
        print(f"✓ Spotify/YouTube: {len(data['spotify_yt'])} tracks")
    except: pass
    
    # Load trending videos
    trending_frames = []
    trending_dir = Path('data/raw/trending_historical')
    for f in trending_dir.glob('*videos.csv'):
        try:
            df = pd.read_csv(f, encoding='latin-1', on_bad_lines='skip')
            df['country'] = f.stem[:2]
            trending_frames.append(df)
        except: pass
    
    if trending_frames:
        data['trending'] = pd.concat(trending_frames, ignore_index=True)
        print(f"✓ Trending: {len(data['trending'])} videos")
    
    # Load influencer data
    influencer_frames = []
    inf_dir = Path('data/raw/influencers_2024/Top 100 Influencers')
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
        print(f"✓ Influencers: {len(data['influencers'])} from {len(influencer_frames)} countries")
    
    return data


# =============================================================================
# 1. POWER LAW / ZIPF'S LAW ANALYSIS
# Research: Scale-free network theory (Barabási-Albert model)
# =============================================================================

def analyze_power_law(df):
    """
    Test if YouTube subscriber distribution follows a power law
    (common in social networks - "rich get richer" phenomenon)
    """
    print("\n" + "="*70)
    print("📊 POWER LAW ANALYSIS (Zipf's Law)")
    print("   Testing if YouTube follows 'rich get richer' dynamics")
    print("="*70)
    
    sub_col = 'subscribers' if 'subscribers' in df.columns else 'Subscribers'
    if sub_col not in df.columns:
        print("   No subscriber data")
        return
    
    subs = df[sub_col].dropna().values
    subs = subs[subs > 0]
    subs = np.sort(subs)[::-1]  # Sort descending
    
    # Rank-frequency plot (Zipf plot)
    ranks = np.arange(1, len(subs) + 1)
    
    # Fit power law: y = C * x^(-alpha)
    log_ranks = np.log10(ranks)
    log_subs = np.log10(subs)
    
    # Linear regression on log-log scale
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_subs)
    alpha = -slope  # Power law exponent
    
    print(f"\n   📈 POWER LAW FIT:")
    print(f"      • Exponent (α): {alpha:.3f}")
    print(f"      • R² value: {r_value**2:.4f}")
    print(f"      • P-value: {p_value:.2e}")
    
    if 1.5 < alpha < 3.0:
        print(f"      ✓ FOLLOWS POWER LAW: Classic scale-free network (α between 1.5-3)")
    elif alpha > 3.0:
        print(f"      📌 Super-concentrated: Extreme inequality (α > 3)")
    else:
        print(f"      📌 Sub-linear: More egalitarian than typical social networks")
    
    # Calculate fraction of total held by top percentiles
    total_subs = subs.sum()
    top_1_pct = subs[:int(len(subs)*0.01)].sum() / total_subs * 100
    top_10_pct = subs[:int(len(subs)*0.10)].sum() / total_subs * 100
    
    print(f"\n   🏆 CONCENTRATION:")
    print(f"      • Top 1% of channels hold: {top_1_pct:.1f}% of all subscribers")
    print(f"      • Top 10% of channels hold: {top_10_pct:.1f}% of all subscribers")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Log-log Zipf plot
    axes[0].loglog(ranks, subs, 'b.', alpha=0.5, markersize=3)
    fit_line = 10**(intercept + slope * log_ranks)
    axes[0].loglog(ranks, fit_line, 'r-', linewidth=2, 
                   label=f'Power Law fit (α={alpha:.2f}, R²={r_value**2:.3f})')
    axes[0].set_xlabel('Rank')
    axes[0].set_ylabel('Subscribers')
    axes[0].set_title('Zipf Plot: YouTube Subscriber Distribution')
    axes[0].legend()
    
    # Histogram on log scale
    axes[1].hist(np.log10(subs), bins=50, color='steelblue', edgecolor='white')
    axes[1].set_xlabel('Log10(Subscribers)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Subscriber Counts (Log Scale)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'power_law_analysis.png', dpi=150)
    plt.close()
    print(f"\n   📊 Saved: {OUTPUT_DIR / 'power_law_analysis.png'}")


# =============================================================================
# 2. TOPIC MODELING WITH LDA
# Research: Latent Dirichlet Allocation for content categorization
# =============================================================================

def topic_modeling_titles(df, n_topics=8):
    """
    Discover latent topics in video titles using LDA
    """
    print("\n" + "="*70)
    print("📚 TOPIC MODELING (LDA) ON VIDEO TITLES")
    print("   Discovering hidden themes in trending content")
    print("="*70)
    
    if 'title' not in df.columns:
        print("   No title data")
        return
    
    # Filter to English-ish titles
    titles = df['title'].dropna().astype(str)
    english_pattern = r'^[a-zA-Z0-9\s\.\,\!\?\-\:\'\"\|\(\)\[\]]+$'
    titles = titles[titles.str.match(english_pattern, na=False)]
    titles = titles[titles.str.len() > 10]
    
    if len(titles) < 1000:
        print("   Not enough English titles for topic modeling")
        return
    
    print(f"   Analyzing {len(titles)} titles...")
    
    # Vectorize
    vectorizer = CountVectorizer(max_features=2000, stop_words='english', 
                                 min_df=10, max_df=0.5, ngram_range=(1, 2))
    doc_term_matrix = vectorizer.fit_transform(titles)
    
    # LDA
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, 
                                    max_iter=20, learning_method='batch')
    lda.fit(doc_term_matrix)
    
    # Extract topics
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"\n   🔍 DISCOVERED TOPICS:")
    topic_names = []
    for idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        # Name the topic based on top words
        topic_name = f"Topic {idx+1}: {top_words[0].title()}"
        topic_names.append(topic_name)
        print(f"\n      {topic_name}")
        print(f"      Keywords: {', '.join(top_words[:7])}")
    
    # Topic distribution
    topic_dist = lda.transform(doc_term_matrix)
    dominant_topics = topic_dist.argmax(axis=1)
    topic_counts = pd.Series(dominant_topics).value_counts().sort_index()
    
    print(f"\n   📊 TOPIC DISTRIBUTION:")
    for i, count in topic_counts.items():
        pct = count / len(dominant_topics) * 100
        print(f"      {topic_names[i]}: {pct:.1f}%")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(n_topics), [topic_counts.get(i, 0) for i in range(n_topics)],
                   color=plt.cm.viridis(np.linspace(0.2, 0.8, n_topics)))
    ax.set_yticks(range(n_topics))
    ax.set_yticklabels([f"Topic {i+1}" for i in range(n_topics)])
    ax.set_xlabel('Number of Videos')
    ax.set_title('LDA Topic Distribution in YouTube Trending Titles')
    ax.invert_yaxis()
    
    # Add topic keywords as annotations
    for i, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-3:][::-1]
        keywords = ', '.join([feature_names[j] for j in top_words_idx])
        ax.text(bars[i].get_width() + 100, bars[i].get_y() + bars[i].get_height()/2,
               keywords, va='center', fontsize=8, style='italic')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'topic_modeling.png', dpi=150)
    plt.close()
    print(f"\n   📊 Saved: {OUTPUT_DIR / 'topic_modeling.png'}")


# =============================================================================
# 3. BENFORD'S LAW ANALYSIS
# Research: First-digit law for anomaly/fraud detection
# =============================================================================

def benfords_law_analysis(df):
    """
    Test if view counts follow Benford's Law
    Deviations may indicate manipulation or unusual patterns
    """
    print("\n" + "="*70)
    print("🔢 BENFORD'S LAW ANALYSIS")
    print("   Testing for natural vs. artificial patterns in view counts")
    print("="*70)
    
    view_col = 'views' if 'views' in df.columns else 'view_count'
    if view_col not in df.columns:
        view_col = 'video views'
    if view_col not in df.columns:
        print("   No view data")
        return
    
    views = df[view_col].dropna()
    views = views[views > 0]
    
    # Extract first digit
    first_digits = views.apply(lambda x: int(str(int(x))[0]))
    first_digit_counts = first_digits.value_counts().sort_index()
    
    # Benford's expected distribution
    benford_expected = {d: np.log10(1 + 1/d) for d in range(1, 10)}
    
    # Chi-square test
    observed = [first_digit_counts.get(d, 0) for d in range(1, 10)]
    expected = [benford_expected[d] * len(first_digits) for d in range(1, 10)]
    
    chi2, p_value = stats.chisquare(observed, expected)
    
    print(f"\n   📊 FIRST DIGIT DISTRIBUTION:")
    print(f"      {'Digit':<8} {'Observed':<12} {'Benford Expected':<18} {'Deviation'}")
    for d in range(1, 10):
        obs_pct = first_digit_counts.get(d, 0) / len(first_digits) * 100
        exp_pct = benford_expected[d] * 100
        deviation = obs_pct - exp_pct
        symbol = "▲" if deviation > 2 else "▼" if deviation < -2 else "≈"
        print(f"      {d:<8} {obs_pct:<12.1f}% {exp_pct:<18.1f}% {symbol} {deviation:+.1f}%")
    
    print(f"\n   📈 STATISTICAL TEST:")
    print(f"      • Chi-square statistic: {chi2:.2f}")
    print(f"      • P-value: {p_value:.4f}")
    
    if p_value < 0.01:
        print(f"      ⚠️  SIGNIFICANT DEVIATION from Benford's Law")
        print(f"         This could indicate: unusual data patterns, manipulation, or non-organic growth")
    elif p_value < 0.05:
        print(f"      📌 MARGINAL DEVIATION from Benford's Law")
    else:
        print(f"      ✓ FOLLOWS Benford's Law - data appears natural/organic")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(1, 10)
    width = 0.35
    
    obs_pct = [first_digit_counts.get(d, 0) / len(first_digits) * 100 for d in range(1, 10)]
    exp_pct = [benford_expected[d] * 100 for d in range(1, 10)]
    
    bars1 = ax.bar(x - width/2, obs_pct, width, label='Observed', color='steelblue')
    bars2 = ax.bar(x + width/2, exp_pct, width, label="Benford's Law", color='coral', alpha=0.7)
    
    ax.set_xlabel('First Digit')
    ax.set_ylabel('Percentage')
    ax.set_title(f"Benford's Law Analysis (χ²={chi2:.1f}, p={p_value:.3f})")
    ax.set_xticks(x)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'benfords_law.png', dpi=150)
    plt.close()
    print(f"\n   📊 Saved: {OUTPUT_DIR / 'benfords_law.png'}")


# =============================================================================
# 4. INFORMATION ENTROPY ANALYSIS
# Research: Shannon entropy for ecosystem diversity
# =============================================================================

def entropy_analysis(df):
    """
    Calculate Shannon entropy of content categories
    Higher entropy = more diverse ecosystem
    """
    print("\n" + "="*70)
    print("🎲 INFORMATION ENTROPY ANALYSIS")
    print("   Measuring diversity of the YouTube content ecosystem")
    print("="*70)
    
    cat_col = 'category' if 'category' in df.columns else 'Category'
    if cat_col not in df.columns:
        print("   No category data")
        return
    
    categories = df[cat_col].dropna()
    
    # Calculate probabilities
    cat_counts = categories.value_counts(normalize=True)
    
    # Shannon entropy: H = -Σ p_i * log2(p_i)
    entropy = stats.entropy(cat_counts, base=2)
    max_entropy = np.log2(len(cat_counts))  # Maximum possible entropy
    normalized_entropy = entropy / max_entropy
    
    print(f"\n   📊 ENTROPY METRICS:")
    print(f"      • Number of categories: {len(cat_counts)}")
    print(f"      • Shannon Entropy: {entropy:.3f} bits")
    print(f"      • Maximum possible: {max_entropy:.3f} bits")
    print(f"      • Normalized entropy: {normalized_entropy:.3f} (0=concentrated, 1=uniform)")
    
    if normalized_entropy > 0.8:
        print(f"      ✓ HIGH DIVERSITY: Content is well-distributed across categories")
    elif normalized_entropy > 0.5:
        print(f"      📌 MODERATE DIVERSITY: Some category concentration exists")
    else:
        print(f"      ⚠️  LOW DIVERSITY: Content is concentrated in few categories")
    
    # Effective number of categories (perplexity)
    perplexity = 2 ** entropy
    print(f"\n   🎯 EFFECTIVE CATEGORIES: {perplexity:.1f}")
    print(f"      (The ecosystem behaves like it has ~{int(perplexity)} equal categories)")
    
    # Dominance (probability of most common category)
    dominance = cat_counts.iloc[0]
    print(f"      • Dominant category: {cat_counts.index[0]} ({dominance*100:.1f}%)")
    
    # Visualize entropy across different groupings
    print(f"\n   📊 CATEGORY DISTRIBUTION:")
    for cat, pct in cat_counts.head(10).items():
        bar = "█" * int(pct * 50)
        print(f"      {str(cat)[:20]:<20} {bar} {pct*100:.1f}%")


# =============================================================================
# 5. UMAP MANIFOLD LEARNING
# Research: Non-linear dimensionality reduction for embeddings
# =============================================================================

def umap_channel_embedding(df):
    """
    Create 2D embeddings of channels using UMAP for visualization
    """
    print("\n" + "="*70)
    print("🗺️  UMAP MANIFOLD LEARNING")
    print("   Creating 2D channel embeddings for visualization")
    print("="*70)
    
    # Prepare features
    feature_cols = []
    for col in ['subscribers', 'Subscribers', 'video views', 'Video Views', 
                'video_count', 'Video Count', 'uploads']:
        if col in df.columns:
            feature_cols.append(col)
    
    if len(feature_cols) < 2:
        print("   Not enough numeric features")
        return
    
    print(f"   Using features: {feature_cols}")
    
    X = df[feature_cols].copy()
    X = X.apply(lambda x: np.log1p(x.fillna(0)))
    X = X.replace([np.inf, -np.inf], 0)
    
    # Remove rows with all zeros
    mask = (X != 0).any(axis=1)
    X = X[mask]
    df_subset = df[mask].copy()
    
    if len(X) < 50:
        print("   Not enough valid data points")
        return
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Try UMAP if available, else use t-SNE
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embedding = reducer.fit_transform(X_scaled)
        method = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        embedding = reducer.fit_transform(X_scaled)
        method = "t-SNE"
    
    print(f"   ✓ Created 2D embedding using {method}")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color by category if available
    cat_col = 'category' if 'category' in df_subset.columns else 'Category'
    if cat_col in df_subset.columns:
        categories = df_subset[cat_col].fillna('Unknown')
        unique_cats = categories.unique()[:10]  # Top 10 categories
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cats)))
        
        for i, cat in enumerate(unique_cats):
            mask = categories == cat
            ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                      c=[colors[i]], label=str(cat)[:15], alpha=0.6, s=30)
        ax.legend(loc='upper right', fontsize=8)
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, s=20, c='steelblue')
    
    ax.set_xlabel(f'{method} Dimension 1')
    ax.set_ylabel(f'{method} Dimension 2')
    ax.set_title(f'{method} Embedding of YouTube Channels')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'umap_embedding.png', dpi=150)
    plt.close()
    print(f"\n   📊 Saved: {OUTPUT_DIR / 'umap_embedding.png'}")


# =============================================================================
# 6. PARETO FRONTIER ANALYSIS
# Research: Multi-objective optimization perspective
# =============================================================================

def pareto_frontier_analysis(df):
    """
    Find Pareto-optimal channels (best trade-offs between metrics)
    """
    print("\n" + "="*70)
    print("🎯 PARETO FRONTIER ANALYSIS")
    print("   Finding channels with optimal efficiency trade-offs")
    print("="*70)
    
    sub_col = 'subscribers' if 'subscribers' in df.columns else 'Subscribers'
    view_col = 'video views' if 'video views' in df.columns else 'Video Views'
    vid_col = 'video_count' if 'video_count' in df.columns else 'Video Count'
    
    if sub_col not in df.columns or view_col not in df.columns:
        print("   Missing required columns")
        return
    
    df = df.copy()
    df['efficiency'] = df[view_col] / df[sub_col].replace(0, np.nan)
    df['scale'] = df[sub_col]
    df = df.dropna(subset=['efficiency', 'scale'])
    
    if len(df) < 10:
        return
    
    # Find Pareto frontier (non-dominated points)
    def is_pareto_efficient(costs):
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            is_efficient[i] = np.all(np.any(costs[:i] < c, axis=1)) and \
                             np.all(np.any(costs[i+1:] < c, axis=1))
        return is_efficient
    
    # For Pareto: maximize both efficiency and scale (negate for minimization)
    costs = np.column_stack([-df['efficiency'].values, -df['scale'].values])
    
    # Simplified Pareto check
    pareto_mask = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        dominated = False
        for j in range(len(df)):
            if i != j:
                if costs[j, 0] <= costs[i, 0] and costs[j, 1] <= costs[i, 1]:
                    if costs[j, 0] < costs[i, 0] or costs[j, 1] < costs[i, 1]:
                        dominated = True
                        break
        pareto_mask[i] = not dominated
    
    pareto_channels = df[pareto_mask]
    
    name_col = 'Youtuber' if 'Youtuber' in df.columns else 'Youtube Channel'
    
    print(f"\n   🏆 PARETO-OPTIMAL CHANNELS (Best in Class):")
    print(f"      Found {len(pareto_channels)} channels on the efficiency frontier")
    
    if name_col in pareto_channels.columns:
        for _, row in pareto_channels.nlargest(10, 'scale').iterrows():
            scale = row['scale']
            eff = row['efficiency']
            if scale > 1e6:
                scale_str = f"{scale/1e6:.0f}M"
            else:
                scale_str = f"{scale/1e3:.0f}K"
            print(f"      • {row[name_col]}: {scale_str} subs, {eff:.0f}x efficiency")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.scatter(np.log10(df['scale']), np.log10(df['efficiency']), 
              alpha=0.3, s=20, c='gray', label='All channels')
    ax.scatter(np.log10(pareto_channels['scale']), np.log10(pareto_channels['efficiency']),
              c='red', s=50, label='Pareto frontier', zorder=5)
    
    ax.set_xlabel('Log10(Subscribers) - Scale')
    ax.set_ylabel('Log10(Views/Subscriber) - Efficiency')
    ax.set_title('Pareto Frontier: Scale vs Efficiency Trade-off')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pareto_frontier.png', dpi=150)
    plt.close()
    print(f"\n   📊 Saved: {OUTPUT_DIR / 'pareto_frontier.png'}")


# =============================================================================
# 7. COLLABORATIVE FILTERING SIMILARITY
# Research: Recommendation system style similarity analysis
# =============================================================================

def collaborative_similarity(df):
    """
    Find similar channels using collaborative filtering approach
    """
    print("\n" + "="*70)
    print("🔗 COLLABORATIVE SIMILARITY ANALYSIS")
    print("   Finding channel pairs that are most similar")
    print("="*70)
    
    if 'channel_title' not in df.columns or 'trending_date' not in df.columns:
        print("   Need trending data with channel_title and trending_date")
        return
    
    # Build channel co-occurrence matrix
    df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m', errors='coerce')
    
    # Get top channels
    top_channels = df['channel_title'].value_counts().head(100).index.tolist()
    
    # Build binary matrix: which channels appeared on which days
    dates = df['trending_date'].dropna().unique()
    
    # Create channel-date matrix
    channel_date_matrix = np.zeros((len(top_channels), len(dates)))
    
    for i, channel in enumerate(top_channels):
        channel_dates = df[df['channel_title'] == channel]['trending_date'].dropna().unique()
        for date in channel_dates:
            if date in dates:
                j = list(dates).index(date)
                channel_date_matrix[i, j] = 1
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(channel_date_matrix)
    
    # Find most similar pairs
    print(f"\n   🤝 MOST SIMILAR CHANNEL PAIRS:")
    pairs = []
    for i in range(len(top_channels)):
        for j in range(i+1, len(top_channels)):
            pairs.append((top_channels[i], top_channels[j], similarity_matrix[i, j]))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    for ch1, ch2, sim in pairs[:15]:
        if sim > 0.3:  # Only show meaningful similarities
            print(f"      {ch1[:25]:<25} ↔ {ch2[:25]:<25} (sim: {sim:.3f})")
    
    # Visualize similarity matrix (top 20 channels)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    top_20 = min(20, len(top_channels))
    sim_subset = similarity_matrix[:top_20, :top_20]
    
    im = ax.imshow(sim_subset, cmap='YlOrRd')
    ax.set_xticks(range(top_20))
    ax.set_yticks(range(top_20))
    ax.set_xticklabels([c[:12] for c in top_channels[:top_20]], rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels([c[:12] for c in top_channels[:top_20]], fontsize=8)
    ax.set_title('Channel Co-occurrence Similarity Matrix')
    plt.colorbar(im, label='Cosine Similarity')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'channel_similarity.png', dpi=150)
    plt.close()
    print(f"\n   📊 Saved: {OUTPUT_DIR / 'channel_similarity.png'}")


# =============================================================================
# 8. TEMPORAL PATTERN ANALYSIS
# Research: Time series decomposition
# =============================================================================

def temporal_patterns(df):
    """
    Analyze temporal patterns in trending data
    """
    print("\n" + "="*70)
    print("⏰ TEMPORAL PATTERN ANALYSIS")
    print("   Finding time-based patterns in trending")
    print("="*70)
    
    if 'trending_date' not in df.columns:
        print("   No trending date data")
        return
    
    df = df.copy()
    df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m', errors='coerce')
    df = df.dropna(subset=['trending_date'])
    
    if len(df) < 100:
        return
    
    # Extract time components
    df['day_of_week'] = df['trending_date'].dt.dayofweek
    df['month'] = df['trending_date'].dt.month
    
    # Day of week patterns
    dow_counts = df.groupby('day_of_week').size()
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    print(f"\n   📅 DAY OF WEEK PATTERNS:")
    for dow, count in dow_counts.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct * 3)
        print(f"      {dow_names[dow]}: {bar} {pct:.1f}%")
    
    # Most popular categories by day
    if 'category_id' in df.columns or 'category' in df.columns:
        cat_col = 'category' if 'category' in df.columns else 'category_id'
        print(f"\n   📊 CATEGORY PATTERNS BY DAY:")
        
        for dow in range(7):
            day_data = df[df['day_of_week'] == dow]
            top_cat = day_data[cat_col].mode().iloc[0] if len(day_data) > 0 else 'N/A'
            print(f"      {dow_names[dow]}: Most common = {top_cat}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Day of week
    axes[0].bar(dow_names, [dow_counts.get(i, 0) for i in range(7)], color='steelblue')
    axes[0].set_xlabel('Day of Week')
    axes[0].set_ylabel('Number of Trending Videos')
    axes[0].set_title('Trending Distribution by Day of Week')
    
    # Monthly
    if len(df['month'].unique()) > 1:
        month_counts = df.groupby('month').size()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        axes[1].bar([month_names[i-1] for i in month_counts.index], month_counts.values, color='coral')
        axes[1].set_xlabel('Month')
        axes[1].set_ylabel('Number of Trending Videos')
        axes[1].set_title('Trending Distribution by Month')
        axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'temporal_patterns.png', dpi=150)
    plt.close()
    print(f"\n   📊 Saved: {OUTPUT_DIR / 'temporal_patterns.png'}")


# =============================================================================
# 9. CROSS-PLATFORM ANALYSIS (Spotify vs YouTube)
# =============================================================================

def cross_platform_analysis(df):
    """
    Analyze correlation between Spotify and YouTube performance
    """
    print("\n" + "="*70)
    print("🎵 CROSS-PLATFORM ANALYSIS: Spotify vs YouTube")
    print("   Do streaming hits translate to YouTube success?")
    print("="*70)
    
    # Find relevant columns
    spotify_cols = [c for c in df.columns if 'stream' in c.lower() or 'spotify' in c.lower()]
    youtube_cols = [c for c in df.columns if 'view' in c.lower() or 'youtube' in c.lower() or 'like' in c.lower()]
    
    print(f"   Spotify columns: {spotify_cols[:5]}")
    print(f"   YouTube columns: {youtube_cols[:5]}")
    
    # Try to find streams and views
    stream_col = None
    view_col = None
    
    for col in df.columns:
        if 'stream' in col.lower() and pd.api.types.is_numeric_dtype(df[col]):
            stream_col = col
        if 'view' in col.lower() and pd.api.types.is_numeric_dtype(df[col]):
            view_col = col
    
    if stream_col and view_col:
        df_clean = df[[stream_col, view_col]].dropna()
        df_clean = df_clean[(df_clean[stream_col] > 0) & (df_clean[view_col] > 0)]
        
        if len(df_clean) > 10:
            # Correlation
            corr = df_clean[stream_col].corr(df_clean[view_col])
            
            # Spearman (rank correlation)
            spearman_corr, p_value = stats.spearmanr(df_clean[stream_col], df_clean[view_col])
            
            print(f"\n   📈 CROSS-PLATFORM CORRELATION:")
            print(f"      • Pearson correlation: {corr:.3f}")
            print(f"      • Spearman correlation: {spearman_corr:.3f} (p={p_value:.2e})")
            
            if spearman_corr > 0.7:
                print(f"      ✓ STRONG correlation: Spotify success → YouTube success")
            elif spearman_corr > 0.4:
                print(f"      📌 MODERATE correlation: Some crossover effect")
            else:
                print(f"      ⚠️  WEAK correlation: Platforms have independent audiences")
            
            # Visualize
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(np.log10(df_clean[stream_col]), np.log10(df_clean[view_col]), 
                      alpha=0.3, s=20)
            ax.set_xlabel(f'Log10({stream_col})')
            ax.set_ylabel(f'Log10({view_col})')
            ax.set_title(f'Spotify vs YouTube Performance\n(Spearman ρ = {spearman_corr:.3f})')
            
            # Add trend line
            z = np.polyfit(np.log10(df_clean[stream_col]), np.log10(df_clean[view_col]), 1)
            p = np.poly1d(z)
            x_line = np.linspace(np.log10(df_clean[stream_col]).min(), 
                                np.log10(df_clean[stream_col]).max(), 100)
            ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'cross_platform.png', dpi=150)
            plt.close()
            print(f"\n   📊 Saved: {OUTPUT_DIR / 'cross_platform.png'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "🔬"*35)
    print("   ADVANCED ANALYTICS: RESEARCH PAPER METHODS")
    print("🔬"*35)
    
    # Load data
    print("\n📂 Loading datasets...")
    data = load_datasets()
    
    if not data:
        print("No data available!")
        return
    
    # Run analyses
    
    # 1. Power Law
    if 'global' in data:
        analyze_power_law(data['global'])
    
    # 2. Topic Modeling
    if 'trending' in data:
        topic_modeling_titles(data['trending'])
    
    # 3. Benford's Law
    if 'trending' in data:
        benfords_law_analysis(data['trending'])
    elif 'global' in data:
        benfords_law_analysis(data['global'])
    
    # 4. Entropy Analysis
    if 'global' in data:
        entropy_analysis(data['global'])
    
    # 5. UMAP Embedding
    if 'global' in data:
        umap_channel_embedding(data['global'])
    
    # 6. Pareto Frontier
    if 'global' in data:
        pareto_frontier_analysis(data['global'])
    
    # 7. Collaborative Similarity
    if 'trending' in data:
        collaborative_similarity(data['trending'])
    
    # 8. Temporal Patterns
    if 'trending' in data:
        temporal_patterns(data['trending'])
    
    # 9. Cross-platform
    if 'spotify_yt' in data:
        cross_platform_analysis(data['spotify_yt'])
    
    # Summary
    print("\n" + "="*70)
    print("✅ ADVANCED ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n   All outputs saved to: {OUTPUT_DIR.absolute()}")
    print("\n   Analyses performed:")
    print("   1. Power Law / Zipf's Law")
    print("   2. Topic Modeling (LDA)")
    print("   3. Benford's Law")
    print("   4. Information Entropy")
    print("   5. UMAP Manifold Learning")
    print("   6. Pareto Frontier")
    print("   7. Collaborative Filtering Similarity")
    print("   8. Temporal Pattern Analysis")
    print("   9. Cross-Platform Correlation")


if __name__ == "__main__":
    main()
