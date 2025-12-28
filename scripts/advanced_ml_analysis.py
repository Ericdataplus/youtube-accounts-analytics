"""
🔬 Advanced Non-Predictive ML Analysis on YouTube Data
Asking questions normal analysts wouldn't ask

Techniques: Clustering, Anomaly Detection, Dimensionality Reduction,
           Topic Modeling, Network Analysis, Statistical Outliers
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import Counter
import re

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

OUTPUT_DIR = Path("data/processed/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_data():
    """Load all available datasets"""
    data = {}
    
    # Global stats
    try:
        data['global'] = pd.read_csv('data/raw/global_stats/Global YouTube Statistics.csv', 
                                      encoding='latin-1')
        print(f"✓ Global stats: {len(data['global'])} channels")
    except: pass
    
    # Top channels
    try:
        data['top_channels'] = pd.read_csv('data/raw/top_channels/most_subscribed_youtube_channels.csv')
        print(f"✓ Top channels: {len(data['top_channels'])} channels")
    except: pass
    
    # GitHub top subscribed
    try:
        data['github_top'] = pd.read_csv('data/raw/github_channel_growth/data/topSubscribed.csv')
        print(f"✓ GitHub top: {len(data['github_top'])} channels")
    except: pass
    
    # Trending videos (combine all countries)
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
        print(f"✓ Trending videos: {len(data['trending'])} videos, {data['trending']['channel_title'].nunique()} unique channels")
    
    return data


# =============================================================================
# UNCONVENTIONAL QUESTION 1: Are there hidden "tribes" of YouTube channels?
# Using multi-dimensional clustering to find non-obvious groupings
# =============================================================================

def discover_channel_tribes(df):
    """
    Question: Beyond obvious categories, what hidden tribes exist among top channels?
    Normal analysts just look at categories - we'll find emergent behavioral clusters.
    """
    print("\n" + "="*70)
    print("🔍 QUESTION 1: What hidden 'tribes' exist among YouTube channels?")
    print("   (Beyond official categories - finding behavioral clusters)")
    print("="*70)
    
    # Prepare features
    features = []
    feature_names = []
    
    # Check available columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Try to build feature matrix
    if 'subscribers' in df.columns:
        features.append(np.log1p(df['subscribers'].fillna(0)))
        feature_names.append('log_subscribers')
    
    if 'video views' in df.columns:
        features.append(np.log1p(df['video views'].fillna(0)))
        feature_names.append('log_views')
    elif 'Video Views' in df.columns:
        features.append(np.log1p(df['Video Views'].fillna(0)))
        feature_names.append('log_views')
    
    if 'video_count' in df.columns or 'Video Count' in df.columns:
        col = 'video_count' if 'video_count' in df.columns else 'Video Count'
        features.append(np.log1p(df[col].fillna(0)))
        feature_names.append('log_videos')
    
    if len(features) < 2:
        print("   Not enough numeric features for clustering")
        return None
    
    X = np.column_stack(features)
    X = np.nan_to_num(X, nan=0)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal k using silhouette score
    best_k = 4
    best_score = -1
    
    for k in range(3, 10):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k
    
    # Final clustering
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['tribe'] = kmeans.fit_predict(X_scaled)
    
    print(f"\n   ✓ Found {best_k} hidden tribes (silhouette score: {best_score:.3f})")
    
    # Analyze each tribe
    print("\n   TRIBE PROFILES:")
    for tribe in range(best_k):
        tribe_data = df[df['tribe'] == tribe]
        n = len(tribe_data)
        
        name_col = 'Youtuber' if 'Youtuber' in df.columns else 'Youtube Channel'
        if name_col in df.columns:
            examples = tribe_data[name_col].head(3).tolist()
        else:
            examples = []
        
        print(f"\n   TRIBE {tribe} ({n} channels):")
        
        for i, feat in enumerate(feature_names):
            col_map = {
                'log_subscribers': 'subscribers',
                'log_views': 'video views',
                'log_videos': 'video_count'
            }
            if feat == 'log_views':
                col = 'video views' if 'video views' in df.columns else 'Video Views'
            elif feat == 'log_videos':
                col = 'video_count' if 'video_count' in df.columns else 'Video Count'
            else:
                col = 'subscribers'
            
            if col in tribe_data.columns:
                median = tribe_data[col].median()
                if median > 1e9:
                    print(f"      • Median {feat}: {median/1e9:.1f}B")
                elif median > 1e6:
                    print(f"      • Median {feat}: {median/1e6:.1f}M")
                else:
                    print(f"      • Median {feat}: {median:,.0f}")
        
        if examples:
            print(f"      • Examples: {', '.join(str(e) for e in examples[:3])}")
    
    # Visualize with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df['tribe'], cmap='viridis', alpha=0.6, s=50)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.set_title('Hidden YouTube Channel Tribes\n(Clustered by behavioral patterns, not categories)')
    plt.colorbar(scatter, label='Tribe')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hidden_tribes.png', dpi=150)
    plt.close()
    print(f"\n   📊 Visualization saved: {OUTPUT_DIR / 'hidden_tribes.png'}")
    
    return df


# =============================================================================
# UNCONVENTIONAL QUESTION 2: What is the "efficiency" of a channel?
# Views per video, engagement per subscriber - finding over/underperformers
# =============================================================================

def analyze_channel_efficiency(df):
    """
    Question: Which channels are "efficient" - getting outsized results vs effort?
    Normal analysts look at raw numbers - we calculate efficiency ratios.
    """
    print("\n" + "="*70)
    print("🔍 QUESTION 2: Which channels are mysteriously 'efficient' or 'inefficient'?")
    print("   (Finding channels that defy expectations)")
    print("="*70)
    
    df = df.copy()
    
    # Calculate efficiency metrics
    if 'video views' in df.columns and 'subscribers' in df.columns:
        df['views_per_sub'] = df['video views'] / df['subscribers'].replace(0, np.nan)
    elif 'Video Views' in df.columns and 'Subscribers' in df.columns:
        df['views_per_sub'] = df['Video Views'] / df['Subscribers'].replace(0, np.nan)
    
    vid_col = 'video_count' if 'video_count' in df.columns else 'Video Count'
    view_col = 'video views' if 'video views' in df.columns else 'Video Views'
    
    if vid_col in df.columns and view_col in df.columns:
        df['views_per_video'] = df[view_col] / df[vid_col].replace(0, np.nan)
    
    # Find anomalies using IQR
    efficiency_cols = ['views_per_sub', 'views_per_video']
    existing_cols = [c for c in efficiency_cols if c in df.columns]
    
    if not existing_cols:
        print("   Not enough data for efficiency analysis")
        return df
    
    for col in existing_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df[f'{col}_zscore'] = stats.zscore(df[col].fillna(0))
    
    # Find over-performers (high efficiency outliers)
    print("\n   🚀 OVER-PERFORMERS (mysteriously efficient):")
    name_col = 'Youtuber' if 'Youtuber' in df.columns else 'Youtube Channel'
    
    if 'views_per_video' in df.columns and name_col in df.columns:
        top_efficient = df.nlargest(5, 'views_per_video')
        for _, row in top_efficient.iterrows():
            vpv = row['views_per_video']
            if vpv > 1e9:
                vpv_str = f"{vpv/1e9:.1f}B"
            elif vpv > 1e6:
                vpv_str = f"{vpv/1e6:.1f}M"
            else:
                vpv_str = f"{vpv:,.0f}"
            print(f"      • {row[name_col]}: {vpv_str} views/video")
    
    # Find under-performers (low efficiency despite high subs)
    print("\n   🐢 UNDER-PERFORMERS (high subs, low engagement):")
    
    sub_col = 'subscribers' if 'subscribers' in df.columns else 'Subscribers'
    if 'views_per_sub' in df.columns and sub_col in df.columns:
        # High subs but low views per sub
        high_subs = df[df[sub_col] > df[sub_col].median()]
        under = high_subs.nsmallest(5, 'views_per_sub')
        for _, row in under.iterrows():
            vps = row['views_per_sub']
            print(f"      • {row[name_col]}: {vps:.1f} views per subscriber")
    
    # Visualize efficiency landscape
    if 'views_per_video' in df.columns and 'views_per_sub' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.log1p(df['views_per_video'].fillna(0))
        y = np.log1p(df['views_per_sub'].fillna(0))
        
        scatter = ax.scatter(x, y, alpha=0.5, s=30, c='steelblue')
        ax.set_xlabel('Log(Views per Video) - Content Efficiency')
        ax.set_ylabel('Log(Views per Subscriber) - Audience Reach')
        ax.set_title('YouTube Channel Efficiency Landscape\n(Top right = most efficient)')
        
        # Mark extreme outliers
        threshold = df['views_per_video'].quantile(0.95)
        outliers = df[df['views_per_video'] > threshold]
        if name_col in outliers.columns:
            for _, row in outliers.head(5).iterrows():
                ax.annotate(str(row[name_col])[:15], 
                           (np.log1p(row['views_per_video']), np.log1p(row['views_per_sub'])),
                           fontsize=8)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'channel_efficiency.png', dpi=150)
        plt.close()
        print(f"\n   📊 Visualization saved: {OUTPUT_DIR / 'channel_efficiency.png'}")
    
    return df


# =============================================================================
# UNCONVENTIONAL QUESTION 3: What makes a video title "magnetic"?
# NLP analysis on trending video titles
# =============================================================================

def analyze_title_magnetism(df):
    """
    Question: What linguistic patterns make video titles irresistible?
    Beyond basic keywords - analyzing structure, emotion, and psychology.
    """
    print("\n" + "="*70)
    print("🔍 QUESTION 3: What makes a video title 'magnetic' (irresistible)?")
    print("   (NLP patterns in trending video titles)")
    print("="*70)
    
    if 'title' not in df.columns:
        print("   No title data available")
        return
    
    titles = df['title'].dropna().astype(str).tolist()
    
    # Title length analysis
    df['title_length'] = df['title'].astype(str).apply(len)
    df['word_count'] = df['title'].astype(str).apply(lambda x: len(x.split()))
    
    print(f"\n   📏 TITLE LENGTH PATTERNS:")
    print(f"      • Average title length: {df['title_length'].mean():.0f} characters")
    print(f"      • Average word count: {df['word_count'].mean():.1f} words")
    
    # Psychological triggers
    triggers = {
        'urgency': ['now', 'today', 'breaking', 'live', 'urgent', 'just'],
        'numbers': r'\d+',
        'questions': ['?', 'how', 'why', 'what', 'which', 'who'],
        'emotion': ['amazing', 'shocking', 'incredible', 'unbelievable', 'insane', 'crazy', 'best', 'worst'],
        'personal': ['you', 'your', 'i ', "i'", 'my', 'we'],
        'capitalized': r'^[A-Z\s]{10,}',  # ALL CAPS sections
        'emojis': r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]',
        'controversy': ['vs', 'fight', 'exposed', 'truth', 'real', 'fake', 'scam', 'drama']
    }
    
    print(f"\n   🧠 PSYCHOLOGICAL TRIGGERS IN TITLES:")
    trigger_counts = {}
    
    for trigger, pattern in triggers.items():
        if isinstance(pattern, list):
            count = sum(1 for t in titles if any(p in t.lower() for p in pattern))
        else:
            count = sum(1 for t in titles if re.search(pattern, t))
        
        pct = count / len(titles) * 100
        trigger_counts[trigger] = pct
        print(f"      • {trigger.upper()}: {pct:.1f}% of titles")
    
    # TF-IDF analysis to find unique patterns
    print(f"\n   🔤 UNIQUE WORD PATTERNS (TF-IDF Top Terms):")
    
    try:
        # Filter for English-looking titles
        english_titles = [t for t in titles if re.match(r'^[a-zA-Z0-9\s\.\,\!\?\-\:\'\"\|]+$', t[:20])]
        
        if len(english_titles) > 100:
            vectorizer = TfidfVectorizer(max_features=50, stop_words='english', 
                                         ngram_range=(1, 2), min_df=5)
            tfidf = vectorizer.fit_transform(english_titles)
            
            feature_names = vectorizer.get_feature_names_out()
            avg_scores = tfidf.mean(axis=0).A1
            top_indices = avg_scores.argsort()[-15:][::-1]
            
            print("      Top distinctive phrases:")
            for i in top_indices[:10]:
                print(f"         - '{feature_names[i]}' (score: {avg_scores[i]:.3f})")
    except Exception as e:
        print(f"      Could not perform TF-IDF: {e}")
    
    # Title patterns that correlate with high views
    if 'views' in df.columns or 'view_count' in df.columns:
        view_col = 'views' if 'views' in df.columns else 'view_count'
        
        print(f"\n   📈 TITLE PATTERNS VS VIEWS:")
        
        # Compare metrics between high and low performing videos
        median_views = df[view_col].median()
        high_views = df[df[view_col] > median_views]
        low_views = df[df[view_col] <= median_views]
        
        print(f"      High-performing titles:")
        print(f"         - Avg length: {high_views['title_length'].mean():.0f} chars")
        print(f"         - Avg words: {high_views['word_count'].mean():.1f}")
        
        print(f"      Low-performing titles:")
        print(f"         - Avg length: {low_views['title_length'].mean():.0f} chars")
        print(f"         - Avg words: {low_views['word_count'].mean():.1f}")
    
    # Visualize trigger distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    triggers_sorted = sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)
    names = [t[0] for t in triggers_sorted]
    values = [t[1] for t in triggers_sorted]
    
    bars = ax.barh(names, values, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(names))))
    ax.set_xlabel('Percentage of Titles')
    ax.set_title('Psychological Triggers in YouTube Trending Titles')
    
    for bar, val in zip(bars, values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
                va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'title_triggers.png', dpi=150)
    plt.close()
    print(f"\n   📊 Visualization saved: {OUTPUT_DIR / 'title_triggers.png'}")


# =============================================================================
# UNCONVENTIONAL QUESTION 4: Is there a "YouTube ecosystem" structure?
# Network analysis of channel co-occurrence in trending
# =============================================================================

def analyze_trending_ecosystem(df):
    """
    Question: Which channels form an "ecosystem" - trending together?
    Finding hidden competitive or complementary relationships.
    """
    print("\n" + "="*70)
    print("🔍 QUESTION 4: What is the hidden 'ecosystem' structure of YouTube?")
    print("   (Which channels trend together - co-occurrence network)")
    print("="*70)
    
    if 'channel_title' not in df.columns or 'trending_date' not in df.columns:
        print("   Need channel_title and trending_date for ecosystem analysis")
        return
    
    # Find channels that appear on the same trending day
    df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m', errors='coerce')
    
    # Get co-occurrences per day
    cooccurrence = Counter()
    
    for date in df['trending_date'].dropna().unique():
        day_channels = df[df['trending_date'] == date]['channel_title'].unique()
        # Create pairs
        for i, ch1 in enumerate(day_channels):
            for ch2 in day_channels[i+1:]:
                pair = tuple(sorted([ch1, ch2]))
                cooccurrence[pair] += 1
    
    print(f"\n   Found {len(cooccurrence)} channel pairs that co-occurred in trending")
    
    # Top pairs
    print(f"\n   🤝 TOP CHANNEL PAIRS (Frequently trending together):")
    for pair, count in cooccurrence.most_common(15):
        print(f"      {count}x: {pair[0][:25]} <-> {pair[1][:25]}")
    
    # Find channel clusters using co-occurrence
    top_channels = df['channel_title'].value_counts().head(50).index.tolist()
    
    # Build adjacency matrix for top channels
    adj_matrix = np.zeros((len(top_channels), len(top_channels)))
    
    for (ch1, ch2), count in cooccurrence.items():
        if ch1 in top_channels and ch2 in top_channels:
            i, j = top_channels.index(ch1), top_channels.index(ch2)
            adj_matrix[i, j] = count
            adj_matrix[j, i] = count
    
    # Cluster channels based on co-occurrence patterns
    from sklearn.cluster import SpectralClustering
    
    if adj_matrix.sum() > 0:
        # Normalize
        adj_normalized = adj_matrix / (adj_matrix.max() + 1)
        
        try:
            clustering = SpectralClustering(n_clusters=5, affinity='precomputed', 
                                           random_state=42, assign_labels='kmeans')
            labels = clustering.fit_predict(adj_normalized + 0.01)  # Add small value for stability
            
            print(f"\n   🌐 ECOSYSTEM CLUSTERS (Channels that trend together):")
            for cluster in range(5):
                cluster_channels = [top_channels[i] for i in range(len(labels)) if labels[i] == cluster]
                if cluster_channels:
                    print(f"\n      Cluster {cluster + 1}:")
                    for ch in cluster_channels[:5]:
                        print(f"         • {ch[:40]}")
                    if len(cluster_channels) > 5:
                        print(f"         ... and {len(cluster_channels) - 5} more")
        except Exception as e:
            print(f"   Could not perform spectral clustering: {e}")
    
    # Visualize network
    try:
        import networkx as nx
        
        G = nx.Graph()
        
        # Add edges for top co-occurrences
        for (ch1, ch2), count in cooccurrence.most_common(100):
            if count >= 5:  # Only strong connections
                G.add_edge(ch1[:20], ch2[:20], weight=count)
        
        if len(G.edges()) > 0:
            fig, ax = plt.subplots(figsize=(15, 15))
            
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            
            nx.draw_networkx_nodes(G, pos, node_size=100, node_color='steelblue', alpha=0.7, ax=ax)
            nx.draw_networkx_edges(G, pos, width=[w/max(weights)*3 for w in weights], 
                                  alpha=0.3, edge_color='gray', ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)
            
            ax.set_title('YouTube Trending Ecosystem\n(Channels that frequently trend together)')
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'trending_ecosystem.png', dpi=150)
            plt.close()
            print(f"\n   📊 Visualization saved: {OUTPUT_DIR / 'trending_ecosystem.png'}")
    except ImportError:
        print("   NetworkX not available for visualization")


# =============================================================================
# UNCONVENTIONAL QUESTION 5: Are there temporal anomalies in the data?
# Time-series anomaly detection
# =============================================================================

def detect_temporal_anomalies(df):
    """
    Question: Are there suspicious patterns or "manufactured" trends?
    Looking for statistical anomalies in view/engagement patterns.
    """
    print("\n" + "="*70)
    print("🔍 QUESTION 5: Are there suspicious anomalies in the data?")
    print("   (Statistical outliers that might indicate unusual activity)")
    print("="*70)
    
    # Use Isolation Forest for anomaly detection
    numeric_cols = []
    
    for col in ['views', 'view_count', 'likes', 'dislikes', 'comment_count']:
        if col in df.columns:
            numeric_cols.append(col)
    
    if len(numeric_cols) < 2:
        print("   Not enough numeric columns for anomaly detection")
        return
    
    # Prepare feature matrix
    X = df[numeric_cols].apply(lambda x: np.log1p(x.fillna(0)))
    X = X.replace([np.inf, -np.inf], 0)
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
    df['anomaly_score'] = iso_forest.fit_predict(X)
    df['anomaly_raw_score'] = iso_forest.decision_function(X)
    
    anomalies = df[df['anomaly_score'] == -1]
    
    print(f"\n   Found {len(anomalies)} anomalous entries ({len(anomalies)/len(df)*100:.1f}%)")
    
    # Analyze anomalies
    print(f"\n   🚨 ANOMALY CHARACTERISTICS:")
    
    for col in numeric_cols:
        normal_median = df[df['anomaly_score'] == 1][col].median()
        anomaly_median = anomalies[col].median()
        ratio = anomaly_median / max(normal_median, 1)
        print(f"      • {col}: Anomalies are {ratio:.1f}x the normal median")
    
    # Show top anomalies
    print(f"\n   🔴 TOP ANOMALIES:")
    top_anomalies = anomalies.nsmallest(10, 'anomaly_raw_score')
    
    title_col = 'title' if 'title' in df.columns else None
    channel_col = 'channel_title' if 'channel_title' in df.columns else None
    
    for _, row in top_anomalies.head(5).iterrows():
        info = []
        if title_col:
            info.append(f"'{str(row[title_col])[:40]}...'")
        if channel_col:
            info.append(f"by {row[channel_col]}")
        
        metrics = []
        for col in numeric_cols[:3]:
            val = row[col]
            if val > 1e6:
                metrics.append(f"{col}: {val/1e6:.1f}M")
            else:
                metrics.append(f"{col}: {val:,.0f}")
        
        print(f"      • {' '.join(info)}")
        print(f"        {', '.join(metrics)}")
    
    # Visualize anomalies
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.log1p(df[numeric_cols[0]].fillna(0))
        y = np.log1p(df[numeric_cols[1]].fillna(0))
        
        normal = df['anomaly_score'] == 1
        anomaly = df['anomaly_score'] == -1
        
        ax.scatter(x[normal], y[normal], c='steelblue', alpha=0.3, s=10, label='Normal')
        ax.scatter(x[anomaly], y[anomaly], c='red', alpha=0.7, s=50, label='Anomaly', marker='x')
        
        ax.set_xlabel(f'Log({numeric_cols[0]})')
        ax.set_ylabel(f'Log({numeric_cols[1]})')
        ax.set_title('Anomaly Detection in YouTube Data\n(Red X = Statistical Outliers)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'anomaly_detection.png', dpi=150)
        plt.close()
        print(f"\n   📊 Visualization saved: {OUTPUT_DIR / 'anomaly_detection.png'}")


# =============================================================================
# UNCONVENTIONAL QUESTION 6: Geographic power distribution
# Are certain countries "over-represented" in YouTube success?
# =============================================================================

def analyze_geographic_power(df, trending_df=None):
    """
    Question: Is YouTube success distributed fairly, or concentrated in certain regions?
    Looking for power law distributions and geographic biases.
    """
    print("\n" + "="*70)
    print("🔍 QUESTION 6: Is YouTube success geographically concentrated?")
    print("   (Power distribution and regional biases)")
    print("="*70)
    
    country_col = None
    for col in ['Country', 'country', 'Country of origin']:
        if col in df.columns:
            country_col = col
            break
    
    if not country_col:
        print("   No country data available in channel dataset")
        
        # Try trending data
        if trending_df is not None and 'country' in trending_df.columns:
            print("   Using trending video country distribution instead")
            country_counts = trending_df['country'].value_counts()
            
            print(f"\n   📍 TRENDING VIDEOS BY COUNTRY:")
            for country, count in country_counts.items():
                pct = count / len(trending_df) * 100
                print(f"      {country}: {count:,} videos ({pct:.1f}%)")
        return
    
    # Analyze country distribution
    country_counts = df[country_col].value_counts()
    
    print(f"\n   📍 TOP YOUTUBE CHANNEL COUNTRIES:")
    for country, count in country_counts.head(10).items():
        pct = count / len(df) * 100
        print(f"      {country}: {count} channels ({pct:.1f}%)")
    
    # Check for power law distribution
    counts = country_counts.values
    
    # Gini coefficient (inequality measure)
    n = len(counts)
    if n > 1:
        sorted_counts = np.sort(counts)
        cumulative = np.cumsum(sorted_counts) / sorted_counts.sum()
        gini = 1 - 2 * np.sum(cumulative) / n
        
        print(f"\n   📊 INEQUALITY ANALYSIS:")
        print(f"      • Gini coefficient: {gini:.3f} (0=equal, 1=unequal)")
        
        top_3_share = country_counts.head(3).sum() / len(df) * 100
        print(f"      • Top 3 countries control: {top_3_share:.1f}% of top channels")
        
        if gini > 0.6:
            print(f"      ⚠️  HIGH CONCENTRATION: YouTube success is heavily geographically concentrated")
        elif gini > 0.4:
            print(f"      📌 MODERATE CONCENTRATION: Some geographic bias exists")
        else:
            print(f"      ✓ RELATIVELY DISTRIBUTED: Success is spread across countries")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    top_10 = country_counts.head(10)
    axes[0].barh(range(len(top_10)), top_10.values, color=plt.cm.viridis(np.linspace(0.2, 0.8, 10)))
    axes[0].set_yticks(range(len(top_10)))
    axes[0].set_yticklabels(top_10.index)
    axes[0].set_xlabel('Number of Top Channels')
    axes[0].set_title('Geographic Distribution of YouTube Success')
    axes[0].invert_yaxis()
    
    # Lorenz curve (inequality visualization)
    sorted_counts = np.sort(counts)[::-1]
    cumulative_share = np.cumsum(sorted_counts) / sorted_counts.sum()
    country_share = np.arange(1, len(counts) + 1) / len(counts)
    
    axes[1].plot(country_share, cumulative_share, 'b-', linewidth=2, label='Actual')
    axes[1].plot([0, 1], [0, 1], 'r--', label='Perfect Equality')
    axes[1].fill_between(country_share, cumulative_share, country_share, alpha=0.3)
    axes[1].set_xlabel('Cumulative Share of Countries')
    axes[1].set_ylabel('Cumulative Share of Channels')
    axes[1].set_title(f'Lorenz Curve (Gini = {gini:.3f})')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'geographic_power.png', dpi=150)
    plt.close()
    print(f"\n   📊 Visualization saved: {OUTPUT_DIR / 'geographic_power.png'}")


# =============================================================================
# UNCONVENTIONAL QUESTION 7: Evolution of "started" year patterns
# Do newer channels need more or less content to succeed?
# =============================================================================

def analyze_era_patterns(df):
    """
    Question: Has the "recipe for success" changed over YouTube eras?
    Comparing what it takes to succeed in different time periods.
    """
    print("\n" + "="*70)
    print("🔍 QUESTION 7: How has the 'recipe for success' evolved?")
    print("   (Comparing success patterns across YouTube eras)")
    print("="*70)
    
    year_col = None
    for col in ['started', 'Started', 'Channel Started', 'created_year']:
        if col in df.columns:
            year_col = col
            break
    
    if not year_col:
        print("   No 'started' year data available")
        return
    
    df = df.copy()
    df['year'] = pd.to_numeric(df[year_col], errors='coerce')
    df = df[df['year'].between(2005, 2025)]
    
    if len(df) < 50:
        print(f"   Not enough data with year information")
        return
    
    # Define eras
    era_bins = [2005, 2010, 2015, 2020, 2026]
    era_labels = ['2005-2009\n(Pioneer)', '2010-2014\n(Growth)', '2015-2019\n(Mature)', '2020+\n(Modern)']
    df['era'] = pd.cut(df['year'], bins=era_bins, labels=era_labels, include_lowest=True)
    
    print(f"\n   📅 CHANNELS BY ERA:")
    era_counts = df['era'].value_counts().sort_index()
    for era, count in era_counts.items():
        print(f"      {era.replace(chr(10), ' ')}: {count} channels")
    
    # Compare metrics by era
    print(f"\n   📈 SUCCESS METRICS BY ERA:")
    
    # Find available metric columns
    sub_col = 'subscribers' if 'subscribers' in df.columns else 'Subscribers'
    view_col = 'video views' if 'video views' in df.columns else 'Video Views'
    vid_col = 'video_count' if 'video_count' in df.columns else 'Video Count'
    
    era_stats = df.groupby('era', observed=True).agg({
        sub_col: 'median' if sub_col in df.columns else None,
        view_col: 'median' if view_col in df.columns else None,
        vid_col: 'median' if vid_col in df.columns else None
    }).dropna(axis=1)
    
    for col in era_stats.columns:
        print(f"\n      {col}:")
        for era in era_stats.index:
            val = era_stats.loc[era, col]
            if val > 1e9:
                print(f"         {era.replace(chr(10), ' ')}: {val/1e9:.1f}B")
            elif val > 1e6:
                print(f"         {era.replace(chr(10), ' ')}: {val/1e6:.1f}M")
            else:
                print(f"         {era.replace(chr(10), ' ')}: {val:,.0f}")
    
    # Calculate "effort required" (videos per million subscribers)
    if sub_col in df.columns and vid_col in df.columns:
        df['effort'] = df[vid_col] / (df[sub_col] / 1e6).replace(0, np.nan)
        
        print(f"\n   💪 EFFORT REQUIRED (Videos per Million Subscribers):")
        effort_by_era = df.groupby('era', observed=True)['effort'].median()
        for era, effort in effort_by_era.items():
            if pd.notna(effort):
                print(f"      {era.replace(chr(10), ' ')}: {effort:.0f} videos/M subs")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Era distribution
    era_counts.plot(kind='bar', ax=axes[0], color=plt.cm.viridis(np.linspace(0.2, 0.8, 4)))
    axes[0].set_title('Top Channels by Era')
    axes[0].set_xlabel('Era')
    axes[0].set_ylabel('Number of Channels')
    axes[0].tick_params(axis='x', rotation=0)
    
    # Subscriber distribution by era
    if sub_col in df.columns:
        df.boxplot(column=sub_col, by='era', ax=axes[1])
        axes[1].set_yscale('log')
        axes[1].set_title('Subscriber Distribution by Era')
        axes[1].set_xlabel('Era')
        axes[1].set_ylabel('Subscribers (log scale)')
        plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'era_patterns.png', dpi=150)
    plt.close()
    print(f"\n   📊 Visualization saved: {OUTPUT_DIR / 'era_patterns.png'}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("\n" + "🔬"*35)
    print("   ADVANCED NON-PREDICTIVE ML ANALYSIS ON YOUTUBE DATA")
    print("   Asking questions normal analysts wouldn't ask")
    print("🔬"*35)
    
    # Load all data
    print("\n📂 Loading datasets...")
    data = load_all_data()
    
    if not data:
        print("No data available for analysis!")
        return
    
    # Run analyses
    
    # Question 1: Hidden tribes
    if 'global' in data:
        discover_channel_tribes(data['global'])
    elif 'top_channels' in data:
        discover_channel_tribes(data['top_channels'])
    
    # Question 2: Channel efficiency
    if 'global' in data:
        analyze_channel_efficiency(data['global'])
    
    # Question 3: Title magnetism
    if 'trending' in data:
        analyze_title_magnetism(data['trending'])
    
    # Question 4: Trending ecosystem
    if 'trending' in data:
        analyze_trending_ecosystem(data['trending'])
    
    # Question 5: Anomaly detection
    if 'trending' in data:
        detect_temporal_anomalies(data['trending'])
    
    # Question 6: Geographic power
    if 'global' in data:
        analyze_geographic_power(data['global'], data.get('trending'))
    
    # Question 7: Era patterns
    if 'global' in data:
        analyze_era_patterns(data['global'])
    elif 'top_channels' in data:
        analyze_era_patterns(data['top_channels'])
    
    # Summary
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n   All visualizations saved to: {OUTPUT_DIR.absolute()}")
    print("\n   Questions Explored:")
    print("   1. Hidden channel 'tribes' (behavioral clustering)")
    print("   2. Channel efficiency (over/under-performers)")
    print("   3. Title magnetism (psychological triggers)")
    print("   4. Trending ecosystem (co-occurrence network)")
    print("   5. Statistical anomalies (outlier detection)")
    print("   6. Geographic power distribution")
    print("   7. Era-based success patterns")


if __name__ == "__main__":
    main()
