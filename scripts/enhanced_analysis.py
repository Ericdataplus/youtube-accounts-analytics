"""
🔬 Enhanced Analysis with Multi-Platform Influencer Data
Cross-platform comparison and deeper channel insights
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
OUTPUT_DIR = Path("data/processed/analysis")


def load_influencer_data():
    """Load all influencer data from multiple countries"""
    influencer_dir = Path("data/raw/influencers_2024/Top 100 Influencers")
    
    youtube_dfs = []
    
    for country_dir in influencer_dir.iterdir():
        if country_dir.is_dir():
            yt_file = country_dir / f"youtube_data_{country_dir.name}.csv"
            if yt_file.exists():
                try:
                    df = pd.read_csv(yt_file)
                    df['country'] = country_dir.name
                    youtube_dfs.append(df)
                except:
                    pass
    
    if youtube_dfs:
        return pd.concat(youtube_dfs, ignore_index=True)
    return None


def analyze_cross_country_patterns(df):
    """
    Question: Which countries punch above their weight in YouTube influence?
    """
    print("\n" + "="*70)
    print("🌍 CROSS-COUNTRY YOUTUBE INFLUENCE ANALYSIS")
    print("="*70)
    
    if df is None or len(df) == 0:
        print("   No data available")
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
    
    # Country analysis
    country_stats = df.groupby('country').agg({
        'followers_num': ['sum', 'mean', 'count']
    }).round(0)
    country_stats.columns = ['total_followers', 'avg_followers', 'num_influencers']
    country_stats = country_stats.sort_values('total_followers', ascending=False)
    
    print(f"\n   📊 TOP 15 COUNTRIES BY TOTAL YOUTUBE INFLUENCE:")
    for i, (country, row) in enumerate(country_stats.head(15).iterrows()):
        total = row['total_followers']
        if total > 1e9:
            total_str = f"{total/1e9:.1f}B"
        else:
            total_str = f"{total/1e6:.0f}M"
        print(f"      {i+1:2d}. {country:20s}: {total_str:>8} followers ({int(row['num_influencers'])} influencers)")
    
    # Countries with highest AVERAGE (punching above weight)
    print(f"\n   🏆 COUNTRIES WITH HIGHEST AVG FOLLOWERS (Punching Above Weight):")
    avg_sorted = country_stats[country_stats['num_influencers'] >= 5].sort_values('avg_followers', ascending=False)
    for i, (country, row) in enumerate(avg_sorted.head(10).iterrows()):
        avg = row['avg_followers']
        if avg > 1e6:
            avg_str = f"{avg/1e6:.1f}M"
        else:
            avg_str = f"{avg/1e3:.0f}K"
        print(f"      {i+1:2d}. {country:20s}: {avg_str:>8} avg followers")
    
    # Topic analysis
    if 'TOPIC OF INFLUENCE' in df.columns:
        print(f"\n   📑 TOP YOUTUBE INFLUENCE TOPICS:")
        topic_counts = df['TOPIC OF INFLUENCE'].value_counts().head(15)
        for topic, count in topic_counts.items():
            print(f"      • {topic}: {count} influencers")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Top countries
    top_countries = country_stats.head(15)
    y_pos = range(len(top_countries))
    axes[0].barh(y_pos, top_countries['total_followers'] / 1e9, color=plt.cm.viridis(np.linspace(0.2, 0.8, 15)))
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(top_countries.index)
    axes[0].set_xlabel('Total Followers (Billions)')
    axes[0].set_title('Top 15 Countries by YouTube Influence')
    axes[0].invert_yaxis()
    
    # Topics
    if 'TOPIC OF INFLUENCE' in df.columns:
        top_topics = df['TOPIC OF INFLUENCE'].value_counts().head(10)
        axes[1].barh(range(len(top_topics)), top_topics.values, color='steelblue')
        axes[1].set_yticks(range(len(top_topics)))
        axes[1].set_yticklabels(top_topics.index)
        axes[1].set_xlabel('Number of Influencers')
        axes[1].set_title('Top YouTube Influence Topics')
        axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'country_influence.png', dpi=150)
    plt.close()
    print(f"\n   📊 Saved: {OUTPUT_DIR / 'country_influence.png'}")
    
    return df


def compare_all_channel_datasets():
    """
    Question: How do different datasets capture YouTube differently?
    """
    print("\n" + "="*70)
    print("🔍 DATASET COMPARISON: Same Channels, Different Stories")
    print("="*70)
    
    datasets = {}
    
    # Load all datasets
    try:
        datasets['Global Stats'] = pd.read_csv('data/raw/global_stats/Global YouTube Statistics.csv', encoding='latin-1')
    except: pass
    
    try:
        datasets['Top Channels'] = pd.read_csv('data/raw/top_channels/most_subscribed_youtube_channels.csv')
    except: pass
    
    try:
        datasets['GitHub Top'] = pd.read_csv('data/raw/github_channel_growth/data/topSubscribed.csv')
    except: pass
    
    try:
        datasets['Influencers 2024'] = pd.read_csv('data/raw/influencers_2024/Top 100 Influencers/all-countries/youtube_data_all-countries.csv')
    except: pass
    
    print(f"\n   📊 DATASET OVERVIEW:")
    for name, df in datasets.items():
        print(f"      • {name}: {len(df)} entries, {len(df.columns)} columns")
    
    # Find common channels
    all_names = set()
    for name, df in datasets.items():
        for col in ['Youtuber', 'Youtube Channel', 'NAME', 'channel_name']:
            if col in df.columns:
                names = set(df[col].dropna().str.lower().str.strip())
                print(f"      - {name} has {len(names)} unique channel names")
                all_names.update(names)
                break
    
    print(f"\n   Total unique channels across all datasets: {len(all_names)}")
    
    return datasets


def analyze_engagement_anomalies():
    """
    Question: Are there suspicious engagement patterns?
    """
    print("\n" + "="*70)
    print("🚨 ENGAGEMENT ANOMALY DETECTION")
    print("="*70)
    
    try:
        df = pd.read_csv('data/raw/global_stats/Global YouTube Statistics.csv', encoding='latin-1')
    except:
        print("   Could not load global stats")
        return
    
    # Calculate engagement rate anomalies
    if 'video views' in df.columns and 'subscribers' in df.columns:
        df['engagement_ratio'] = df['video views'] / df['subscribers'].replace(0, np.nan)
        
        # Z-score based anomalies
        from scipy import stats
        df['engagement_zscore'] = stats.zscore(df['engagement_ratio'].fillna(0))
        
        print(f"\n   🔴 HIGH ENGAGEMENT ANOMALIES (views/subscriber ratio):")
        high_anomalies = df.nlargest(10, 'engagement_zscore')
        for _, row in high_anomalies.iterrows():
            ratio = row['engagement_ratio']
            print(f"      • {row['Youtuber']}: {ratio:.0f}x views per subscriber")
        
        print(f"\n   🔵 LOW ENGAGEMENT ANOMALIES (high subs, low engagement):")
        # High subscribers but low views per sub
        high_subs = df[df['subscribers'] > df['subscribers'].median()]
        low_eng = high_subs.nsmallest(10, 'engagement_ratio')
        for _, row in low_eng.iterrows():
            ratio = row['engagement_ratio']
            subs = row['subscribers']
            if subs > 1e6:
                subs_str = f"{subs/1e6:.0f}M"
            else:
                subs_str = f"{subs/1e3:.0f}K"
            print(f"      • {row['Youtuber']}: {subs_str} subs, only {ratio:.1f}x views/sub")


def analyze_category_dynamics():
    """
    Question: Which categories are growing vs declining?
    """
    print("\n" + "="*70)
    print("📈 CATEGORY ECOSYSTEM ANALYSIS")
    print("="*70)
    
    try:
        df = pd.read_csv('data/raw/global_stats/Global YouTube Statistics.csv', encoding='latin-1')
    except:
        print("   Could not load data")
        return
    
    if 'category' not in df.columns:
        print("   No category data available")
        return
    
    # Category statistics
    category_stats = df.groupby('category').agg({
        'subscribers': ['sum', 'mean', 'count'],
        'video views': ['sum', 'mean']
    }).round(0)
    category_stats.columns = ['total_subs', 'avg_subs', 'num_channels', 'total_views', 'avg_views']
    category_stats = category_stats.sort_values('total_subs', ascending=False)
    
    print(f"\n   📊 CATEGORY POWER RANKINGS:")
    for i, (cat, row) in enumerate(category_stats.head(15).iterrows()):
        total = row['total_subs']
        if total > 1e9:
            total_str = f"{total/1e9:.1f}B"
        else:
            total_str = f"{total/1e6:.0f}M"
        n = int(row['num_channels'])
        print(f"      {i+1:2d}. {str(cat):20s}: {total_str:>8} subs ({n} channels)")
    
    # Efficiency by category
    category_stats['views_per_sub'] = category_stats['total_views'] / category_stats['total_subs']
    
    print(f"\n   ⚡ MOST EFFICIENT CATEGORIES (Views per Subscriber):")
    efficient = category_stats.sort_values('views_per_sub', ascending=False)
    for i, (cat, row) in enumerate(efficient.head(10).iterrows()):
        vps = row['views_per_sub']
        print(f"      {i+1:2d}. {str(cat):20s}: {vps:.0f}x views per subscriber")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(14, 8))
    
    top_cats = category_stats.head(12)
    x = np.arange(len(top_cats))
    width = 0.35
    
    ax.bar(x - width/2, top_cats['total_subs']/1e9, width, label='Total Subscribers (B)', color='steelblue')
    ax.bar(x + width/2, top_cats['total_views']/1e12, width, label='Total Views (T)', color='coral')
    
    ax.set_ylabel('Billions / Trillions')
    ax.set_xlabel('Category')
    ax.set_title('YouTube Category Power Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels([str(c)[:12] for c in top_cats.index], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'category_power.png', dpi=150)
    plt.close()
    print(f"\n   📊 Saved: {OUTPUT_DIR / 'category_power.png'}")


def main():
    print("\n" + "🔬"*35)
    print("   ENHANCED YOUTUBE ANALYTICS WITH MULTI-PLATFORM DATA")
    print("🔬"*35)
    
    # Load influencer data
    print("\n📂 Loading influencer data...")
    influencer_df = load_influencer_data()
    if influencer_df is not None:
        print(f"   ✓ Loaded {len(influencer_df)} influencer records from multiple countries")
    
    # Analysis 1: Cross-country patterns
    analyze_cross_country_patterns(influencer_df)
    
    # Analysis 2: Dataset comparison
    compare_all_channel_datasets()
    
    # Analysis 3: Engagement anomalies
    analyze_engagement_anomalies()
    
    # Analysis 4: Category dynamics
    analyze_category_dynamics()
    
    print("\n" + "="*70)
    print("✅ ENHANCED ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n   All outputs saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
