"""
🔬 Specialized Analysis: Time-Series Forecasting & NLP Deep-Dive
Advanced analytics for YouTube channel and content analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
OUTPUT_DIR = Path("data/processed/analysis/specialized")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load all datasets"""
    data = {}
    
    try:
        data['global'] = pd.read_csv('data/raw/global_stats/Global YouTube Statistics.csv', encoding='latin-1')
        print(f"✓ Global: {len(data['global'])} channels")
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
    
    return data


# =============================================================================
# PART 1: TIME-SERIES ANALYSIS
# =============================================================================

def time_series_trending_analysis(df):
    """
    Time-series decomposition of trending video patterns
    """
    print("\n" + "="*70)
    print("📈 TIME-SERIES ANALYSIS: Trending Patterns")
    print("   Decomposing temporal dynamics in YouTube trending")
    print("="*70)
    
    df = df.copy()
    df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m', errors='coerce')
    df = df.dropna(subset=['trending_date'])
    
    # Daily counts
    daily_counts = df.groupby('trending_date').size()
    daily_counts = daily_counts.sort_index()
    
    print(f"\n   📊 TIME RANGE:")
    print(f"      Start: {daily_counts.index.min().strftime('%Y-%m-%d')}")
    print(f"      End: {daily_counts.index.max().strftime('%Y-%m-%d')}")
    print(f"      Days: {len(daily_counts)}")
    
    # Basic statistics
    print(f"\n   📊 DAILY TRENDING VIDEOS:")
    print(f"      Mean: {daily_counts.mean():.0f}")
    print(f"      Std: {daily_counts.std():.0f}")
    print(f"      Min: {daily_counts.min():.0f}")
    print(f"      Max: {daily_counts.max():.0f}")
    
    # Trend detection using linear regression
    X = np.arange(len(daily_counts)).reshape(-1, 1)
    y = daily_counts.values
    
    model = LinearRegression()
    model.fit(X, y)
    
    trend_slope = model.coef_[0]
    trend_direction = "increasing" if trend_slope > 0 else "decreasing"
    
    print(f"\n   📈 TREND ANALYSIS:")
    print(f"      Slope: {trend_slope:.2f} videos/day")
    print(f"      Direction: {trend_direction}")
    
    # Day of week patterns
    df['dow'] = df['trending_date'].dt.dayofweek
    dow_counts = df.groupby('dow').size()
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    print(f"\n   📅 DAY OF WEEK EFFECTS:")
    best_day = dow_names[dow_counts.idxmax()]
    worst_day = dow_names[dow_counts.idxmin()]
    print(f"      Best day: {best_day} ({dow_counts.max()} videos)")
    print(f"      Worst day: {worst_day} ({dow_counts.min()} videos)")
    
    # Peak detection
    peaks, properties = find_peaks(daily_counts.values, height=daily_counts.mean() + daily_counts.std())
    
    print(f"\n   🔝 PEAK DAYS DETECTED: {len(peaks)}")
    if len(peaks) > 0:
        for idx in peaks[:5]:
            date = daily_counts.index[idx]
            count = daily_counts.iloc[idx]
            print(f"      {date.strftime('%Y-%m-%d')}: {count} videos")
    
    # View count time series
    if 'views' in df.columns or 'view_count' in df.columns:
        view_col = 'views' if 'views' in df.columns else 'view_count'
        daily_views = df.groupby('trending_date')[view_col].sum()
        
        print(f"\n   👀 DAILY TOTAL VIEWS:")
        print(f"      Mean: {daily_views.mean()/1e9:.2f}B")
        print(f"      Peak: {daily_views.max()/1e9:.2f}B on {daily_views.idxmax().strftime('%Y-%m-%d')}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Time series with trend
    axes[0, 0].plot(daily_counts.index, daily_counts.values, alpha=0.7, linewidth=0.8)
    trend_line = model.predict(X)
    axes[0, 0].plot(daily_counts.index, trend_line, 'r-', linewidth=2, label='Trend')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Videos Trending')
    axes[0, 0].set_title('Daily Trending Videos with Trend Line')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Day of week
    colors = ['#e74c3c' if i in [5, 6] else '#3498db' for i in range(7)]
    axes[0, 1].bar(dow_names, [dow_counts.get(i, 0) for i in range(7)], color=colors)
    axes[0, 1].set_xlabel('Day of Week')
    axes[0, 1].set_ylabel('Total Videos')
    axes[0, 1].set_title('Trending Videos by Day of Week')
    
    # Rolling average
    rolling = daily_counts.rolling(window=7).mean()
    axes[1, 0].plot(daily_counts.index, daily_counts.values, alpha=0.3, label='Daily')
    axes[1, 0].plot(daily_counts.index, rolling, 'r-', linewidth=2, label='7-day MA')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Videos')
    axes[1, 0].set_title('7-Day Moving Average')
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Views time series
    if 'views' in df.columns or 'view_count' in df.columns:
        daily_views_roll = daily_views.rolling(window=7).mean()
        axes[1, 1].fill_between(daily_views.index, daily_views.values/1e9, alpha=0.3)
        axes[1, 1].plot(daily_views.index, daily_views_roll/1e9, 'r-', linewidth=2)
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Total Views (Billions)')
        axes[1, 1].set_title('Daily Total Views (7-day MA)')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'timeseries_trends.png', dpi=150)
    plt.close()
    print(f"\n   📊 Saved: {OUTPUT_DIR / 'timeseries_trends.png'}")


def category_time_evolution(df):
    """
    How do categories change in popularity over time?
    """
    print("\n" + "="*70)
    print("📈 CATEGORY EVOLUTION OVER TIME")
    print("   Which categories are growing/declining?")
    print("="*70)
    
    df = df.copy()
    df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m', errors='coerce')
    
    cat_col = 'category_id' if 'category_id' in df.columns else None
    if cat_col is None:
        print("   No category data for time evolution")
        return
    
    df['month'] = df['trending_date'].dt.to_period('M')
    
    # Category counts by month
    cat_monthly = df.groupby(['month', cat_col]).size().unstack(fill_value=0)
    
    # Calculate growth rate for each category
    print(f"\n   📊 CATEGORY GROWTH RATES:")
    
    growth_rates = {}
    for cat in cat_monthly.columns:
        if len(cat_monthly[cat]) > 1:
            first_half = cat_monthly[cat].iloc[:len(cat_monthly)//2].mean()
            second_half = cat_monthly[cat].iloc[len(cat_monthly)//2:].mean()
            if first_half > 0:
                growth = (second_half - first_half) / first_half * 100
                growth_rates[cat] = growth
    
    # Sort by growth
    sorted_growth = sorted(growth_rates.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n   🚀 GROWING CATEGORIES:")
    for cat, growth in sorted_growth[:5]:
        print(f"      Category {cat}: +{growth:.1f}%")
    
    print(f"\n   📉 DECLINING CATEGORIES:")
    for cat, growth in sorted_growth[-5:]:
        print(f"      Category {cat}: {growth:.1f}%")
    
    # Visualize top categories over time
    top_cats = cat_monthly.sum().nlargest(6).index
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for cat in top_cats:
        ax.plot(cat_monthly.index.astype(str), cat_monthly[cat], marker='o', 
                label=f'Category {cat}', linewidth=2, markersize=4)
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Trending Videos')
    ax.set_title('Category Popularity Over Time')
    ax.legend(loc='upper right')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'category_evolution.png', dpi=150)
    plt.close()
    print(f"\n   📊 Saved: {OUTPUT_DIR / 'category_evolution.png'}")


def subscriber_growth_model(df):
    """
    Model subscriber growth patterns based on channel age
    """
    print("\n" + "="*70)
    print("📈 SUBSCRIBER GROWTH MODELING")
    print("   Predicting growth trajectories")
    print("="*70)
    
    year_col = 'created_year' if 'created_year' in df.columns else 'started'
    sub_col = 'subscribers' if 'subscribers' in df.columns else 'Subscribers'
    
    if year_col not in df.columns or sub_col not in df.columns:
        print("   Missing required columns")
        return
    
    df = df.copy()
    df['year'] = pd.to_numeric(df[year_col], errors='coerce')
    df = df.dropna(subset=['year', sub_col])
    df = df[df['year'].between(2006, 2022)]
    df = df[df[sub_col] > 0]
    
    current_year = 2024
    df['age'] = current_year - df['year']
    
    # Calculate subscribers per year of existence
    df['growth_rate'] = df[sub_col] / df['age']
    
    # Group by age and calculate statistics
    age_stats = df.groupby('age').agg({
        sub_col: ['mean', 'median', 'std', 'count']
    }).round(0)
    age_stats.columns = ['mean', 'median', 'std', 'count']
    
    print(f"\n   📊 SUBSCRIBER PATTERNS BY CHANNEL AGE:")
    for age in range(1, 16):
        if age in age_stats.index:
            median = age_stats.loc[age, 'median']
            count = int(age_stats.loc[age, 'count'])
            if median > 1e6:
                print(f"      {age:2d} years: {median/1e6:.1f}M median ({count} channels)")
            else:
                print(f"      {age:2d} years: {median/1e3:.0f}K median ({count} channels)")
    
    # Fit growth model: log(subs) = a + b*log(age)
    log_age = np.log(df['age'].values)
    log_subs = np.log(df[sub_col].values)
    
    slope, intercept, r_value, p_value, _ = stats.linregress(log_age, log_subs)
    
    print(f"\n   📈 GROWTH MODEL (Power Law: subs ∝ age^β):")
    print(f"      Exponent (β): {slope:.3f}")
    print(f"      R² value: {r_value**2:.3f}")
    
    if slope > 1:
        print(f"      → Superlinear growth: Older channels grow faster")
    elif slope > 0:
        print(f"      → Sublinear growth: Growth slows with age")
    else:
        print(f"      → Decline: Older channels losing ground")
    
    # Predict future
    print(f"\n   🔮 GROWTH PREDICTIONS:")
    for years in [1, 2, 5]:
        extra_growth = np.exp(slope * np.log(1 + years/10))
        print(f"      +{years} year(s): ~{(extra_growth-1)*100:.1f}% growth")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter with power law fit
    axes[0].scatter(df['age'], df[sub_col], alpha=0.3, s=10)
    ages_fit = np.linspace(1, 18, 100)
    subs_fit = np.exp(intercept + slope * np.log(ages_fit))
    axes[0].plot(ages_fit, subs_fit, 'r-', linewidth=3, label=f'Power Law (β={slope:.2f})')
    axes[0].set_xlabel('Channel Age (years)')
    axes[0].set_ylabel('Subscribers')
    axes[0].set_title('Channel Age vs Subscribers')
    axes[0].set_yscale('log')
    axes[0].legend()
    
    # Median by age
    axes[1].errorbar(age_stats.index, age_stats['median'], 
                     yerr=age_stats['std']/np.sqrt(age_stats['count']),
                     fmt='o-', capsize=3, capthick=1)
    axes[1].set_xlabel('Channel Age (years)')
    axes[1].set_ylabel('Median Subscribers')
    axes[1].set_title('Median Subscribers by Channel Age (with SE)')
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'growth_model.png', dpi=150)
    plt.close()
    print(f"\n   📊 Saved: {OUTPUT_DIR / 'growth_model.png'}")


# =============================================================================
# PART 2: NLP DEEP-DIVE
# =============================================================================

def ngram_analysis(df, max_n=3):
    """
    Deep N-gram analysis of video titles
    """
    print("\n" + "="*70)
    print("📝 N-GRAM ANALYSIS")
    print("   Finding common phrases in trending titles")
    print("="*70)
    
    if 'title' not in df.columns:
        print("   No title data")
        return
    
    titles = df['title'].dropna().astype(str)
    
    # Filter to English
    english_pattern = r'^[a-zA-Z0-9\s\.\,\!\?\-\:\'\"\|\(\)\[\]]+$'
    titles = titles[titles.str.match(english_pattern, na=False)]
    titles = titles[titles.str.len() > 5]
    
    print(f"   Analyzing {len(titles)} titles...")
    
    all_ngrams = {}
    
    for n in range(1, max_n + 1):
        vectorizer = CountVectorizer(ngram_range=(n, n), max_features=30, 
                                     stop_words='english', min_df=10)
        try:
            X = vectorizer.fit_transform(titles)
            freqs = X.sum(axis=0).A1
            features = vectorizer.get_feature_names_out()
            
            ngram_freq = list(zip(features, freqs))
            ngram_freq.sort(key=lambda x: x[1], reverse=True)
            all_ngrams[n] = ngram_freq[:15]
            
            print(f"\n   {'📊' if n == 1 else '📜' if n == 2 else '📖'} TOP {n}-GRAMS:")
            for phrase, count in ngram_freq[:10]:
                print(f"      '{phrase}': {int(count):,}")
        except:
            pass
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, n in enumerate([1, 2, 3]):
        if n in all_ngrams:
            phrases = [p for p, _ in all_ngrams[n][:12]]
            counts = [c for _, c in all_ngrams[n][:12]]
            
            bars = axes[i].barh(range(len(phrases)), counts, color=plt.cm.viridis(0.2 + i*0.3))
            axes[i].set_yticks(range(len(phrases)))
            axes[i].set_yticklabels(phrases, fontsize=9)
            axes[i].set_xlabel('Frequency')
            axes[i].set_title(f'Top {n}-grams in Titles')
            axes[i].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ngram_analysis.png', dpi=150)
    plt.close()
    print(f"\n   📊 Saved: {OUTPUT_DIR / 'ngram_analysis.png'}")


def title_structure_analysis(df):
    """
    Analyze structural patterns in titles
    """
    print("\n" + "="*70)
    print("🏗️  TITLE STRUCTURE ANALYSIS")
    print("   Deconstructing what makes a successful title")
    print("="*70)
    
    if 'title' not in df.columns:
        print("   No title data")
        return
    
    df = df.copy()
    titles = df['title'].dropna().astype(str)
    
    # Extract structural features
    df['title_length'] = titles.str.len()
    df['word_count'] = titles.str.split().str.len()
    df['uppercase_ratio'] = titles.apply(lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1))
    df['punctuation_count'] = titles.str.count(r'[^\w\s]')
    df['number_count'] = titles.str.count(r'\d')
    df['special_chars'] = titles.str.contains(r'[|•→🔥💯]', regex=True).astype(int)
    
    # Patterns
    patterns = {
        'has_pipe': titles.str.contains(r'\|').mean() * 100,
        'has_colon': titles.str.contains(r':').mean() * 100,
        'has_dash': titles.str.contains(r' - ').mean() * 100,
        'has_hashtag': titles.str.contains(r'#').mean() * 100,
        'has_at': titles.str.contains(r'@').mean() * 100,
        'has_ft': titles.str.lower().str.contains(r'\bft\b|\bfeat\b').mean() * 100,
        'is_all_caps': (df['uppercase_ratio'] > 0.7).mean() * 100,
        'has_year': titles.str.contains(r'20\d\d').mean() * 100,
        'starts_number': titles.str.match(r'^\d').mean() * 100,
        'has_question': titles.str.contains(r'\?').mean() * 100,
        'has_exclamation': titles.str.contains(r'!').mean() * 100,
    }
    
    print(f"\n   📊 TITLE PATTERNS:")
    for pattern, pct in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(pct)
        print(f"      {pattern:<20}: {bar} {pct:.1f}%")
    
    # Optimal length analysis
    if 'views' in df.columns or 'view_count' in df.columns:
        view_col = 'views' if 'views' in df.columns else 'view_count'
        
        # Bin by length
        df['length_bin'] = pd.cut(df['title_length'], bins=[0, 30, 50, 70, 100, 200], 
                                  labels=['<30', '30-50', '50-70', '70-100', '100+'])
        
        length_views = df.groupby('length_bin', observed=True)[view_col].median()
        
        print(f"\n   📏 OPTIMAL TITLE LENGTH:")
        best_length = length_views.idxmax()
        for length, views in length_views.items():
            marker = "👑" if length == best_length else "  "
            if views > 1e6:
                print(f"      {marker} {length}: {views/1e6:.1f}M median views")
            else:
                print(f"      {marker} {length}: {views/1e3:.0f}K median views")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Title length distribution
    axes[0, 0].hist(df['title_length'], bins=50, color='steelblue', edgecolor='white')
    axes[0, 0].axvline(df['title_length'].median(), color='red', linestyle='--', label=f"Median: {df['title_length'].median():.0f}")
    axes[0, 0].set_xlabel('Title Length (characters)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Title Length Distribution')
    axes[0, 0].legend()
    
    # Word count distribution
    axes[0, 1].hist(df['word_count'].clip(upper=20), bins=20, color='coral', edgecolor='white')
    axes[0, 1].set_xlabel('Word Count')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Title Word Count Distribution')
    
    # Pattern frequencies
    patterns_sorted = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:10]
    names = [p[0].replace('has_', '').replace('_', ' ') for p in patterns_sorted]
    values = [p[1] for p in patterns_sorted]
    axes[1, 0].barh(range(len(names)), values, color='teal')
    axes[1, 0].set_yticks(range(len(names)))
    axes[1, 0].set_yticklabels(names)
    axes[1, 0].set_xlabel('Percentage of Titles')
    axes[1, 0].set_title('Common Title Patterns')
    axes[1, 0].invert_yaxis()
    
    # Views by title length
    if 'views' in df.columns or 'view_count' in df.columns:
        df_clean = df.dropna(subset=[view_col, 'length_bin'])
        df_clean = df_clean[df_clean[view_col] > 0]
        
        df_clean.boxplot(column=view_col, by='length_bin', ax=axes[1, 1])
        axes[1, 1].set_yscale('log')
        axes[1, 1].set_xlabel('Title Length Range')
        axes[1, 1].set_ylabel('Views (log scale)')
        axes[1, 1].set_title('Views by Title Length')
        plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'title_structure.png', dpi=150)
    plt.close()
    print(f"\n   📊 Saved: {OUTPUT_DIR / 'title_structure.png'}")


def clickbait_detector(df):
    """
    Build a clickbait detection model
    """
    print("\n" + "="*70)
    print("🎣 CLICKBAIT DETECTION ANALYSIS")
    print("   Identifying clickbait patterns in titles")
    print("="*70)
    
    if 'title' not in df.columns:
        print("   No title data")
        return
    
    titles = df['title'].dropna().astype(str)
    
    # Clickbait indicators
    clickbait_patterns = {
        'number_list': r'^\d+\s',  # "10 things..."
        'you_wont_believe': r'(you won\'t believe|shocking|unbelievable)',
        'this_is': r'\bthis (is|will)\b',
        'amazing': r'\b(amazing|incredible|insane|crazy|epic)\b',
        'find_out': r'(find out|discover|learn|see what)',
        'makes_you': r'(will make you|makes you)',
        'everyone': r'(everyone|nobody|no one)',
        'secret': r'(secret|hidden|truth)',
        'must_see': r'(must see|watch this|you need)',
        'emoji_heavy': r'[🔥💯😱😍🤯]{2,}',
        'all_caps_words': r'\b[A-Z]{4,}\b',
        'excessive_punctuation': r'[!?]{2,}',
    }
    
    clickbait_scores = {}
    
    for name, pattern in clickbait_patterns.items():
        matches = titles.str.lower().str.contains(pattern, regex=True, na=False)
        clickbait_scores[name] = matches.mean() * 100
    
    print(f"\n   🎣 CLICKBAIT INDICATOR FREQUENCY:")
    sorted_scores = sorted(clickbait_scores.items(), key=lambda x: x[1], reverse=True)
    for name, pct in sorted_scores:
        if pct > 0.1:
            bar = "█" * int(pct * 5)
            print(f"      {name:<25}: {bar} {pct:.1f}%")
    
    # Calculate overall clickbait score
    df = df.copy()
    df['clickbait_score'] = 0
    
    for name, pattern in clickbait_patterns.items():
        df['clickbait_score'] += df['title'].astype(str).str.lower().str.contains(
            pattern, regex=True, na=False).astype(int)
    
    print(f"\n   📊 CLICKBAIT SCORE DISTRIBUTION:")
    for score in range(0, 5):
        count = (df['clickbait_score'] == score).sum()
        pct = count / len(df) * 100
        print(f"      Score {score}: {pct:.1f}% ({count:,} titles)")
    
    # Most clickbaity titles
    print(f"\n   🎣 MOST CLICKBAITY TITLES:")
    top_clickbait = df.nlargest(5, 'clickbait_score')
    for _, row in top_clickbait.iterrows():
        score = row['clickbait_score']
        title = str(row['title'])[:60]
        print(f"      [{score}] {title}...")
    
    # Clickbait vs views
    if 'views' in df.columns or 'view_count' in df.columns:
        view_col = 'views' if 'views' in df.columns else 'view_count'
        
        clickbait_views = df.groupby('clickbait_score')[view_col].median()
        
        print(f"\n   📈 CLICKBAIT SCORE VS VIEWS:")
        for score in sorted(clickbait_views.index)[:5]:
            views = clickbait_views[score]
            if views > 1e6:
                print(f"      Score {score}: {views/1e6:.1f}M median views")
            else:
                print(f"      Score {score}: {views/1e3:.0f}K median views")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Indicator frequencies
    names = [s[0].replace('_', ' ') for s in sorted_scores[:10]]
    values = [s[1] for s in sorted_scores[:10]]
    axes[0].barh(range(len(names)), values, color='crimson')
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=9)
    axes[0].set_xlabel('Percentage of Titles')
    axes[0].set_title('Clickbait Indicators')
    axes[0].invert_yaxis()
    
    # Clickbait score distribution
    score_counts = df['clickbait_score'].value_counts().sort_index()
    axes[1].bar(score_counts.index, score_counts.values, color='steelblue')
    axes[1].set_xlabel('Clickbait Score')
    axes[1].set_ylabel('Number of Videos')
    axes[1].set_title('Clickbait Score Distribution')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'clickbait_analysis.png', dpi=150)
    plt.close()
    print(f"\n   📊 Saved: {OUTPUT_DIR / 'clickbait_analysis.png'}")


def semantic_clustering(df, n_clusters=8):
    """
    Cluster titles by semantic similarity using TF-IDF + NMF
    """
    print("\n" + "="*70)
    print("🔮 SEMANTIC CLUSTERING")
    print("   Grouping similar video titles")
    print("="*70)
    
    if 'title' not in df.columns:
        print("   No title data")
        return
    
    # Filter titles
    titles = df['title'].dropna().astype(str)
    english = titles[titles.str.match(r'^[a-zA-Z0-9\s\.\,\!\?\-\:\'\"\|]+$', na=False)]
    english = english[english.str.len() > 10]
    
    if len(english) > 30000:
        english = english.sample(30000, random_state=42)
    
    print(f"   Clustering {len(english)} titles into {n_clusters} groups...")
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', 
                                 min_df=10, max_df=0.5)
    tfidf = vectorizer.fit_transform(english)
    
    # NMF for topic extraction
    nmf = NMF(n_components=n_clusters, random_state=42, max_iter=200)
    doc_topics = nmf.fit_transform(tfidf)
    
    # Assign clusters
    clusters = doc_topics.argmax(axis=1)
    
    # Get topic keywords
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"\n   📊 SEMANTIC CLUSTERS:")
    for topic_idx, topic in enumerate(nmf.components_):
        top_words_idx = topic.argsort()[-8:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        count = (clusters == topic_idx).sum()
        pct = count / len(clusters) * 100
        
        print(f"\n      Cluster {topic_idx + 1} ({pct:.1f}%):")
        print(f"         Keywords: {', '.join(top_words[:5])}")
        
        # Example titles
        cluster_titles = english[clusters == topic_idx].head(2).tolist()
        for t in cluster_titles:
            print(f"         Example: \"{t[:50]}...\"")
    
    # Visualize with dimensionality reduction
    from sklearn.decomposition import TruncatedSVD
    
    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(tfidf)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=clusters, cmap='tab10', 
                        alpha=0.5, s=10)
    ax.set_xlabel('SVD Dimension 1')
    ax.set_ylabel('SVD Dimension 2')
    ax.set_title(f'Semantic Clustering of {len(english)} Video Titles')
    
    # Add cluster centers
    for i in range(n_clusters):
        mask = clusters == i
        if mask.sum() > 0:
            center = coords[mask].mean(axis=0)
            top_words = [feature_names[j] for j in nmf.components_[i].argsort()[-3:][::-1]]
            ax.annotate(f"C{i+1}: {top_words[0]}", center, fontsize=9, 
                       fontweight='bold', ha='center')
    
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'semantic_clusters.png', dpi=150)
    plt.close()
    print(f"\n   📊 Saved: {OUTPUT_DIR / 'semantic_clusters.png'}")


def word_cooccurrence_network(df, top_n=50):
    """
    Build word co-occurrence network from titles
    """
    print("\n" + "="*70)
    print("🕸️  WORD CO-OCCURRENCE NETWORK")
    print("   Finding which words appear together")
    print("="*70)
    
    if 'title' not in df.columns:
        print("   No title data")
        return
    
    titles = df['title'].dropna().astype(str)
    
    # Tokenize and filter
    from collections import defaultdict
    
    stopwords = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 
                 'is', 'it', 'this', 'that', 'with', 'from', 'as', 'by', 'be', 'are',
                 'was', 'were', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
                 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'if'}
    
    cooccurrence = defaultdict(int)
    word_freq = Counter()
    
    for title in titles:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', str(title).lower())
        words = [w for w in words if w not in stopwords][:10]  # Limit words per title
        word_freq.update(words)
        
        # Count pairs
        for i, w1 in enumerate(words):
            for w2 in words[i+1:]:
                pair = tuple(sorted([w1, w2]))
                cooccurrence[pair] += 1
    
    # Top co-occurring pairs
    top_pairs = sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)[:30]
    
    print(f"\n   🔗 TOP WORD CO-OCCURRENCES:")
    for (w1, w2), count in top_pairs[:15]:
        print(f"      '{w1}' + '{w2}': {count:,}")
    
    # Build graph visualization
    try:
        import networkx as nx
        
        G = nx.Graph()
        
        # Add top words as nodes
        top_words = [w for w, _ in word_freq.most_common(top_n)]
        
        # Add edges for strong co-occurrences
        for (w1, w2), count in top_pairs:
            if w1 in top_words and w2 in top_words and count > 50:
                G.add_edge(w1, w2, weight=count)
        
        if len(G.edges()) > 0:
            fig, ax = plt.subplots(figsize=(14, 14))
            
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
            
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            max_weight = max(weights) if weights else 1
            
            nx.draw_networkx_nodes(G, pos, node_size=200, node_color='lightblue', 
                                  alpha=0.8, ax=ax)
            nx.draw_networkx_edges(G, pos, width=[w/max_weight*5 for w in weights], 
                                  alpha=0.5, edge_color='gray', ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
            
            ax.set_title('Word Co-occurrence Network in Video Titles')
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'word_network.png', dpi=150)
            plt.close()
            print(f"\n   📊 Saved: {OUTPUT_DIR / 'word_network.png'}")
        
    except ImportError:
        print("   NetworkX not available for visualization")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "🔬"*35)
    print("   SPECIALIZED ANALYSIS")
    print("   Time-Series & NLP Deep-Dive")
    print("🔬"*35)
    
    # Load data
    print("\n📂 Loading datasets...")
    data = load_data()
    
    if not data:
        print("No data!")
        return
    
    # TIME-SERIES ANALYSIS
    print("\n" + "🕐"*35)
    print("   PART 1: TIME-SERIES ANALYSIS")
    print("🕐"*35)
    
    if 'trending' in data:
        time_series_trending_analysis(data['trending'])
        category_time_evolution(data['trending'])
    
    if 'global' in data:
        subscriber_growth_model(data['global'])
    
    # NLP DEEP-DIVE
    print("\n" + "📝"*35)
    print("   PART 2: NLP DEEP-DIVE")
    print("📝"*35)
    
    if 'trending' in data:
        ngram_analysis(data['trending'])
        title_structure_analysis(data['trending'])
        clickbait_detector(data['trending'])
        semantic_clustering(data['trending'])
        word_cooccurrence_network(data['trending'])
    
    # Summary
    print("\n" + "="*70)
    print("✅ SPECIALIZED ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n   All outputs saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
