"""
YouTube Channel Data Overview
Shows all available datasets and their statistics
"""

import pandas as pd
from pathlib import Path
import os

def analyze_dataset(filepath, name, encoding='utf-8'):
    """Analyze a single CSV dataset"""
    try:
        df = pd.read_csv(filepath, encoding=encoding, on_bad_lines='skip')
        print(f"\n{'='*60}")
        print(f"📊 {name}")
        print(f"   File: {filepath.name}")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {list(df.columns)}")
        
        # Check for channel-related columns
        channel_cols = [c for c in df.columns if any(x in c.lower() for x in ['channel', 'youtuber', 'subscriber', 'views'])]
        if channel_cols:
            print(f"   Key columns: {channel_cols}")
        
        # Show sample
        if 'channel_title' in df.columns:
            unique_channels = df['channel_title'].nunique()
            print(f"   Unique channels: {unique_channels:,}")
        
        return df
    except Exception as e:
        print(f"   Error reading: {e}")
        return None


def main():
    print("="*60)
    print("🎬 YOUTUBE CHANNEL ANALYTICS - DATA INVENTORY")
    print("="*60)
    
    data_dir = Path("data/raw")
    
    # 1. Global Statistics
    analyze_dataset(
        data_dir / "global_stats" / "Global YouTube Statistics.csv",
        "Global YouTube Statistics 2023 (Top Channels)",
        encoding='latin-1'
    )
    
    # 2. Top Channels
    analyze_dataset(
        data_dir / "top_channels" / "most_subscribed_youtube_channels.csv",
        "Most Subscribed YouTube Channels"
    )
    
    # 3. GitHub Top Subscribed
    analyze_dataset(
        data_dir / "github_channel_growth" / "data" / "topSubscribed.csv",
        "GitHub: Top 1000 Subscribed Channels"
    )
    
    # 4. Statistics General
    for csv_file in (data_dir / "statistics_general").glob("*.csv"):
        analyze_dataset(csv_file, f"General Statistics: {csv_file.stem}")
    
    # 5. Trending Historical (multiple countries)
    print(f"\n{'='*60}")
    print("📊 TRENDING VIDEOS BY COUNTRY (Historical)")
    trending_dir = data_dir / "trending_historical"
    
    total_videos = 0
    total_channels = set()
    
    for country_file in sorted(trending_dir.glob("*videos.csv")):
        country = country_file.stem[:2]
        try:
            df = pd.read_csv(country_file, encoding='latin-1', on_bad_lines='skip')
            channels = set(df['channel_title'].dropna().unique())
            total_videos += len(df)
            total_channels.update(channels)
            print(f"   {country}: {len(df):,} videos, {len(channels):,} unique channels")
        except Exception as e:
            print(f"   {country}: Error - {e}")
    
    print(f"\n   TOTAL: {total_videos:,} videos, {len(total_channels):,} unique channels")
    
    # 6. YouTube Main
    print(f"\n{'='*60}")
    print("📊 YOUTUBE MAIN (US/GB with Comments)")
    main_dir = data_dir / "youtube_main"
    for csv_file in sorted(main_dir.glob("*videos.csv")):
        analyze_dataset(csv_file, f"Main: {csv_file.stem}", encoding='latin-1')
    
    # Summary
    print(f"\n{'='*60}")
    print("📈 SUMMARY")
    print("="*60)
    
    total_size = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file())
    total_files = len(list(data_dir.rglob("*.csv")))
    
    print(f"   Total CSV files: {total_files}")
    print(f"   Total size: {total_size / (1024*1024):.1f} MB")
    print(f"   Unique channels across trending: {len(total_channels):,}")
    
    print("\n📋 DATA COVERAGE:")
    print("   ✓ Channel metadata (subscribers, views, category)")
    print("   ✓ Video titles from trending")  
    print("   ✓ Engagement metrics (likes, comments)")
    print("   ✓ Multi-country data (US, GB, CA, DE, FR, IN, JP, KR, MX, RU)")
    
    print("\n⚠️  GAPS TO FILL:")
    print("   • 2024-2025 channel data (requires Kaggle terms acceptance)")
    print("   • Historical subscriber growth over time")
    print("   • Channel creation dates for full 2005-2025 timeline")


if __name__ == "__main__":
    main()
