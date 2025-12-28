"""
YouTube Datasets Downloader
Downloads comprehensive YouTube data from Kaggle (2005-2025)
"""

import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Dataset configurations
DATASETS = {
    # Historical Data (2005) - YouTube's earliest videos
    "youtube_origins": {
        "kaggle_id": "gaiustulius/youtube-origins-trailblazing-early-uploads",
        "description": "YouTube's earliest uploads (2005)",
        "folder": "2005_origins"
    },
    
    # User Activity & Trending (2017-2018)
    "youtube_new": {
        "kaggle_id": "datasnaek/youtube-new",
        "description": "Trending videos data with engagement metrics",
        "folder": "trending_historical"
    },
    
    # Daily Updated Trending Videos (2020-2025)
    "trending_daily": {
        "kaggle_id": "rsrishav/youtube-trending-video-dataset",
        "description": "Daily updated trending videos (multi-country)",
        "folder": "trending_daily_2025"
    },
    
    # 2025 Channel Data  
    "channels_2025": {
        "kaggle_id": "bechahmed4/youtube-2025-channels",
        "description": "YouTube channels metadata (Oct 2025)",
        "folder": "channels_2025"
    },
    
    # Top Channel Videos 2025
    "top_videos_2025": {
        "kaggle_id": "bechahmed4/youtube-top-channel-videos-2025",
        "description": "Latest videos from top channels (2025)",
        "folder": "top_videos_2025"
    },
    
    # YouTube Statistics (general metrics)
    "youtube_statistics": {
        "kaggle_id": "advaypatil/youtube-statistics",
        "description": "General YouTube video statistics",
        "folder": "statistics_general"
    },
}

def check_kaggle_auth():
    """Verify Kaggle API credentials are configured"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    
    if kaggle_json.exists():
        print("✓ Kaggle credentials found at ~/.kaggle/kaggle.json")
        return True
    
    # Check environment variables
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    
    if username and key and username != "your_kaggle_username":
        print("✓ Kaggle credentials found in .env")
        # Create kaggle.json from env vars
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        
        import json
        with open(kaggle_json, 'w') as f:
            json.dump({"username": username, "key": key}, f)
        
        # Set permissions (Windows doesn't need chmod)
        print(f"  Created {kaggle_json}")
        return True
    
    print("✗ Kaggle credentials not found!")
    print("\nTo fix this:")
    print("1. Go to https://www.kaggle.com/settings")
    print("2. Click 'Create New Token' to download kaggle.json")  
    print("3. Either:")
    print("   a) Copy kaggle.json to ~/.kaggle/kaggle.json")
    print("   b) Update KAGGLE_USERNAME and KAGGLE_KEY in .env")
    return False


def download_dataset(name: str, config: dict, data_dir: Path):
    """Download a single dataset from Kaggle"""
    output_path = data_dir / config["folder"]
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Downloading: {config['description']}")
    print(f"Dataset: {config['kaggle_id']}")
    print(f"Target: {output_path}")
    print('='*60)
    
    try:
        cmd = [
            "kaggle", "datasets", "download",
            "-d", config["kaggle_id"],
            "-p", str(output_path),
            "--unzip"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Successfully downloaded {name}")
            # List downloaded files
            files = list(output_path.glob("*"))
            for f in files[:5]:  # Show first 5 files
                size_mb = f.stat().st_size / (1024*1024) if f.is_file() else 0
                print(f"  - {f.name} ({size_mb:.1f} MB)" if f.is_file() else f"  - {f.name}/")
            if len(files) > 5:
                print(f"  ... and {len(files)-5} more files")
        else:
            print(f"✗ Failed to download {name}")
            print(f"  Error: {result.stderr}")
            
    except FileNotFoundError:
        print("✗ Kaggle CLI not found. Make sure it's installed: pip install kaggle")
    except Exception as e:
        print(f"✗ Error: {e}")


def main():
    print("=" * 60)
    print("YouTube Data Downloader (2005-2025)")
    print("=" * 60)
    
    # Check authentication
    if not check_kaggle_auth():
        return
    
    # Setup directories
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading {len(DATASETS)} datasets to {data_dir.absolute()}")
    print("This may take a while depending on your connection...\n")
    
    # Download all datasets
    for name, config in DATASETS.items():
        download_dataset(name, config, data_dir)
    
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    print(f"\nData saved to: {data_dir.absolute()}")
    print("\nNext steps:")
    print("1. Run 'jupyter notebook' to start exploring the data")
    print("2. Check notebooks/ for analysis templates")


if __name__ == "__main__":
    main()
