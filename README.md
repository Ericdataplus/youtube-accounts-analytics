# 🎬 YouTube Analytics Dashboard

Advanced Machine Learning Analysis on YouTube Channel & Video Data

[![GitHub Pages](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-blue?style=for-the-badge)](https://ericdataplus.github.io/youtube-accounts-analytics/)

## 📊 Project Overview

This project applies **28 advanced visualization techniques** and **research-grade machine learning methods** to analyze YouTube channel and video data spanning from 2005 to 2024.

### Key Statistics
- **700+ MB** of data analyzed
- **375,942** trending videos
- **61 countries** of influencer data
- **28** unique visualizations

## 🔍 Key Findings

| Finding | Value | Description |
|---------|-------|-------------|
| Cross-Platform Correlation | ρ = 0.608 | Spotify streams ↔ YouTube views |
| Optimal Title Length | <30 chars | 360K median views vs 93K for long titles |
| Best Trending Day | Saturday | 55K+ videos vs 53K on Thursday |
| Growth Exponent | β = 0.128 | Sublinear growth - older channels grow slower |
| Clickbait Premium | 8x | High clickbait score → more views |

## 🔬 Research Methods Applied

- **Power Law / Zipf's Law** - Scale-free network analysis
- **Latent Dirichlet Allocation (LDA)** - Topic modeling on 157K titles
- **Benford's Law** - First-digit anomaly detection
- **PageRank Algorithm** - Channel influence ranking
- **UMAP/t-SNE** - Manifold learning embeddings
- **Pareto Frontier Analysis** - Multi-objective optimization
- **Collaborative Filtering** - Channel similarity
- **Shannon Entropy** - Content diversity measurement
- **NMF Topic Modeling** - Semantic clustering
- **Isolation Forest** - Anomaly detection
- **Random Forest** - Viral feature prediction

## 📁 Project Structure

```
youtube-accounts-analytics/
├── index.html              # Interactive dashboard (GitHub Pages)
├── data/
│   ├── raw/                # Downloaded datasets
│   └── processed/
│       └── analysis/       # Generated visualizations
│           ├── advanced/   # Research method charts
│           └── specialized/ # Time-series & NLP charts
├── scripts/
│   ├── download_datasets.py
│   ├── advanced_ml_analysis.py
│   ├── research_methods_analysis.py
│   ├── advanced_analytics_part2.py
│   └── specialized_analysis.py
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/youtube-accounts-analytics.git
cd youtube-accounts-analytics

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run analyses
python scripts/advanced_ml_analysis.py
python scripts/research_methods_analysis.py
python scripts/specialized_analysis.py
```

## 📚 Data Sources

| Dataset | Description | Source |
|---------|-------------|--------|
| Global YouTube Statistics 2023 | Top 995 channels | [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/global-youtube-statistics-2023) |
| YouTube Trending Videos | 375K videos from 10 countries | [Kaggle](https://www.kaggle.com/datasets/datasnaek/youtube-new) |
| Most Subscribed Channels | Top 1,000 channels | [Kaggle](https://www.kaggle.com/datasets/themrityunjaypathak/most-subscribed-1000-youtube-channels) |
| Social Media Influencers 2024 | 61 countries | [Kaggle](https://www.kaggle.com/datasets/bhavyadhingra00020/top-100-social-media-influencers-2024-countrywise) |
| Spotify & YouTube | 20K cross-platform tracks | [Kaggle](https://www.kaggle.com/datasets/salvatorerastelli/spotify-and-youtube) |

## 📈 Visualizations

### Time-Series Analysis
- Daily trending patterns with 7-day moving average
- Category evolution over time
- Subscriber growth power law model

### NLP Deep-Dive
- N-gram frequency analysis (1/2/3-grams)
- Title structure optimization
- Clickbait detection model
- Semantic clustering (8 topics)
- Word co-occurrence network

### Research Methods
- Zipf's Law / Power Law distribution
- Benford's Law anomaly detection
- PageRank influence scoring
- Pareto frontier optimization
- UMAP channel embeddings

## 📄 License

This project is for educational and research purposes. Data sourced from Kaggle and GitHub with appropriate licenses.

## 🙏 Acknowledgments

- Kaggle dataset contributors
- Open source ML/visualization libraries: scikit-learn, matplotlib, seaborn, networkx

---
Created with 🔬 Advanced Machine Learning | December 2024
