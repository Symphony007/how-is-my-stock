# 📈 How is my Stock?

A futuristic, community-powered stock sentiment analysis platform for Indian markets (NSE/BSE).  
Analyze real-time market data, vote on stocks, track sentiment trends, and engage in stock-specific discussions — all within a sleek cyberpunk-inspired UI.

---

## 🔍 Features

- ✅ Real-time Sentiment Analysis using FinBERT and News APIs  
- 📊 Live Market Charts with Technical Metrics (Price, PE, EPS, 52W High/Low)  
- 🧠 Buy/Hold/Sell Voting System (1 vote/user/stock/day)  
- 🗣️ Reddit-style Discussion Forums for each stock  
- 📁 User Profiles with activity logs and custom Watchlist  
- ⚙️ Built using Flask, SQLite, YFinance, and News APIs  
- 🎨 Responsive UI with a cyberpunk-futuristic design

---

## 🗂 Folder Structure

HowIsMyStock/
│  
├── instance/               # Contains SQLite DB  
│   └── stocks.db  
│  
├── migrations/             # Flask-Migrate files (if used)  
│  
├── static/  
│   ├── css/  
│   │   ├── loader.css  
│   │   ├── profile.css  
│   │   ├── results.css  
│   │   └── styles.css  
│   ├── js/  
│   │   ├── chart.js  
│   │   ├── main.js  
│   │   ├── results.js  
│   │   └── search.js  
│   └── images/  
│       ├── bg.png  
│       └── pfp.png  
│  
├── templates/  
│   ├── layout.html  
│   ├── index.html  
│   ├── login.html  
│   ├── register.html  
│   ├── profile.html  
│   ├── results.html  
│   ├── watchlist.html  
│   └── discussion.html  
│  
├── .env                    # API keys and secrets  
├── app.py                  # Flask backend logic  
├── NSE_List.csv            # List of NSE stocks (symbol + name)  
├── requirements.txt        # Python dependencies  
└── README.md               # This file

---

## ⚙️ Tech Stack

- Frontend: HTML5, CSS3, JavaScript  
- Backend: Python (Flask)  
- Database: SQLite  
- APIs Used:  
  - YFinance (market data)  
  - Alpha Vantage (technical indicators)  
  - Finnhub (fundamentals)  
  - NewsAPI (financial headlines)  
  - FinBERT (NLP sentiment classification)

---

## 🧪 Requirements

Ensure Python 3.10+ is installed. Install dependencies via:

pip install -r requirements.txt

---

## 🚀 Getting Started (Local Setup)

# 1. Clone the repository  
git clone https://github.com/YOUR_USERNAME/how-is-my-stock.git  
cd how-is-my-stock  

# 2. Set up a virtual environment  
python -m venv venv  
source venv/bin/activate        # On Windows: venv\Scripts\activate  

# 3. Install dependencies  
pip install -r requirements.txt  

# 4. Set up environment variables  
touch .env                      # Or create manually  

# Inside .env, add:  
SECRET_KEY=your_secret_key  
FINNHUB_API_KEY=your_key  
ALPHA_VANTAGE_API_KEY=your_key  
NEWSAPI_KEY=your_key  

# 5. Initialize the database (only once)  
flask db init  
flask db migrate -m "Initial migration"  
flask db upgrade  

# 6. Run the Flask app  
flask run

---

## 🧠 Future Roadmap

- [ ] Google OAuth 2.0 Integration  
- [ ] Auto-refreshing NSE stock list via cron/script  
- [ ] Per-stock sentiment history with graphs  
- [ ] Notifications for Watchlist updates  
- [ ] Caching & performance improvements  
- [ ] Dockerized deployment + production hosting (Fly.io / Railway)

---

## 📣 About the Developer

Hi, I'm Deepmalya Mallick — a curious developer & creative technologist passionate about fusing AI, finance, and modern UI.

📧 mallickdeepmalya05@gmail.com    
🔗 LinkedIn: https://www.linkedin.com/in/deepmalya-mallick-62a321305/  

---
