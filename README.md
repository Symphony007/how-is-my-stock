# 📈 How is my Stock?

A futuristic, community-powered **stock sentiment analysis platform** for **Indian markets (NSE/BSE)**. Analyze real-time market data, vote on stocks, and participate in discussion forums – all in one sleek interface.

---

## 🔍 Features

- ✅ **Real-time Sentiment Analysis** using FinBERT & market APIs
- 📊 **Live Charts** and Technical Metrics (Price, PE, EPS, etc.)
- 🧠 **Buy/Hold/Sell Voting System** (1 vote/user/stock/day)
- 🗣️ **Stock-specific Discussion Forums** (like Reddit)
- 🧾 **Watchlist & User Profiles**
- ⚙️ Built using **Flask**, **SQLite**, **YFinance**, **News APIs**
- 🎨 Stylish UI with cyberpunk-inspired design

---

## 📁 Folder Structure

```
HowIsMyStock/
│
├── instance/               # Contains SQLite DB
│   └── stocks.db
│
├── migrations/             # (If using Flask-Migrate for DB updates)
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
├── app.py                 # Flask backend logic
├── NSE_List.csv           # List of NSE stocks
└── README.md              # Project documentation
```

---

## ⚙️ Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript
- **Backend**: Python (Flask)
- **Database**: SQLite
- **APIs**:
  - `YFinance`
  - `Alpha Vantage`
  - `Finnhub`
  - `FinBERT (Hugging Face)`
  - `NewsAPI`

---

## 🧪 Requirements

Install the dependencies:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
flask
flask_sqlalchemy
flask_cors
flask_caching
flask_migrate
flask_wtf
python-dotenv
requests
yfinance
pandas
numpy
bs4
newsapi-python
transformers
torch
nsepython
feedparser
werkzeug
```
---

## 🚀 Getting Started (Local Setup)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/how-is-my-stock.git
cd how-is-my-stock

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
Create a `.env` file:
FLASK_APP=app.py
SECRET_KEY=your_secret_key
FINNHUB_API_KEY=your_key
ALPHA_VANTAGE_API_KEY=your_key
NEWSAPI_KEY=your_key

# 5. Run the app
flask run
```

---

## 🧠 Future Plans

- [ ] Google OAuth Login
- [ ] Auto-update NSE List
- [ ] Sentiment charts per stock
- [ ] Push notifications for watchlist changes

---