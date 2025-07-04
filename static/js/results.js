function showLoader() {
    const loader = document.getElementById('cyberpunk-loader');
    if (loader) {
        loader.style.display = 'flex';
        setTimeout(() => {
            loader.classList.add('active');
        }, 10);
    }
}

function hideLoader() {
    const loader = document.getElementById('cyberpunk-loader');
    if (loader) {
        loader.classList.remove('active');
        setTimeout(() => {
            loader.style.display = 'none';
        }, 500);
    }
}

function enhanceNewsDisplay() {
    document.querySelectorAll('.news-card').forEach(card => {
        const sourceElement = card.querySelector('.news-source-badge');
        if (sourceElement) {
            const source = sourceElement.textContent.trim();
            card.setAttribute('data-source', source);
            
            const link = card.querySelector('a.news-title');
            if (link) {
                link.addEventListener('click', function() {
                    const stockSymbol = JSON.parse(document.getElementById('chart-data').dataset.symbol);
                    trackNewsClick(stockSymbol, source, this.href);
                });
            }
        }
    });
}

function trackNewsClick(stockSymbol, source, url) {
    if (navigator.sendBeacon) {
        const data = new FormData();
        data.append('symbol', stockSymbol);
        data.append('source', source);
        data.append('url', url);
        navigator.sendBeacon('/api/track_news_click', data);
    }
}

function initializeChart() {
    try {
        const chartElement = document.getElementById('stockChart');
        if (!chartElement) return;

        const ctx = chartElement.getContext('2d');
        const chartDataElement = document.getElementById('chart-data');
        
        if (!chartDataElement) return;

        const chartData = JSON.parse(chartDataElement.dataset.prices || '[]');
        const chartLabels = JSON.parse(chartDataElement.dataset.dates || '[]');
        const chartType = chartDataElement.dataset.type;
        const marketStatus = chartDataElement.dataset.marketStatus;
        
        const marketStatusBadge = document.getElementById('market-status-badge');
        if (marketStatusBadge) {
            marketStatusBadge.textContent = marketStatus === 'open' 
                ? 'Market Open - Live Data' 
                : 'Market Closed - Previous Session';
        }

        if (chartData.length === 0 || chartLabels.length === 0) {
            console.warn('No chart data available');
            return;
        }

        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: chartLabels,
                datasets: [{
                    label: 'Stock Price',
                    data: chartData,
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1,
                    pointRadius: chartType === 'realtime' ? 0 : 2,
                    pointHoverRadius: chartType === 'realtime' ? 0 : 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(30, 20, 60, 0.95)',
                        titleColor: '#e0e0ff',
                        bodyColor: '#e0e0ff',
                        borderColor: '#8b5cf6',
                        borderWidth: 1,
                        padding: 12,
                        callbacks: {
                            label: function(context) {
                                return 'Price: ₹' + context.parsed.y.toFixed(2);
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(167, 139, 250, 0.1)'
                        },
                        ticks: {
                            color: '#c4b5fd'
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(167, 139, 250, 0.1)'
                        },
                        ticks: {
                            color: '#c4b5fd',
                            callback: function(value) {
                                return '₹' + value;
                            }
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        });

        if (chartType === 'realtime' && marketStatus === 'open') {
            setInterval(fetchRealtimeData, 300000);
        }

        return chart;
    } catch (error) {
        console.error('Chart initialization error:', error);
        return null;
    }
}

async function fetchRealtimeData() {
    try {
        const chartDataElement = document.getElementById('chart-data');
        if (!chartDataElement) return;
        
        const symbol = JSON.parse(chartDataElement.dataset.symbol);
        const response = await fetch(`/realtime_data?symbol=${symbol}`);
        
        if (response.ok) {
            const newData = await response.json();
            if (newData.times && newData.prices) {
                chartDataElement.dataset.dates = JSON.stringify(newData.times);
                chartDataElement.dataset.prices = JSON.stringify(newData.prices);
                initializeChart();
            }
        }
    } catch (error) {
        console.error('Failed to fetch real-time data:', error);
    }
}

async function loadTechnicalIndicators(symbol) {
    try {
        document.getElementById('rsi-value').textContent = 'N/A';
        document.getElementById('macd-value').textContent = 'N/A';
        document.getElementById('bollinger-value').textContent = 'N/A';
        document.getElementById('support-level').textContent = 'N/A';
        document.getElementById('resistance-level').textContent = 'N/A';
        
        document.getElementById('rsi-status').textContent = 'Loading...';
        document.getElementById('macd-status').textContent = 'Loading...';
        document.getElementById('bollinger-status').textContent = 'Loading...';
        
        const response = await fetch(`/api/technical_indicators/${symbol}`);
        if (!response.ok) {
            throw new Error('Failed to fetch technical indicators');
        }
        
        const data = await response.json();
        
        if (data.status === 'success') {
            if (data.rsi !== null && !isNaN(data.rsi)) {
                const rsiValue = Math.round(data.rsi);
                document.getElementById('rsi-value').textContent = rsiValue;
                const rsiBar = document.getElementById('rsi-bar');
                rsiBar.style.width = `${Math.min(100, Math.max(0, rsiValue))}%`;
                rsiBar.style.background = rsiValue > 70 ? '#f87171' : rsiValue < 30 ? '#4ade80' : '#a78bfa';
                document.getElementById('rsi-status').textContent = 
                    rsiValue > 70 ? 'Overbought' : rsiValue < 30 ? 'Oversold' : 'Neutral';
                document.getElementById('rsi-status').className = 
                    `status-indicator ${rsiValue > 70 ? 'negative' : rsiValue < 30 ? 'positive' : 'neutral'}`;
            } else {
                document.getElementById('rsi-status').textContent = 'Data unavailable';
            }
            
            if (data.macd) {
                document.getElementById('macd-value').textContent = 
                    `MACD: ${data.macd.macd_line.toFixed(2)} | Signal: ${data.macd.signal_line.toFixed(2)}`;
                document.getElementById('macd-status').textContent = 
                    `Histogram: ${data.macd.histogram.toFixed(2)} (${data.macd.trend})`;
                document.getElementById('macd-status').className = 
                    `status-indicator ${data.macd.trend === 'bullish' ? 'positive' : 'negative'}`;
            } else {
                document.getElementById('macd-status').textContent = 'Data unavailable';
            }
            
            if (data.bollinger) {
                document.getElementById('bollinger-value').textContent = 
                    data.bollinger.price_position === 'upper' ? 'Upper Band' :
                    data.bollinger.price_position === 'lower' ? 'Lower Band' : 'Middle Band';
                document.getElementById('bollinger-status').textContent = 
                    `Upper: ${data.bollinger.upper.toFixed(2)} | Lower: ${data.bollinger.lower.toFixed(2)}`;
            } else {
                document.getElementById('bollinger-status').textContent = 'Data unavailable';
            }
            
            if (data.support && data.resistance) {
                document.getElementById('support-level').textContent = data.support.toFixed(2);
                document.getElementById('resistance-level').textContent = data.resistance.toFixed(2);
            }
        }
    } catch (error) {
        console.error('Error loading technical indicators:', error);
        document.getElementById('rsi-status').textContent = 'Error loading data';
        document.getElementById('macd-status').textContent = 'Error loading data';
        document.getElementById('bollinger-status').textContent = 'Error loading data';
    }
}

function initializeVoting() {
    const stockSymbol = JSON.parse(document.getElementById('chart-data').dataset.symbol);
    const voteButtons = document.querySelectorAll('.vote-btn');
    const voteMessage = document.getElementById('vote-message');
    
    // Check if user has already voted in the last 24 hours
    const lastVote = localStorage.getItem(`lastVote_${stockSymbol}`);
    if (lastVote) {
        const lastVoteDate = new Date(lastVote);
        const now = new Date();
        const hoursSinceVote = (now - lastVoteDate) / (1000 * 60 * 60);
        
        if (hoursSinceVote < 24) {
            voteButtons.forEach(btn => {
                btn.disabled = true;
            });
            
            const hoursLeft = Math.floor(24 - hoursSinceVote);
            voteMessage.textContent = `You can vote again in ${hoursLeft} hours.`;
            voteMessage.className = 'status-indicator neutral';
        }
    }
    
    // Add click handlers to vote buttons
    voteButtons.forEach(button => {
        button.addEventListener('click', function() {
            const voteType = this.textContent.trim();
            castVote(stockSymbol, voteType);
        });
    });
}

function castVote(stockSymbol, voteType) {
    const csrfToken = document.querySelector('meta[name="csrf-token"]').content;
    
    fetch('/vote', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken
        },
        body: JSON.stringify({
            stock_symbol: stockSymbol,
            vote: voteType
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            // Update the vote counts and bars
            updateVoteDisplay(data.vote_counts);
            
            // Show success message
            showVoteMessage(data.message, 'positive');
            
            // Disable voting buttons for 24 hours
            disableVotingButtons();
            
            // Store the vote timestamp in localStorage
            localStorage.setItem(`lastVote_${stockSymbol}`, new Date().toISOString());
        } else {
            showVoteMessage(data.message, 'negative');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showVoteMessage('Error submitting vote. Please try again.', 'negative');
    });
}

function updateVoteDisplay(voteCounts) {
    // Update vote count numbers
    document.querySelector('.vote-bar.buy .vote-count').textContent = voteCounts.Buy;
    document.querySelector('.vote-bar.hold .vote-count').textContent = voteCounts.Hold;
    document.querySelector('.vote-bar.sell .vote-count').textContent = voteCounts.Sell;
    
    // Calculate total votes
    const totalVotes = voteCounts.Buy + voteCounts.Hold + voteCounts.Sell;
    
    // Update bar heights (max height 150px)
    const buyHeight = totalVotes > 0 ? (voteCounts.Buy / totalVotes) * 150 : 50;
    const holdHeight = totalVotes > 0 ? (voteCounts.Hold / totalVotes) * 150 : 50;
    const sellHeight = totalVotes > 0 ? (voteCounts.Sell / totalVotes) * 150 : 50;
    
    document.querySelector('.vote-bar.buy').style.height = `${buyHeight}px`;
    document.querySelector('.vote-bar.hold').style.height = `${holdHeight}px`;
    document.querySelector('.vote-bar.sell').style.height = `${sellHeight}px`;
}

function showVoteMessage(message, type) {
    const voteMessage = document.getElementById('vote-message');
    voteMessage.textContent = message;
    voteMessage.className = `status-indicator ${type}`;
    
    // Auto-hide message after 5 seconds
    setTimeout(() => {
        voteMessage.textContent = '';
        voteMessage.className = 'status-indicator';
    }, 5000);
}

function disableVotingButtons() {
    document.querySelectorAll('.vote-btn').forEach(btn => {
        btn.disabled = true;
    });
}

function setupMarketStatusPolling() {
    if (document.getElementById('market-status-badge')) {
        setInterval(() => {
            fetch('/api/market_status')
                .then(response => response.json())
                .then(data => {
                    const badge = document.getElementById('market-status-badge');
                    if (badge && data.is_market_open !== undefined) {
                        badge.textContent = data.is_market_open ? 
                            'Market Open - Live Data' : 
                            'Market Closed - Previous Session';
                        badge.className = `market-status ${data.is_market_open ? 'open' : 'closed'}`;
                    }
                })
                .catch(console.error);
        }, 60000);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const loader = document.getElementById('cyberpunk-loader');
    let startTime = Date.now();
    const minDisplayTime = 1500;
    
    if (loader) loader.classList.add('active');
    
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            if (loader) loader.classList.add('active');
        });
    });
    
    window.addEventListener('pageshow', function(event) {
        if (event.persisted) {
            if (loader) loader.classList.remove('active', 'hide');
        }
    });

    const stockSymbol = document.getElementById('chart-data') ? 
        JSON.parse(document.getElementById('chart-data').dataset.symbol) : null;
    
    initializeChart();
    
    if (stockSymbol) {
        loadTechnicalIndicators(stockSymbol);
    }
    
    enhanceNewsDisplay();
    initializeVoting();
    setupMarketStatusPolling();

    window.addEventListener('load', function() {
        const elapsed = Date.now() - startTime;
        const remaining = Math.max(0, minDisplayTime - elapsed);
        
        setTimeout(() => {
            if (loader) {
                loader.classList.add('hide');
                setTimeout(() => {
                    loader.classList.remove('active', 'hide');
                }, 1200);
            }
        }, remaining);
    });
});

setTimeout(hideLoader, 5000);

window.addEventListener('pageshow', function(event) {
    if (event.persisted) {
        hideLoader();
    }
});
