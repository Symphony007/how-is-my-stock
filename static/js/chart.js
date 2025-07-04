// Initialize all charts on page
document.addEventListener('DOMContentLoaded', function() {
    // PE Ratio Chart
    initPERatioChart();
    
    // Other charts can be initialized here
  });
  
  function initPERatioChart() {
    const ctx = document.getElementById('peRatioChart').getContext('2d');
    const peRatioChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        datasets: [{
          label: 'PE Ratio',
          data: [22, 24, 19, 27, 25, 23],
          borderColor: '#8b5cf6',
          borderWidth: 2,
          pointBackgroundColor: '#a78bfa',
          pointRadius: 5,
          pointHoverRadius: 7,
          tension: 0.1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: {
              font: {
                family: "'Rajdhani', sans-serif",
                size: 14,
                weight: 600
              },
              color: '#c4b5fd',
              padding: 20
            }
          },
          title: {
            display: true,
            text: 'PE Ratio Trend',
            font: {
              family: "'Orbitron', sans-serif",
              size: 18,
              weight: 700
            },
            color: '#8b5cf6',
            padding: {
              top: 10,
              bottom: 20
            }
          },
          tooltip: {
            bodyFont: {
              family: "'Exo 2', sans-serif",
              size: 13,
              weight: 500
            },
            titleFont: {
              family: "'Rajdhani', sans-serif",
              size: 14,
              weight: 600
            },
            backgroundColor: 'rgba(30, 20, 60, 0.95)',
            borderColor: '#8b5cf6',
            borderWidth: 1,
            cornerRadius: 4,
            padding: 12,
            displayColors: false
          }
        },
        scales: {
          x: {
            grid: {
              color: 'rgba(139, 92, 246, 0.2)',
              drawBorder: false
            },
            ticks: {
              font: {
                family: "'Exo 2', sans-serif",
                size: 12,
                weight: 500
              },
              color: '#e0e0ff'
            }
          },
          y: {
            grid: {
              color: 'rgba(139, 92, 246, 0.2)',
              drawBorder: false
            },
            ticks: {
              font: {
                family: "'Exo 2', sans-serif",
                size: 12,
                weight: 500
              },
              color: '#e0e0ff',
              callback: function(value) {
                return value + 'x';
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
  }