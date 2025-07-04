document.addEventListener('DOMContentLoaded', function() {
  const companySearch = document.getElementById('company-search');
  const customDropdown = document.getElementById('custom-dropdown');
  const loadingIndicator = document.querySelector('.loading-indicator');
  const cyberpunkLoader = document.getElementById('cyberpunk-loader');
  
  async function fetchWithLoader(url, options) {
    if (cyberpunkLoader) cyberpunkLoader.classList.add('active');
    try {
      const response = await fetch(url, options);
      return response;
    } finally {
      if (cyberpunkLoader) cyberpunkLoader.classList.remove('active');
    }
  }
  
  if (companySearch) {
    companySearch.addEventListener('input', async function() {
      const query = this.value.trim();
      customDropdown.innerHTML = '';
      
      if (query.length < 2) {
        customDropdown.classList.remove('show');
        loadingIndicator.style.display = 'none';
        return;
      }
      
      loadingIndicator.style.display = 'block';
      customDropdown.classList.remove('show');
      
      try {
        const response = await fetchWithLoader(`/api/search_companies?q=${encodeURIComponent(query)}`);
        const companies = await response.json();
        
        loadingIndicator.style.display = 'none';
        
        if (companies.length > 0) {
          companies.forEach(company => {
            const option = document.createElement('div');
            option.className = 'dropdown-option';
            option.innerHTML = `
              <span class="company-name">${company.name}</span>
              <span class="company-symbol">${company.symbol}</span>
            `;
            option.addEventListener('click', function() {
              companySearch.value = company.name;
              document.getElementById('stock-symbol').value = company.symbol;
              customDropdown.classList.remove('show');
            });
            customDropdown.appendChild(option);
          });
          customDropdown.classList.add('show');
        }
      } catch (error) {
        console.error('Search error:', error);
        loadingIndicator.style.display = 'none';
        if (cyberpunkLoader) cyberpunkLoader.classList.remove('active');
      }
    });
    
    document.addEventListener('click', function(e) {
      if (!companySearch.contains(e.target) && !customDropdown.contains(e.target)) {
        customDropdown.classList.remove('show');
      }
    });
  }
});