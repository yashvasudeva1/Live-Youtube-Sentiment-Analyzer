document.addEventListener('DOMContentLoaded', () => {
  const analyzeBtn    = document.getElementById('analyze-btn');
  const loadingDiv    = document.getElementById('loading');
  const resultsDiv    = document.getElementById('results');
  const errorDiv      = document.getElementById('error');
  const errorMessage  = document.getElementById('error-message');
  const statusText    = document.getElementById('status-text');

  let pieChartInstance  = null;
  let barChartInstance  = null;
  let lineChartInstance = null;

  analyzeBtn.addEventListener('click', async () => {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab || !tab.url) { showError('Could not get active tab.'); return; }

      const url = new URL(tab.url);
      if (!url.hostname.includes('youtube.com') || url.pathname !== '/watch') {
        showError('Please navigate to a YouTube video first.'); return;
      }

      const videoId = url.searchParams.get('v');
      if (!videoId) { showError('Could not extract a video ID from the URL.'); return; }

      showLoading(`Fetching and analyzing comments...`);

      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: videoId }),
      });

      if (!response.ok) throw new Error(`Server returned ${response.status}`);
      const data = await response.json();
      renderResults(data);

    } catch (err) {
      showError(`Analysis failed: ${err.message}`);
    }
  });

  /* -------- State Helpers -------- */

  function showLoading(msg) {
    statusText.textContent = msg;
    loadingDiv.classList.remove('hidden');
    resultsDiv.classList.add('hidden');
    errorDiv.classList.add('hidden');
    analyzeBtn.disabled = true;
  }

  function showError(msg) {
    statusText.textContent = 'Error';
    errorMessage.textContent = msg;
    errorDiv.classList.remove('hidden');
    loadingDiv.classList.add('hidden');
    resultsDiv.classList.add('hidden');
    analyzeBtn.disabled = false;
  }

  /* -------- Render -------- */

  function renderResults(data) {
    const totalComments   = data.total_comments        || 0;
    const avgPerMonth     = data.avg_comments_per_month != null ? data.avg_comments_per_month : 0;
    const dominantSent    = data.dominant_sentiment    || 'neutral';
    const sentiments      = data.sentiments            || {};
    const sentimentPct    = data.sentiment_pct         || {};
    const monthlyData     = data.monthly_data          || [];
    const topPositive     = data.top_positive          || [];
    const topNegative     = data.top_negative          || [];

    statusText.textContent = `${totalComments.toLocaleString()} comments analyzed`;
    loadingDiv.classList.add('hidden');
    errorDiv.classList.add('hidden');
    resultsDiv.classList.remove('hidden');
    analyzeBtn.disabled = false;

    // Stat Cards
    document.getElementById('total-comments').textContent = totalComments.toLocaleString();
    document.getElementById('avg-per-month').textContent  = avgPerMonth.toLocaleString();

    const dominantEl = document.getElementById('dominant-sentiment');
    dominantEl.textContent = capitalise(dominantSent);
    dominantEl.className   = `stat-value dominant ${dominantSent}`;

    const peakEl = document.getElementById('most-active-month');
    peakEl.textContent = data.most_active_month ? formatMonth(data.most_active_month) : '—';

    // Breakdown bars
    setBreakdown('positive', sentiments.positive || 0, sentimentPct.positive || 0);
    setBreakdown('negative', sentiments.negative || 0, sentimentPct.negative || 0);
    setBreakdown('neutral',  sentiments.neutral  || 0, sentimentPct.neutral  || 0);

    // Charts
    renderPieChart(sentiments);
    renderBarChart(monthlyData);
    renderLineChart(monthlyData);

    // Top comments
    renderComments('positive-comments', topPositive, 'positive-card');
    renderComments('negative-comments', topNegative, 'negative-card');
  }

  function setBreakdown(type, count, pct) {
    const safeCount = count != null ? count : 0;
    const safePct   = pct   != null ? pct   : 0;
    document.getElementById(`pct-${type}`).textContent   = `${safePct}%`;
    document.getElementById(`bar-${type}`).style.width   = `${safePct}%`;
    document.getElementById(`count-${type}`).textContent = `${safeCount.toLocaleString()} comments`;
  }

  /* -------- Charts -------- */

  const POSITIVE_COLOR = '#16a34a';
  const NEGATIVE_COLOR = '#dc2626';
  const NEUTRAL_COLOR  = '#475569';

  function renderPieChart(sentiments) {
    destroyChart(pieChartInstance);
    const ctx = document.getElementById('pieChart').getContext('2d');
    pieChartInstance = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Positive', 'Negative', 'Neutral'],
        datasets: [{
          data: [sentiments.positive || 0, sentiments.negative || 0, sentiments.neutral || 0],
          backgroundColor: [POSITIVE_COLOR, NEGATIVE_COLOR, NEUTRAL_COLOR],
          borderColor: '#ffffff',
          borderWidth: 3,
          hoverOffset: 6,
        }],
      },
      options: {
        responsive: true,
        cutout: '60%',
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              boxWidth: 10,
              boxHeight: 10,
              borderRadius: 5,
              useBorderRadius: true,
              padding: 14,
              font: { size: 12, family: 'system-ui' },
              color: '#374151',
            },
          },
          tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.raw.toLocaleString()}` } },
        },
      },
    });
  }

  function renderBarChart(monthlyData) {
    destroyChart(barChartInstance);
    const ctx = document.getElementById('barChart').getContext('2d');
    barChartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: monthlyData.map(d => formatMonth(d.month)),
        datasets: [
          { label: 'Positive', data: monthlyData.map(d => d.positive), backgroundColor: POSITIVE_COLOR, borderRadius: 3 },
          { label: 'Negative', data: monthlyData.map(d => d.negative), backgroundColor: NEGATIVE_COLOR, borderRadius: 3 },
          { label: 'Neutral',  data: monthlyData.map(d => d.neutral),  backgroundColor: NEUTRAL_COLOR,  borderRadius: 3 },
        ],
      },
      options: {
        responsive: true,
        scales: {
          x: {
            stacked: true,
            grid: { display: false },
            ticks: { font: { size: 11 }, color: '#6b7280', maxRotation: 45 },
          },
          y: {
            stacked: true,
            beginAtZero: true,
            grid: { color: '#f3f4f6' },
            ticks: { font: { size: 11 }, color: '#6b7280' },
          },
        },
        plugins: {
          legend: {
            position: 'bottom',
            labels: { boxWidth: 10, boxHeight: 10, borderRadius: 5, useBorderRadius: true, padding: 12, font: { size: 12 }, color: '#374151' },
          },
        },
      },
    });
  }

  function renderLineChart(monthlyData) {
    destroyChart(lineChartInstance);
    const ctx = document.getElementById('lineChart').getContext('2d');
    lineChartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels: monthlyData.map(d => formatMonth(d.month)),
        datasets: [
          {
            label: 'Positive',
            data: monthlyData.map(d => d.positive),
            borderColor: POSITIVE_COLOR,
            backgroundColor: POSITIVE_COLOR + '22',
            fill: false,
            tension: 0.35,
            pointRadius: 4,
            pointHoverRadius: 6,
            borderWidth: 2,
          },
          {
            label: 'Negative',
            data: monthlyData.map(d => d.negative),
            borderColor: NEGATIVE_COLOR,
            backgroundColor: NEGATIVE_COLOR + '22',
            fill: false,
            tension: 0.35,
            pointRadius: 4,
            pointHoverRadius: 6,
            borderWidth: 2,
          },
          {
            label: 'Neutral',
            data: monthlyData.map(d => d.neutral),
            borderColor: NEUTRAL_COLOR,
            backgroundColor: NEUTRAL_COLOR + '22',
            fill: false,
            tension: 0.35,
            pointRadius: 4,
            pointHoverRadius: 6,
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        interaction: { mode: 'index', intersect: false },
        scales: {
          x: {
            grid: { display: false },
            ticks: { font: { size: 11 }, color: '#6b7280', maxRotation: 45 },
          },
          y: {
            beginAtZero: true,
            grid: { color: '#f3f4f6' },
            ticks: { font: { size: 11 }, color: '#6b7280' },
          },
        },
        plugins: {
          legend: {
            position: 'bottom',
            labels: { boxWidth: 10, boxHeight: 10, borderRadius: 5, useBorderRadius: true, padding: 12, font: { size: 12 }, color: '#374151' },
          },
        },
      },
    });
  }

  function renderComments(containerId, comments, cardClass) {
    const el = document.getElementById(containerId);
    el.innerHTML = '';
    if (!comments || comments.length === 0) {
      el.innerHTML = '<p style="color:#9ca3af;font-size:12px">No comments found.</p>';
      return;
    }
    comments.forEach(text => {
      const card = document.createElement('div');
      card.className = `comment-card ${cardClass}`;
      card.textContent = text.length > 200 ? text.slice(0, 200) + '…' : text;
      el.appendChild(card);
    });
  }

  /* -------- Utilities -------- */

  function destroyChart(instance) {
    if (instance) instance.destroy();
  }

  function capitalise(str) {
    if (!str) return '—';
    return str.charAt(0).toUpperCase() + str.slice(1);
  }

  function formatMonth(ym) {
    if (!ym || ym === 'Unknown') return ym;
    const [year, month] = ym.split('-');
    const d = new Date(+year, +month - 1);
    return d.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
  }
});
