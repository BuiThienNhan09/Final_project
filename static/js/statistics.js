// ============================================================
// AI Flight Price - Statistics Dashboard (Multi-Model)
// ============================================================

Chart.defaults.color = '#64748b';
Chart.defaults.borderColor = 'rgba(148, 163, 184, 0.1)';
Chart.defaults.font.family = "'Nunito', sans-serif";

let charts = {};
let currentFilter = 'all';
let statisticsData = null;

const colors = {
    cyan: '#0ea5e9', cyanDim: 'rgba(14, 165, 233, 0.2)',
    purple: '#a855f7', purpleDim: 'rgba(168, 85, 247, 0.2)',
    green: '#10b981', greenDim: 'rgba(16, 185, 129, 0.2)',
    orange: '#f59e0b', orangeDim: 'rgba(245, 158, 11, 0.2)',
    blue: '#3b82f6', blueDim: 'rgba(59, 130, 246, 0.2)',
    red: '#ef4444', redDim: 'rgba(239, 68, 68, 0.2)'
};

const modelColors = {
    'ann': { main: '#3b82f6', dim: 'rgba(59, 130, 246, 0.2)' },
    'linear_regression': { main: '#10b981', dim: 'rgba(16, 185, 129, 0.2)' },
    'decision_tree': { main: '#f59e0b', dim: 'rgba(245, 158, 11, 0.2)' }
};

const modelNames = {
    'ann': 'Neural Network',
    'linear_regression': 'Linear Regression',
    'decision_tree': 'Decision Tree'
};

const modelIcons = {
    'ann': 'fa-brain',
    'linear_regression': 'fa-chart-line',
    'decision_tree': 'fa-sitemap'
};

document.addEventListener('DOMContentLoaded', function() {
    setupTabs();
    setupFilterButtons();
    loadStatistics();
});

function setupTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
        });
    });
}

function setupFilterButtons() {
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentFilter = btn.dataset.filter;
            loadStatistics(currentFilter);
        });
    });
}

async function loadStatistics(modelFilter = 'all') {
    const loadingState = document.getElementById('loadingState');
    const dashboardContent = document.getElementById('dashboardContent');
    loadingState.style.display = 'flex';
    loadingState.innerHTML = '<div class="loader-plane"><i class="fas fa-plane"></i></div><p>Đang tải dữ liệu thống kê...</p>';
    dashboardContent.style.display = 'none';
    
    try {
        const response = await fetch(`/api/statistics?model=${modelFilter}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Statistics data:', data);
        
        if (data.success) {
            statisticsData = data;
            updateDashboard(data);
            loadingState.style.display = 'none';
            dashboardContent.style.display = 'block';
        } else {
            throw new Error(data.error || 'Unknown error');
        }
    } catch (error) {
        console.error('Error loading statistics:', error);
        loadingState.innerHTML = `
            <i class="fas fa-exclamation-circle" style="font-size: 3rem; color: #ef4444; margin-bottom: 1rem;"></i>
            <p>Không thể tải dữ liệu: ${error.message}</p>
            <button class="refresh-btn" onclick="loadStatistics('${modelFilter}')">
                <i class="fas fa-sync-alt"></i> Thử lại
            </button>
        `;
    }
}

function updateDashboard(data) {
    updateOverviewStats(data);
    updateModelComparison(data.model_evaluation, data.search_history?.model_distribution || {});
    updateTrainingStats(data.training_data);
    updateSearchHistory(data.search_history);
}

function updateOverviewStats(data) {
    document.getElementById('totalTrainingSamples').textContent = formatNumber(data.training_data?.total_flights || 0);
    document.getElementById('totalSearches').textContent = formatNumber(data.search_history?.total_searches || 0);
    document.getElementById('avgPredictedPrice').textContent = formatPrice(data.search_history?.avg_price || data.training_data?.avg_price || 0);
    
    // Model cards overview
    createModelCardsOverview(data.model_evaluation, data.search_history?.model_distribution || {});
    
    // Charts
    createOverviewRoutesChart(data.search_history?.routes || data.training_data?.top_routes || []);
    createModelDistributionChart(data.search_history?.model_distribution || {});
}

function createModelCardsOverview(modelEval, modelDist) {
    const grid = document.getElementById('modelCardsGrid');
    if (!grid) return;
    
    let html = '';
    const models = ['ann', 'linear_regression', 'decision_tree'];
    
    models.forEach(model => {
        const eval_data = modelEval[model] || {};
        const usage = modelDist[model] || 0;
        const color = modelColors[model]?.main || '#3b82f6';
        const icon = modelIcons[model] || 'fa-robot';
        const name = modelNames[model] || model;
        const accuracy = eval_data.accuracy || 0;
        
        html += `
            <div class="model-overview-card" style="--model-color: ${color}">
                <div class="model-overview-icon" style="background: ${color}">
                    <i class="fas ${icon}"></i>
                </div>
                <div class="model-overview-info">
                    <h4>${name}</h4>
                    <div class="model-overview-stats">
                        <span class="accuracy" style="color: ${color}">${accuracy.toFixed(1)}%</span>
                        <span class="divider">•</span>
                        <span class="usage">${formatNumber(usage)} lượt</span>
                    </div>
                </div>
            </div>
        `;
    });
    
    grid.innerHTML = html;
}

function updateModelComparison(modelEval, modelDist) {
    if (!modelEval) return;
    
    // ANN
    if (modelEval.ann) {
        updateModelCard('ann', modelEval.ann, modelDist.ann || 0);
    }
    
    // Linear Regression
    if (modelEval.linear_regression) {
        updateModelCard('lr', modelEval.linear_regression, modelDist.linear_regression || 0);
    }
    
    // Decision Tree
    if (modelEval.decision_tree) {
        updateModelCard('dt', modelEval.decision_tree, modelDist.decision_tree || 0);
    }
    
    // Create comparison charts
    createAccuracyCompareChart(modelEval);
    createMAECompareChart(modelEval);
    createRadarCompareChart(modelEval);
}

function updateModelCard(prefix, data, usage) {
    const accuracy = data.accuracy || 0;
    
    document.getElementById(`${prefix}Accuracy`).textContent = accuracy.toFixed(1) + '%';
    document.getElementById(`${prefix}MAE`).textContent = formatPrice(data.mae || 0);
    document.getElementById(`${prefix}RMSE`).textContent = formatPrice(data.rmse || 0);
    document.getElementById(`${prefix}R2`).textContent = (data.r2 || 0).toFixed(4);
    document.getElementById(`${prefix}Usage`).textContent = formatNumber(usage);
    
    // Update circle
    const circle = document.getElementById(`${prefix}AccuracyCircle`);
    if (circle) {
        const circumference = 2 * Math.PI * 50;
        circle.style.strokeDasharray = circumference;
        circle.style.strokeDashoffset = circumference - (accuracy / 100) * circumference;
    }
}

function createAccuracyCompareChart(modelEval) {
    const ctx = document.getElementById('accuracyCompareChart');
    if (!ctx) return;
    destroyChart('accuracyCompareChart');
    
    const models = ['ann', 'linear_regression', 'decision_tree'];
    const accuracies = models.map(m => modelEval[m]?.accuracy || 0);
    
    charts.accuracyCompareChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: models.map(m => modelNames[m]),
            datasets: [{
                data: accuracies,
                backgroundColor: models.map(m => modelColors[m].main),
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, max: 100, title: { display: true, text: 'Accuracy (%)' } }
            }
        }
    });
}

function createMAECompareChart(modelEval) {
    const ctx = document.getElementById('maeCompareChart');
    if (!ctx) return;
    destroyChart('maeCompareChart');
    
    const models = ['ann', 'linear_regression', 'decision_tree'];
    const maes = models.map(m => modelEval[m]?.mae || 0);
    
    charts.maeCompareChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: models.map(m => modelNames[m]),
            datasets: [{
                data: maes,
                backgroundColor: models.map(m => modelColors[m].dim),
                borderColor: models.map(m => modelColors[m].main),
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, title: { display: true, text: 'MAE (VNĐ)' } }
            }
        }
    });
}

function createRadarCompareChart(modelEval) {
    const ctx = document.getElementById('radarCompareChart');
    if (!ctx) return;
    destroyChart('radarCompareChart');
    
    const models = ['ann', 'linear_regression', 'decision_tree'];
    
    // Normalize metrics for radar chart
    const maxMAE = Math.max(...models.map(m => modelEval[m]?.mae || 0));
    const maxRMSE = Math.max(...models.map(m => modelEval[m]?.rmse || 0));
    
    const datasets = models.map(model => {
        const data = modelEval[model] || {};
        return {
            label: modelNames[model],
            data: [
                data.accuracy || 0,
                data.r2 ? data.r2 * 100 : 0,
                maxMAE ? (1 - (data.mae || 0) / maxMAE) * 100 : 0,
                maxRMSE ? (1 - (data.rmse || 0) / maxRMSE) * 100 : 0
            ],
            backgroundColor: modelColors[model].dim,
            borderColor: modelColors[model].main,
            borderWidth: 2,
            pointBackgroundColor: modelColors[model].main
        };
    });
    
    charts.radarCompareChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Accuracy (%)', 'R² Score (%)', 'MAE (inverse)', 'RMSE (inverse)'],
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { stepSize: 20 }
                }
            },
            plugins: {
                legend: { position: 'bottom' }
            }
        }
    });
}

function createModelDistributionChart(distribution) {
    const ctx = document.getElementById('modelDistributionChart');
    if (!ctx) return;
    destroyChart('modelDistributionChart');
    
    const labels = Object.keys(distribution).map(k => modelNames[k] || k);
    const data = Object.values(distribution);
    const backgroundColors = Object.keys(distribution).map(k => modelColors[k]?.main || colors.blue);
    
    if (data.length === 0 || data.every(v => v === 0)) {
        charts.modelDistributionChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Chưa có dữ liệu'],
                datasets: [{ data: [1], backgroundColor: ['#e2e8f0'] }]
            },
            options: { responsive: true, maintainAspectRatio: false, cutout: '60%' }
        });
        return;
    }
    
    charts.modelDistributionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: backgroundColors,
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '60%',
            plugins: { legend: { position: 'bottom' } }
        }
    });
}

function updateTrainingStats(training) {
    if (!training) return;
    document.getElementById('trainTotalFlights').textContent = formatNumber(training.total_flights);
    document.getElementById('trainNumAirlines').textContent = training.num_airlines;
    document.getElementById('trainNumRoutes').textContent = training.num_routes;
    document.getElementById('trainAvgPrice').textContent = formatPrice(training.avg_price);
    
    if (training.price_ranges) {
        const container = document.getElementById('priceRanges');
        const total = Object.values(training.price_ranges).reduce((a, b) => a + b, 0);
        container.innerHTML = Object.entries(training.price_ranges).map(([range, count]) => {
            const percent = (count / total * 100).toFixed(1);
            return `<div class="price-range"><span class="range-label">${range}</span><span class="range-value">${formatNumber(count)}</span><div class="range-bar"><div class="range-bar-fill" style="width: ${percent}%"></div></div></div>`;
        }).join('');
    }
    createAirlinePriceChart(training.airlines);
    createClassChart(training.classes);
    createMonthChart(training.months);
    createTopRoutesChart(training.top_routes);
}

function updateSearchHistory(searchData) {
    const noSearchMessage = document.getElementById('noSearchMessage');
    const searchHistoryContent = document.getElementById('searchHistoryContent');
    if (!searchData || searchData.total_searches === 0) {
        noSearchMessage.style.display = 'flex';
        searchHistoryContent.style.display = 'none';
        return;
    }
    noSearchMessage.style.display = 'none';
    searchHistoryContent.style.display = 'block';
    createSearchRoutesChart(searchData.routes || []);
    createSearchAirlinesChart(searchData.airlines || []);
    createSearchPriceChart(searchData.airline_prices || []);
    createSearchClassChart(searchData.classes || {});
    createSearchTimelineChart(searchData.timeline || []);
    updateRecentSearchesTable(searchData.recent_searches || []);
}

function destroyChart(name) { if (charts[name]) { charts[name].destroy(); delete charts[name]; } }

function createOverviewRoutesChart(routes) {
    const ctx = document.getElementById('overviewRoutesChart'); if (!ctx) return;
    destroyChart('overviewRoutesChart');
    charts.overviewRoutesChart = new Chart(ctx, {
        type: 'bar',
        data: { labels: routes.slice(0, 5).map(r => r.route || r), datasets: [{ data: routes.slice(0, 5).map(r => r.count || r.mean || 1), backgroundColor: [colors.cyan, colors.purple, colors.green, colors.orange, colors.blue], borderRadius: 8 }] },
        options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { display: false } }, y: { grid: { display: false } } } }
    });
}

function createAirlinePriceChart(airlines) {
    const ctx = document.getElementById('trainAirlinePriceChart'); if (!ctx || !airlines) return;
    destroyChart('trainAirlinePriceChart');
    const sorted = Object.entries(airlines).sort((a, b) => b[1].mean - a[1].mean).slice(0, 8);
    charts.trainAirlinePriceChart = new Chart(ctx, {
        type: 'bar',
        data: { labels: sorted.map(a => a[0]), datasets: [{ data: sorted.map(a => a[1].mean), backgroundColor: colors.cyanDim, borderColor: colors.cyan, borderWidth: 2, borderRadius: 6 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { display: false } } } }
    });
}

function createClassChart(classes) {
    const ctx = document.getElementById('trainClassChart'); if (!ctx || !classes) return;
    destroyChart('trainClassChart');
    charts.trainClassChart = new Chart(ctx, {
        type: 'doughnut',
        data: { labels: Object.keys(classes).map(c => c === 'ECONOMY' ? 'Phổ thông' : 'Thương gia'), datasets: [{ data: Object.values(classes).map(c => c.count), backgroundColor: [colors.cyan, colors.purple], borderWidth: 0 }] },
        options: { responsive: true, maintainAspectRatio: false, cutout: '65%', plugins: { legend: { position: 'bottom' } } }
    });
}

function createMonthChart(months) {
    const ctx = document.getElementById('trainMonthChart'); if (!ctx || !months) return;
    destroyChart('trainMonthChart');
    const monthNames = ['', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12'];
    const sorted = Object.entries(months).sort((a, b) => parseInt(a[0]) - parseInt(b[0]));
    charts.trainMonthChart = new Chart(ctx, {
        type: 'line',
        data: { labels: sorted.map(m => monthNames[parseInt(m[0])]), datasets: [{ data: sorted.map(m => m[1].mean), borderColor: colors.purple, backgroundColor: colors.purpleDim, fill: true, tension: 0.4 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
    });
}

function createTopRoutesChart(routes) {
    const ctx = document.getElementById('trainTopRoutesChart'); if (!ctx || !routes) return;
    destroyChart('trainTopRoutesChart');
    charts.trainTopRoutesChart = new Chart(ctx, {
        type: 'bar',
        data: { labels: routes.slice(0, 10).map(r => r.route), datasets: [{ data: routes.slice(0, 10).map(r => r.mean), backgroundColor: colors.orangeDim, borderColor: colors.orange, borderWidth: 2, borderRadius: 6 }] },
        options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
    });
}

function createSearchRoutesChart(routes) {
    const ctx = document.getElementById('searchRoutesChart'); if (!ctx) return;
    destroyChart('searchRoutesChart');
    charts.searchRoutesChart = new Chart(ctx, {
        type: 'bar',
        data: { labels: routes.slice(0, 10).map(r => r.route), datasets: [{ data: routes.slice(0, 10).map(r => r.count), backgroundColor: colors.cyanDim, borderColor: colors.cyan, borderWidth: 2, borderRadius: 6 }] },
        options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
    });
}

function createSearchAirlinesChart(airlines) {
    const ctx = document.getElementById('searchAirlinesChart'); if (!ctx) return;
    destroyChart('searchAirlinesChart');
    charts.searchAirlinesChart = new Chart(ctx, {
        type: 'pie',
        data: { labels: airlines.slice(0, 5).map(a => a.airline), datasets: [{ data: airlines.slice(0, 5).map(a => a.count), backgroundColor: [colors.cyan, colors.purple, colors.green, colors.orange, colors.blue], borderWidth: 0 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } }
    });
}

function createSearchPriceChart(airlinePrices) {
    const ctx = document.getElementById('searchPriceChart'); if (!ctx) return;
    destroyChart('searchPriceChart');
    charts.searchPriceChart = new Chart(ctx, {
        type: 'bar',
        data: { labels: airlinePrices.slice(0, 8).map(a => a.airline), datasets: [{ data: airlinePrices.slice(0, 8).map(a => a.avg_price), backgroundColor: colors.greenDim, borderColor: colors.green, borderWidth: 2, borderRadius: 6 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
    });
}

function createSearchClassChart(classes) {
    const ctx = document.getElementById('searchClassChart'); if (!ctx) return;
    destroyChart('searchClassChart');
    charts.searchClassChart = new Chart(ctx, {
        type: 'doughnut',
        data: { labels: Object.keys(classes).map(c => c === 'ECONOMY' ? 'Phổ thông' : 'Thương gia'), datasets: [{ data: Object.values(classes), backgroundColor: [colors.cyan, colors.purple], borderWidth: 0 }] },
        options: { responsive: true, maintainAspectRatio: false, cutout: '65%', plugins: { legend: { position: 'bottom' } } }
    });
}

function createSearchTimelineChart(timeline) {
    const ctx = document.getElementById('searchTimelineChart'); if (!ctx) return;
    destroyChart('searchTimelineChart');
    charts.searchTimelineChart = new Chart(ctx, {
        type: 'line',
        data: { labels: timeline.map(t => t.date), datasets: [{ data: timeline.map(t => t.count), borderColor: colors.cyan, backgroundColor: colors.cyanDim, fill: true, tension: 0.4, pointRadius: 4, pointBackgroundColor: colors.cyan }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { display: false } }, y: { beginAtZero: true } } }
    });
}

function updateRecentSearchesTable(searches) {
    const tbody = document.getElementById('recentSearchesTable'); if (!tbody) return;
    if (!searches.length) { tbody.innerHTML = '<tr><td colspan="7" class="loading-cell">Chưa có dữ liệu</td></tr>'; return; }
    
    tbody.innerHTML = searches.map(s => {
        const modelType = s.model_type || 'ann';
        const modelColor = modelColors[modelType]?.main || '#3b82f6';
        const modelIcon = modelIcons[modelType] || 'fa-robot';
        const modelName = modelNames[modelType] || modelType;
        
        return `<tr>
            <td>${formatDateTime(s.timestamp)}</td>
            <td><span class="model-badge-small" style="background: ${modelColor}"><i class="fas ${modelIcon}"></i> ${modelName.split(' ')[0]}</span></td>
            <td>${s.airline}</td>
            <td><strong>${s.origin}</strong> → <strong>${s.destination}</strong></td>
            <td>${formatDate(s.flight_date)}</td>
            <td>${s.class === 'ECONOMY' ? 'Phổ thông' : 'Thương gia'}</td>
            <td style="color: var(--sky-blue); font-weight: 700;">${formatPrice(s.predicted_price)} đ</td>
        </tr>`;
    }).join('');
}

function formatNumber(num) { if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M'; if (num >= 1000) return (num / 1000).toFixed(1) + 'K'; return num.toLocaleString('vi-VN'); }
function formatPrice(price) { if (price >= 1000000) return (price / 1000000).toFixed(1) + 'M'; if (price >= 1000) return (price / 1000).toFixed(0) + 'K'; return Math.round(price).toLocaleString('vi-VN'); }
function formatPercent(value) { return value.toFixed(1) + '%'; }
function formatDateTime(timestamp) { if (!timestamp) return '-'; const d = new Date(timestamp); return d.toLocaleDateString('vi-VN') + ' ' + d.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' }); }
function formatDate(dateStr) { if (!dateStr) return '-'; const [y, m, d] = dateStr.split('-'); return `${d}/${m}/${y}`; }

window.loadStatistics = loadStatistics;