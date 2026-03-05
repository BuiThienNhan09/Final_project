// ============================================================
// AI Flight Price - Multi-Model Support
// ============================================================

let currentSlide = 0;
let isScrolling = false;
let touchStartY = 0;
const totalSlides = 4;
let validRoutesData = {};
let airplanesCreated = false;
let selectedModel = 'ann';

const routeDurations = {
    'SGN-HAN': 120, 'HAN-SGN': 120, 'SGN-DAD': 75, 'DAD-SGN': 75,
    'SGN-PQC': 60, 'PQC-SGN': 60, 'SGN-CXR': 60, 'CXR-SGN': 60,
    'SGN-DLI': 50, 'DLI-SGN': 50, 'HAN-DAD': 70, 'DAD-HAN': 70,
    'HAN-PQC': 140, 'PQC-HAN': 140
};

const modelIcons = {
    'ann': 'fa-brain',
    'linear_regression': 'fa-chart-line',
    'decision_tree': 'fa-sitemap'
};

const modelColors = {
    'ann': '#3b82f6',
    'linear_regression': '#10b981',
    'decision_tree': '#f59e0b'
};

document.addEventListener('DOMContentLoaded', function() {
    createAirplanes();
    initializeSlides();
    initializeNavDots();
    initializeScrollProgress();
    initializeForm();
    initializeModelSelector();
    updateActiveSlide(0);
    updateEffects(0);
});

function initializeModelSelector() {
    const modelChips = document.querySelectorAll('.model-chip');
    modelChips.forEach(chip => {
        chip.addEventListener('click', function() {
            modelChips.forEach(c => c.classList.remove('active'));
            this.classList.add('active');
            const radio = this.querySelector('input[type="radio"]');
            radio.checked = true;
            selectedModel = radio.value;
            console.log('Selected model:', selectedModel);
        });
    });
}

function triggerLacBirdFlyAway() {
    var c = document.getElementById('lacBirdContainer');
    if (c && !c.classList.contains('fly-away')) {
        c.classList.add('fly-away');
        setTimeout(function() { c.classList.add('hidden'); }, 1200);
    }
}

function resetLacBirds() {
    var c = document.getElementById('lacBirdContainer');
    if (c) c.classList.remove('fly-away', 'hidden');
}

function createAirplanes() {
    if (airplanesCreated) return;
    var container = document.getElementById('airplanesContainer');
    if (!container) return;
    var svg = '<svg viewBox="0 0 80 40"><ellipse cx="40" cy="20" rx="28" ry="7" fill="#fff"/><ellipse cx="63" cy="20" rx="10" ry="5" fill="#e0f2fe"/><path d="M25 20 L38 7 L48 9 L40 20 L48 31 L38 33 Z" fill="#fff"/><path d="M10 20 L4 13 L18 17 L18 23 L4 27 Z" fill="#fff"/><circle cx="50" cy="18" r="2" fill="#0ea5e9"/><circle cx="44" cy="18" r="2" fill="#0ea5e9"/></svg>';
    var planes = [{c:'mini-plane-1',s:55},{c:'mini-plane-2',s:45},{c:'mini-plane-3',s:50},{c:'mini-plane-4',s:42},{c:'mini-plane-5',s:48}];
    planes.forEach(function(p) {
        var el = document.createElement('div');
        el.className = 'mini-plane ' + p.c;
        el.style.width = p.s + 'px';
        el.style.height = (p.s/2) + 'px';
        el.innerHTML = svg;
        container.appendChild(el);
    });
    airplanesCreated = true;
}

function showAirplanes() {
    var c = document.getElementById('airplanesContainer');
    if (c) c.classList.add('visible');
}

function hideAirplanes() {
    var c = document.getElementById('airplanesContainer');
    if (c) c.classList.remove('visible');
}

function updateEffects(idx) {
    if (idx === 0) { resetLacBirds(); hideAirplanes(); }
    else { triggerLacBirdFlyAway(); showAirplanes(); }
}

function initializeSlides() {
    var container = document.getElementById('slidesContainer');
    if (!container) return;
    container.addEventListener('wheel', handleWheel, { passive: false });
    container.addEventListener('touchstart', handleTouchStart, { passive: true });
    container.addEventListener('touchend', handleTouchEnd, { passive: true });
    container.addEventListener('scroll', handleScroll);
}

function handleWheel(e) {
    if (isScrolling) return;
    if (Math.abs(e.deltaY) > 50) {
        if (e.deltaY > 0 && currentSlide < totalSlides - 1) goToSlide(currentSlide + 1);
        else if (e.deltaY < 0 && currentSlide > 0) goToSlide(currentSlide - 1);
    }
}

function handleTouchStart(e) { touchStartY = e.touches[0].clientY; }

function handleTouchEnd(e) {
    if (isScrolling) return;
    var delta = touchStartY - e.changedTouches[0].clientY;
    if (Math.abs(delta) > 50) {
        if (delta > 0 && currentSlide < totalSlides - 1) goToSlide(currentSlide + 1);
        else if (delta < 0 && currentSlide > 0) goToSlide(currentSlide - 1);
    }
}

function handleScroll() {
    var container = document.getElementById('slidesContainer');
    if (!container) return;
    var progress = container.scrollTop / (container.scrollHeight - container.clientHeight);
    var bar = document.getElementById('scrollProgressBar');
    if (bar) bar.style.width = (progress * 100) + '%';
    var newSlide = Math.round(container.scrollTop / container.clientHeight);
    if (newSlide !== currentSlide && newSlide >= 0 && newSlide < totalSlides) {
        currentSlide = newSlide;
        updateNavDots();
        updateActiveSlide(newSlide);
        updateEffects(newSlide);
    }
}

function goToSlide(index) {
    if (index < 0 || index >= totalSlides || isScrolling) return;
    isScrolling = true;
    currentSlide = index;
    var container = document.getElementById('slidesContainer');
    if (container) container.scrollTo({ top: index * container.clientHeight, behavior: 'smooth' });
    updateNavDots();
    updateActiveSlide(index);
    updateEffects(index);
    setTimeout(function() { isScrolling = false; }, 800);
}

function updateActiveSlide(index) {
    document.querySelectorAll('.slide').forEach(function(slide, i) {
        if (i === index) slide.classList.add('active');
        else slide.classList.remove('active');
    });
}

function initializeNavDots() {
    document.querySelectorAll('.slide-nav .nav-dot').forEach(function(dot, i) {
        dot.addEventListener('click', function() { goToSlide(i); });
    });
}

function updateNavDots() {
    document.querySelectorAll('.slide-nav .nav-dot').forEach(function(dot, i) {
        if (i === currentSlide) dot.classList.add('active');
        else dot.classList.remove('active');
    });
}

function initializeScrollProgress() {
    var container = document.getElementById('slidesContainer');
    if (!container) return;
    container.addEventListener('scroll', function() {
        var progress = container.scrollTop / (container.scrollHeight - container.clientHeight);
        var bar = document.getElementById('scrollProgressBar');
        if (bar) bar.style.width = (progress * 100) + '%';
    });
}

async function initializeForm() {
    await loadValidRoutes();
    setupDateInput();
    setupDynamicDestinations();
    setupTripTypeToggle();
    setupFormSubmission();
}

async function loadValidRoutes() {
    try {
        var res = await fetch('/api/valid-routes');
        var data = await res.json();
        if (data.success) validRoutesData = data.routes;
    } catch (e) { console.error(e); }
}

function setupDateInput() {
    var dateInput = document.getElementById('flight_date');
    var returnInput = document.getElementById('return_date');
    if (dateInput) {
        var today = new Date().toISOString().split('T')[0];
        dateInput.setAttribute('min', today);
        dateInput.value = today;
        dateInput.addEventListener('change', function() {
            if (returnInput) {
                var min = new Date(this.value);
                min.setDate(min.getDate() + 1);
                returnInput.setAttribute('min', min.toISOString().split('T')[0]);
            }
        });
    }
}

function setupDynamicDestinations() {
    var originSelect = document.getElementById('origin');
    if (originSelect) {
        originSelect.addEventListener('change', updateDestinationDropdown);
    }
}

function updateDestinationDropdown() {
    var originSelect = document.getElementById('origin');
    var destSelect = document.getElementById('destination');
    var origin = originSelect.value;
    var currentDest = destSelect.value;
    
    destSelect.innerHTML = '<option value="">Chọn điểm đến</option>';
    
    if (origin && validRoutesData[origin]) {
        validRoutesData[origin].forEach(function(dest) {
            var opt = document.createElement('option');
            opt.value = dest;
            opt.textContent = (airports[dest] || dest) + ' (' + dest + ')';
            if (dest === currentDest) opt.selected = true;
            destSelect.appendChild(opt);
        });
    }
}

function setupTripTypeToggle() {
    var radios = document.querySelectorAll('input[name="trip_type"]');
    var returnGroup = document.getElementById('return_date_group');
    var returnInput = document.getElementById('return_date');
    if (!returnGroup || !returnInput) return;
    radios.forEach(function(r) {
        r.addEventListener('change', function() {
            if (this.value === 'roundtrip') {
                returnGroup.style.display = 'block';
                returnInput.required = true;
                var depart = document.getElementById('flight_date').value;
                if (depart) {
                    var min = new Date(depart);
                    min.setDate(min.getDate() + 1);
                    returnInput.setAttribute('min', min.toISOString().split('T')[0]);
                }
            } else {
                returnGroup.style.display = 'none';
                returnInput.required = false;
                returnInput.value = '';
            }
        });
    });
}

function swapLocations() {
    var origin = document.getElementById('origin');
    var dest = document.getElementById('destination');
    var o = origin.value, d = dest.value;
    if (!o) { showToast('Vui lòng chọn điểm đi trước', 'warning'); return; }
    if (!d) { showToast('Vui lòng chọn cả điểm đi và điểm đến', 'warning'); return; }
    var rev = validRoutesData[d];
    if (rev && rev.includes(o)) {
        origin.value = d;
        updateDestinationDropdown();
        setTimeout(function() { dest.value = o; }, 50);
        var btn = document.querySelector('.swap-btn-mini');
        if (btn) { btn.style.transform = 'rotate(180deg)'; setTimeout(function() { btn.style.transform = ''; }, 300); }
    } else {
        showToast('Không có chuyến bay từ ' + d + ' về ' + o, 'error');
    }
}

function calculateDuration() {
    var o = document.getElementById('origin').value;
    var d = document.getElementById('destination').value;
    if (o && d) {
        var dur = routeDurations[o + '-' + d] || 90;
        document.getElementById('duration').value = dur;
        document.getElementById('departure_hour').value = 12;
        document.getElementById('arrival_hour').value = (12 + Math.floor(dur / 60)) % 24;
        return dur;
    }
    return 120;
}

function setupFormSubmission() {
    var form = document.getElementById('searchForm');
    if (!form) return;
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        if (!validateForm()) return;
        var formData = collectFormData();
        var btn = document.getElementById('searchBtn');
        btn.classList.add('loading');
        btn.disabled = true;
        try {
            var res = await fetch('/predict', { 
                method: 'POST', 
                headers: { 'Content-Type': 'application/json' }, 
                body: JSON.stringify(formData) 
            });
            var data = await res.json();
            if (data.success) { 
                displayResults(data, formData); 
                goToSlide(3); 
                showToast('Dự đoán thành công với ' + (data.model_name || 'AI') + '!', 'success'); 
            }
            else showToast(data.error || 'Dự đoán thất bại', 'error');
        } catch (err) { showToast('Lỗi kết nối: ' + err.message, 'error'); }
        finally { btn.classList.remove('loading'); btn.disabled = false; }
    });
}

function validateForm() {
    var o = document.getElementById('origin').value;
    var d = document.getElementById('destination').value;
    var dt = document.getElementById('flight_date').value;
    var al = document.getElementById('airline').value;
    var cl = document.getElementById('seat_class').value;
    if (!o || !d) { showToast('Vui lòng chọn điểm đi và điểm đến!', 'error'); return false; }
    if (o === d) { showToast('Điểm đi và điểm đến phải khác nhau!', 'error'); return false; }
    var valid = validRoutesData[o];
    if (!valid || !valid.includes(d)) { showToast('Tuyến bay ' + o + ' - ' + d + ' chưa được hỗ trợ.', 'error'); return false; }
    if (!dt) { showToast('Vui lòng chọn ngày bay!', 'error'); return false; }
    if (!al) { showToast('Vui lòng chọn hãng bay!', 'error'); return false; }
    if (!cl) { showToast('Vui lòng chọn hạng vé!', 'error'); return false; }
    return true;
}

function collectFormData() {
    var dur = calculateDuration();
    var trip = document.querySelector('input[name="trip_type"]:checked');
    var modelRadio = document.querySelector('input[name="model"]:checked');
    var fd = document.getElementById('flight_date').value;
    var dt = new Date(fd);
    return {
        trip_type: trip ? trip.value : 'oneway',
        model: modelRadio ? modelRadio.value : 'ann',
        origin: document.getElementById('origin').value,
        destination: document.getElementById('destination').value,
        flight_date: fd,
        return_date: document.getElementById('return_date') ? document.getElementById('return_date').value : null,
        airline: document.getElementById('airline').value,
        class: document.getElementById('seat_class').value,
        day: dt.getDate(),
        month: dt.getMonth() + 1,
        year: dt.getFullYear(),
        departure_hour: parseInt(document.getElementById('departure_hour').value) || 12,
        arrival_hour: parseInt(document.getElementById('arrival_hour').value) || 14,
        duration: dur,
        stops: parseInt(document.getElementById('stops').value) || 0
    };
}

function displayResults(data, formData) {
    var empty = document.getElementById('emptyState');
    var content = document.getElementById('resultContent');
    if (empty) empty.style.display = 'none';
    if (content) content.style.display = 'block';
    
    document.getElementById('originCode').textContent = formData.origin;
    document.getElementById('destinationCode').textContent = formData.destination;
    document.getElementById('departureTime').textContent = String(formData.departure_hour).padStart(2,'0') + ':00';
    document.getElementById('arrivalTime').textContent = String(formData.arrival_hour).padStart(2,'0') + ':00';
    
    var h = Math.floor(formData.duration / 60), m = formData.duration % 60;
    document.getElementById('flightDuration').textContent = h + 'h ' + m + 'm';
    document.getElementById('stopsInfo').textContent = formData.stops === 0 ? 'Bay thẳng' : formData.stops + ' điểm dừng';
    document.getElementById('resultAirline').textContent = formData.airline;
    document.getElementById('resultClass').textContent = formData.class === 'Economy' ? 'Phổ thông' : 'Thương gia';
    
    var dt = new Date(formData.flight_date);
    var days = ['CN','T2','T3','T4','T5','T6','T7'];
    var str = days[dt.getDay()] + ', ' + dt.getDate() + '/' + (dt.getMonth()+1);
    if (data.is_round_trip) str += ' (Khứ hồi)';
    document.getElementById('resultDate').textContent = str;
    
    // Update model info
    var modelUsed = data.model_used || 'ann';
    var modelName = data.model_name || 'Neural Network';
    var modelAccuracy = data.model_accuracy || 0;
    
    document.getElementById('modelUsedText').textContent = 'Dự đoán bởi ' + modelName;
    
    var modelBadge = document.getElementById('modelBadgeResult');
    if (modelBadge) {
        var icon = modelIcons[modelUsed] || 'fa-brain';
        var color = modelColors[modelUsed] || '#3b82f6';
        modelBadge.innerHTML = '<i class="fas ' + icon + '"></i><span>' + modelName.split('(')[0].trim() + '</span>';
        modelBadge.style.background = color;
    }
    
    document.getElementById('resultModelAccuracy').textContent = modelAccuracy.toFixed(1) + '%';
    
    animatePrice(document.getElementById('predictedPrice'), 0, data.price);
    
    if (data.amenities) {
        var w = document.getElementById('wifiValue');
        var me = document.getElementById('mealsValue');
        var b = document.getElementById('baggageValue');
        if (w) w.innerHTML = data.amenities.wifi === 'Yes' ? '<i class="fas fa-check" style="color:#10b981"></i> Có' : '<i class="fas fa-times" style="color:#94a3b8"></i> Không';
        if (me) me.innerHTML = data.amenities.meals === 'Yes' ? '<i class="fas fa-check" style="color:#10b981"></i> Có' : '<i class="fas fa-times" style="color:#94a3b8"></i> Không';
        if (b) b.textContent = data.amenities.baggage_kg + ' kg';
    }
}

function animatePrice(el, start, end) {
    var duration = 1500, startTime = performance.now();
    function update(t) {
        var elapsed = t - startTime;
        var progress = Math.min(elapsed / duration, 1);
        var ease = 1 - Math.pow(1 - progress, 3);
        el.textContent = formatPrice(Math.floor(start + (end - start) * ease));
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

function formatPrice(p) { return p.toString().replace(/\B(?=(\d{3})+(?!\d))/g, '.'); }

function searchAgain() {
    var empty = document.getElementById('emptyState');
    var content = document.getElementById('resultContent');
    if (empty) empty.style.display = 'flex';
    if (content) content.style.display = 'none';
    goToSlide(2);
}

function showToast(msg, type) {
    type = type || 'info';
    var container = document.getElementById('toastContainer');
    if (!container) return;
    var toast = document.createElement('div');
    toast.className = 'toast ' + type;
    var icons = { success: 'check-circle', error: 'exclamation-circle', warning: 'exclamation-triangle', info: 'info-circle' };
    toast.innerHTML = '<i class="fas fa-' + (icons[type] || 'info-circle') + '"></i><span>' + msg + '</span>';
    container.appendChild(toast);
    setTimeout(function() { toast.classList.add('removing'); setTimeout(function() { toast.remove(); }, 300); }, 4000);
}

window.goToSlide = goToSlide;
window.searchAgain = searchAgain;
window.swapLocations = swapLocations;
window.showToast = showToast;