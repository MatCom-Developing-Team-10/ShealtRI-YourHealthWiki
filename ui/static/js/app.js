/**
 * ShealtRI UI — JavaScript application logic
 * - Avatar state machine (greetings → idle → responding)
 * - Search query execution
 * - Results rendering with source badges
 * - RAG markdown rendering via marked.js
 * - Spell correction hint
 * - Query time display
 * - Web fallback indicator
 * - Evaluation metrics section (Biblioteca nav)
 * - Query history via localStorage (Historial nav)
 */

// ─────────────────────────────────────────────────────────────────────────────
// Avatar State Machine
// ─────────────────────────────────────────────────────────────────────────────

const AvatarStates = {
    IDLE: 'idle',
    RESPONDING: 'responding',
    GREETINGS: 'greetings',
};

const avatarVideos = {
    idle: document.getElementById('avatar-idle'),
    greetings: document.getElementById('avatar-greetings'),
    responding: document.getElementById('avatar-responding'),
};

let currentAvatarState = null;

function setAvatarState(state) {
    if (currentAvatarState === state) {
        return;
    }

    Object.values(avatarVideos).forEach(video => {
        if (video) {
            video.pause();
            video.style.display = 'none';
        }
    });

    const videoEl = avatarVideos[state];
    if (!videoEl) {
        console.error(`Unknown avatar state: ${state}`);
        return;
    }

    videoEl.style.display = 'block';

    if (state === 'greetings') {
        videoEl.loop = false;
        videoEl.onended = () => setAvatarState(AvatarStates.IDLE);
    } else if (state === 'idle') {
        videoEl.loop = true;
    } else if (state === 'responding') {
        videoEl.loop = true;
    }

    videoEl.play().catch(err => console.error(`Failed to play ${state} video:`, err));
    currentAvatarState = state;
}

function getRespondingHardVariant() {
    return '/static/videos/responding_hard.mp4';
}

function getRandomRespondingVariant() {
    return [
        '/static/videos/responding.mp4',
        '/static/videos/responding_2.mp4',
    ][Math.floor(Math.random() * 2)];
}

// ─────────────────────────────────────────────────────────────────────────────
// Profile Management
// ─────────────────────────────────────────────────────────────────────────────

let currentProfile = 'paciente';

const profileMetadata = {
    paciente: {
        icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
    },
    estudiante: {
        icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>',
    },
    medico: {
        icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v12M6 12h12"/></svg>',
    },
    diagnostico: {
        icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path d="M12 6v6l4 2"/></svg>',
    },
    natural: {
        icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path d="M12 6v12M6 12h12"/></svg>',
    },
    cuidador: {
        icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
    },
};

async function loadProfiles() {
    try {
        const response = await fetch('/api/profiles');
        const profiles = await response.json();

        const container = document.getElementById('profiles-container');
        container.innerHTML = '';

        profiles.forEach(profile => {
            const label = document.createElement('label');
            label.className = 'profile-option';
            const metadata = profileMetadata[profile.slug] || { icon: '○' };

            if (profile.slug === currentProfile) {
                label.classList.add('selected');
            }

            const radio = document.createElement('input');
            radio.type = 'radio';
            radio.name = 'user-profile';
            radio.value = profile.slug;
            radio.checked = profile.slug === currentProfile;

            radio.addEventListener('change', () => {
                currentProfile = profile.slug;
                document.querySelectorAll('.profile-option').forEach(el => el.classList.remove('selected'));
                label.classList.add('selected');
            });

            const iconSpan = document.createElement('span');
            iconSpan.className = 'profile-icon';
            iconSpan.innerHTML = metadata.icon;

            label.appendChild(radio);
            label.appendChild(iconSpan);
            label.appendChild(document.createTextNode(profile.label));
            container.appendChild(label);
        });
    } catch (err) {
        console.error('Failed to load profiles:', err);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Results Rendering
// ─────────────────────────────────────────────────────────────────────────────

function renderResults(results) {
    const container = document.getElementById('results-container');

    if (!results || results.length === 0) {
        container.innerHTML = '<div class="results-placeholder"><p>No se encontraron resultados para esta consulta</p></div>';
        return;
    }

    container.innerHTML = '';

    results.forEach(result => {
        const card = document.createElement('div');
        card.className = 'result-card';

        const relevancePercent = (result.relevance * 100).toFixed(0);
        const sourceType = result.source_type || 'local';
        const sourceLabel = sourceType === 'web' ? 'Web' : 'Local';

        card.innerHTML = `
            <div class="result-title">
                <span>${escapeHtml(result.title)}</span>
                <div class="result-badges">
                    <span class="source-badge source-${escapeHtml(sourceType)}">${sourceLabel}</span>
                    <span class="result-relevance">${relevancePercent}%</span>
                </div>
            </div>
            <div class="result-source">${escapeHtml(result.source)}</div>
            <div class="result-snippet">${escapeHtml(result.snippet)}</div>
        `;

        container.appendChild(card);
    });
}

function renderRagResponse(text) {
    const container = document.getElementById('rag-response');
    if (!text || text.trim() === '') {
        container.innerHTML = '<p class="response-placeholder">La respuesta generada aparecerá aquí</p>';
    } else {
        if (typeof marked !== 'undefined' && marked.parse) {
            container.innerHTML = marked.parse(text);
        } else {
            container.innerHTML = escapeHtml(text).replace(/\n/g, '<br>');
        }
    }
}

function escapeHtml(text) {
    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
    return String(text).replace(/[&<>"']/g, m => map[m]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Query History (localStorage)
// ─────────────────────────────────────────────────────────────────────────────

const HISTORY_KEY = 'shealtri_history';
const HISTORY_MAX = 10;

function saveToHistory(queryText) {
    const raw = localStorage.getItem(HISTORY_KEY);
    let history = [];
    try { history = raw ? JSON.parse(raw) : []; } catch { history = []; }

    // Remove duplicate if exists, then prepend
    history = history.filter(h => h.text !== queryText);
    history.unshift({ text: queryText, timestamp: Date.now() });
    if (history.length > HISTORY_MAX) history = history.slice(0, HISTORY_MAX);

    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
}

function relativeTime(ts) {
    const diff = Math.floor((Date.now() - ts) / 1000);
    if (diff < 60) return 'hace un momento';
    if (diff < 3600) return `hace ${Math.floor(diff / 60)} min`;
    if (diff < 86400) return `hace ${Math.floor(diff / 3600)} h`;
    return `hace ${Math.floor(diff / 86400)} d`;
}

function renderHistorial() {
    const panel = document.getElementById('historial-panel');
    const list = document.getElementById('historial-list');
    const raw = localStorage.getItem(HISTORY_KEY);
    let history = [];
    try { history = raw ? JSON.parse(raw) : []; } catch { history = []; }

    list.innerHTML = '';

    if (history.length === 0) {
        list.innerHTML = '<p class="historial-empty">No hay búsquedas recientes</p>';
    } else {
        history.forEach(item => {
            const btn = document.createElement('button');
            btn.className = 'historial-item';
            btn.innerHTML = `
                <span class="historial-text">${escapeHtml(item.text)}</span>
                <span class="historial-time">${relativeTime(item.timestamp)}</span>
            `;
            btn.addEventListener('click', () => {
                document.getElementById('search-input').value = item.text;
                panel.hidden = true;
                executeQuery(item.text);
            });
            list.appendChild(btn);
        });
    }

    panel.hidden = false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Evaluation Metrics (Biblioteca nav)
// ─────────────────────────────────────────────────────────────────────────────

let _lastResults = null;

async function showEvaluation() {
    const container = document.getElementById('results-container');
    const meta = document.getElementById('results-meta');
    meta.hidden = true;
    document.getElementById('spell-hint').hidden = true;
    document.getElementById('web-fallback-banner').hidden = true;

    container.innerHTML = `
        <div class="results-placeholder">
            <p>Calculando métricas de evaluación… (puede tardar unos segundos)</p>
        </div>
    `;

    try {
        const response = await fetch('/api/eval?k=10');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        const metrics = data.aggregated;
        const rows = Object.entries(metrics)
            .map(([name, val]) => `<tr><td>${escapeHtml(name)}</td><td>${(val * 100).toFixed(2)}%</td></tr>`)
            .join('');

        container.innerHTML = `
            <div class="eval-section">
                <div class="eval-header">
                    <h3>Evaluación del sistema (k=${data.k}, ${data.num_queries} consultas)</h3>
                    <button id="eval-back-btn" class="eval-back-btn">← Volver</button>
                </div>
                <table class="metrics-table">
                    <thead><tr><th>Métrica</th><th>Valor</th></tr></thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        `;

        document.getElementById('eval-back-btn').addEventListener('click', () => {
            if (_lastResults) {
                renderResults(_lastResults);
            } else {
                container.innerHTML = '<div class="results-placeholder"><p>Ingresa una consulta para comenzar</p></div>';
            }
        });
    } catch (err) {
        container.innerHTML = `<div class="results-placeholder"><p>Error al cargar métricas: ${escapeHtml(err.message)}</p></div>`;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Query Execution
// ─────────────────────────────────────────────────────────────────────────────

async function executeQuery(queryText) {
    if (!queryText.trim()) return;

    const searchBtn = document.getElementById('search-btn');
    const resultsContainer = document.getElementById('results-container');
    const spellHint = document.getElementById('spell-hint');
    const resultsMeta = document.getElementById('results-meta');
    const webBanner = document.getElementById('web-fallback-banner');
    const historialPanel = document.getElementById('historial-panel');

    // Hide auxiliary elements at the start
    spellHint.hidden = true;
    resultsMeta.hidden = true;
    webBanner.hidden = true;
    historialPanel.hidden = true;

    searchBtn.classList.add('loading');
    searchBtn.disabled = true;
    searchBtn.textContent = 'Buscando...';

    // Switch avatar to responding_hard during search
    const respondingSource = avatarVideos.responding.querySelector('source');
    if (respondingSource) {
        respondingSource.src = getRespondingHardVariant();
        avatarVideos.responding.load();
        avatarVideos.responding.play().catch(() => {});
    }
    setAvatarState(AvatarStates.RESPONDING);

    resultsContainer.innerHTML = '<div class="results-placeholder"><p>Buscando información...</p></div>';
    renderRagResponse('Generando respuesta...');

    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: queryText,
                profile: currentProfile,
                top_k: 5,
                force_web: document.getElementById('force-web-toggle')?.checked ?? false,
            }),
        });

        if (!response.ok) throw new Error(`API error: ${response.statusText}`);

        const data = await response.json();

        // Render results
        _lastResults = data.results;
        renderResults(data.results);
        renderRagResponse(data.rag_response);

        // Query time
        if (data.query_time_ms != null) {
            resultsMeta.textContent = `${data.results.length} resultado${data.results.length !== 1 ? 's' : ''} en ${data.query_time_ms} ms`;
            resultsMeta.hidden = false;
        }

        // Spell correction hint
        if (data.corrected_query && data.corrected_query.trim() !== queryText.trim().toLowerCase()) {
            spellHint.innerHTML = `¿Quisiste decir: <button class="spell-suggestion" data-query="${escapeHtml(data.corrected_query)}">${escapeHtml(data.corrected_query)}</button>?`;
            spellHint.hidden = false;
            spellHint.querySelector('.spell-suggestion').addEventListener('click', e => {
                const q = e.currentTarget.dataset.query;
                document.getElementById('search-input').value = q;
                executeQuery(q);
            });
        }

        // Web fallback banner
        if (data.used_web_fallback) {
            webBanner.hidden = false;
        }

        saveToHistory(queryText);

        // Switch to calmer responding video, then idle after 5 s
        if (respondingSource) {
            respondingSource.src = getRandomRespondingVariant();
            avatarVideos.responding.load();
            avatarVideos.responding.play().catch(() => {});
        }
        setTimeout(() => setAvatarState(AvatarStates.IDLE), 5000);

    } catch (err) {
        console.error('Query execution failed:', err);
        resultsContainer.innerHTML = `<div class="results-placeholder"><p>Error al ejecutar la búsqueda: ${escapeHtml(err.message)}</p></div>`;
        renderRagResponse('');
        setAvatarState(AvatarStates.IDLE);
    } finally {
        searchBtn.classList.remove('loading');
        searchBtn.disabled = false;
        searchBtn.textContent = 'Buscar';
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Event Listeners
// ─────────────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
    await loadProfiles();

    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');

    searchBtn.addEventListener('click', () => executeQuery(searchInput.value));
    searchInput.addEventListener('keypress', e => {
        if (e.key === 'Enter') executeQuery(searchInput.value);
    });

    document.getElementById('nav-historial').addEventListener('click', () => {
        const panel = document.getElementById('historial-panel');
        if (panel.hidden) {
            renderHistorial();
        } else {
            panel.hidden = true;
        }
    });

    document.getElementById('historial-close').addEventListener('click', () => {
        document.getElementById('historial-panel').hidden = true;
    });

    document.getElementById('nav-biblioteca').addEventListener('click', () => {
        showEvaluation();
    });

    setAvatarState(AvatarStates.GREETINGS);
});
