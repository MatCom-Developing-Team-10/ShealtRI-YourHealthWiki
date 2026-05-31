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

let _lastQueryText = '';

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

        // Title row (innerHTML is safe here — values are escaped)
        card.innerHTML = `
            <div class="result-title">
                <span>${escapeHtml(result.title)}</span>
                <div class="result-badges">
                    <span class="source-badge source-${escapeHtml(sourceType)}">${sourceLabel}</span>
                    <span class="result-relevance">${relevancePercent}%</span>
                </div>
            </div>
        `;

        // Source — DOM element so Firefox treats it as a real anchor
        const src = (result.source || '').trim();
        const isWebUrl = /^https?:\/\//i.test(src);
        const isLocalPath = src && !isWebUrl && src !== '(sin URL)';
        const hasLink = isWebUrl || isLocalPath;
        const sourceEl = document.createElement(hasLink ? 'a' : 'div');
        sourceEl.className = 'result-source';
        sourceEl.textContent = src || '(sin URL)';
        if (isWebUrl) {
            sourceEl.href = src;
            sourceEl.target = '_blank';
            sourceEl.rel = 'noopener noreferrer';
        } else if (isLocalPath) {
            sourceEl.href = `/api/document?path=${encodeURIComponent(src)}`;
            sourceEl.target = '_blank';
            sourceEl.rel = 'noopener noreferrer';
        }
        card.appendChild(sourceEl);

        // Snippet
        const snippetEl = document.createElement('div');
        snippetEl.className = 'result-snippet';
        snippetEl.textContent = result.snippet;
        card.appendChild(snippetEl);

        // Relevance feedback (Rocchio) — only for indexed docs; web fallback
        // results aren't in the latent space so feedback would be a no-op.
        if (result.doc_id && sourceType !== 'web') {
            card.appendChild(buildFeedbackBar(result.doc_id));
        }

        container.appendChild(card);
    });
}

function buildFeedbackBar(docId) {
    const bar = document.createElement('div');
    bar.className = 'feedback-bar';
    bar.innerHTML = `
        <span class="feedback-label">¿Te resultó útil?</span>
        <button class="feedback-btn feedback-up" data-doc="${escapeHtml(docId)}" data-relevant="true" aria-label="Marcar como relevante">👍</button>
        <button class="feedback-btn feedback-down" data-doc="${escapeHtml(docId)}" data-relevant="false" aria-label="Marcar como no relevante">👎</button>
        <span class="judgment-count" hidden></span>
    `;
    bar.querySelectorAll('.feedback-btn').forEach(btn => {
        btn.addEventListener('click', () => sendFeedback(btn));
    });
    return bar;
}

async function sendFeedback(btn) {
    if (!_lastQueryText) return;
    const docId = btn.dataset.doc;
    const relevant = btn.dataset.relevant === 'true';

    // Visual feedback first (optimistic) — the call is fire-and-forget for UX.
    const siblings = btn.parentElement.querySelectorAll('.feedback-btn');
    siblings.forEach(s => s.classList.remove('feedback-selected'));
    btn.classList.add('feedback-selected');

    try {
        const response = await fetch('/api/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: _lastQueryText, doc_id: docId, relevant }),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        // Echo back how many judgments this query has accumulated so the user
        // can see Rocchio is building up evidence.
        const total = data.total_judgments_for_query;
        if (typeof total === 'number') {
            const badge = btn.parentElement.querySelector('.judgment-count');
            if (badge) {
                badge.textContent = `Rocchio · ${total} juicio${total !== 1 ? 's' : ''}`;
                badge.hidden = false;
            }
            // Cheap refresh of the global counter in the footer.
            loadStats();
        }
    } catch (err) {
        console.error('Failed to record feedback:', err);
        btn.classList.remove('feedback-selected');
    }
}

async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const s = await response.json();
        const footer = document.getElementById('stats-footer');
        footer.innerHTML = `
            <span>📚 ${s.n_documents.toLocaleString()} documentos · ${s.n_chunks.toLocaleString()} chunks</span>
            <span>🔤 vocab ${s.n_terms.toLocaleString()}</span>
            <span>📐 LSI k=${s.lsi_k}</span>
            <span>📝 ${s.avg_tokens_per_doc.toFixed(0)} tokens/doc</span>
            <span>👍 ${s.feedback_judgments} juicios Rocchio</span>
        `;
        footer.hidden = false;
    } catch (err) {
        console.error('Failed to load stats:', err);
    }
}

async function fetchRecommendations(seedDocIds, profile, excludeIds = null) {
    const panel = document.getElementById('recommendations-panel');
    const list = document.getElementById('recommendations-list');

    if (!seedDocIds || seedDocIds.length === 0) {
        panel.hidden = true;
        return;
    }

    try {
        const response = await fetch('/api/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                seed_doc_ids: seedDocIds,
                top_k: 3,
                profile: profile,
                exclude_ids: excludeIds || seedDocIds,
            }),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        if (!data.items || data.items.length === 0) {
            panel.hidden = true;
            return;
        }

        list.innerHTML = data.items.map(item => `
            <div class="recommendation-card" data-doc="${escapeHtml(item.doc_id)}" data-url="${escapeHtml(item.source)}">
                <div class="recommendation-title">${escapeHtml(item.title)}</div>
                <div class="recommendation-snippet">${escapeHtml(item.snippet)}</div>
                <div class="recommendation-footer">
                    <span>${escapeHtml(extractSourceLabel(item.source))}</span>
                    <span class="recommendation-similarity">${(item.similarity * 100).toFixed(0)}% similitud</span>
                </div>
            </div>
        `).join('');

        // Clicking a recommendation opens its URL in a new tab (web) or
        // serves the local file (local). Reuses the same routing as the
        // result cards above.
        list.querySelectorAll('.recommendation-card').forEach(card => {
            card.addEventListener('click', () => {
                const url = card.dataset.url;
                if (url && url !== '(sin URL)') {
                    if (url.startsWith('http')) {
                        window.open(url, '_blank', 'noopener');
                    } else {
                        window.open(`/api/document?path=${encodeURIComponent(url)}`, '_blank', 'noopener');
                    }
                }
            });
        });

        panel.hidden = false;
    } catch (err) {
        console.error('Failed to fetch recommendations:', err);
        panel.hidden = true;
    }
}

function extractSourceLabel(url) {
    if (!url || url === '(sin URL)') return 'local';
    try {
        const host = new URL(url).hostname.replace(/^www\./, '');
        return host;
    } catch {
        return 'local';
    }
}

/**
 * Render the bidirectional spell-correction hint.
 *
 * @param {HTMLElement} hint       The #spell-hint container.
 * @param {string} typedQuery      Exactly what the user typed.
 * @param {string|null} correction The available correction, or null/empty.
 * @param {boolean} applied        Whether retrieval used the correction.
 *
 * Two states, mirroring Google:
 *   - applied  → "Mostrando resultados para «X». Buscar en su lugar «typed»"
 *                clicking re-runs verbatim (apply_correction = false).
 *   - !applied → "¿Quisiste decir «X»?"
 *                clicking re-runs the same query corrected (apply_correction = true).
 */
function renderSpellHint(hint, typedQuery, correction, applied) {
    hint.hidden = true;
    hint.innerHTML = '';

    const typed = (typedQuery || '').trim();
    const corrected = (correction || '').trim();
    if (!corrected || corrected === typed.toLowerCase()) {
        return;
    }

    if (applied) {
        hint.innerHTML = `Mostrando resultados para <strong>${escapeHtml(corrected)}</strong>. `
            + `Buscar en su lugar: `
            + `<button class="spell-suggestion" data-query="${escapeHtml(typed)}" data-correct="false">${escapeHtml(typed)}</button>`;
    } else {
        hint.innerHTML = `¿Quisiste decir: `
            + `<button class="spell-suggestion" data-query="${escapeHtml(typed)}" data-correct="true">${escapeHtml(corrected)}</button>?`;
    }
    hint.hidden = false;

    hint.querySelector('.spell-suggestion').addEventListener('click', e => {
        const q = e.currentTarget.dataset.query;
        const correct = e.currentTarget.dataset.correct === 'true';
        document.getElementById('search-input').value = q;
        executeQuery(q, { applyCorrection: correct });
    });
}

function renderExpansion(terms) {
    const hint = document.getElementById('expansion-hint');
    if (!terms || terms.length === 0) {
        hint.hidden = true;
        return;
    }
    const chips = terms
        .map(t => `<span class="expansion-chip">${escapeHtml(t)}</span>`)
        .join('');
    hint.innerHTML = `<span class="expansion-prefix">También buscamos:</span> ${chips}`;
    hint.hidden = false;
}

function renderRagMeta(usedLlm, modelName) {
    const meta = document.getElementById('rag-meta');
    if (!modelName) {
        meta.hidden = true;
        return;
    }
    const backend = usedLlm ? modelName : 'plantilla (sin LLM)';
    meta.innerHTML = `<span class="rag-meta-pill">Backend: ${escapeHtml(backend)}</span>`;
    meta.hidden = false;
}

function renderRagQuality(quality) {
    const container = document.getElementById('rag-quality');
    if (!quality) {
        container.hidden = true;
        return;
    }
    const labels = {
        faithfulness: 'Fidelidad',
        groundedness: 'Anclaje',
        context_relevance: 'Relevancia del contexto',
    };
    const chips = Object.entries(quality)
        .map(([k, v]) => `<span class="quality-chip" title="${escapeHtml(k)}">${escapeHtml(labels[k] || k)}: ${(v * 100).toFixed(0)}%</span>`)
        .join('');
    container.innerHTML = `<div class="quality-label">Calidad RAG</div><div class="quality-chips">${chips}</div>`;
    container.hidden = false;
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
    document.getElementById('expansion-hint').hidden = true;
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
        const aggregatedRows = Object.entries(metrics)
            .map(([name, val]) => `<tr><td>${escapeHtml(name)}</td><td>${(val * 100).toFixed(2)}%</td></tr>`)
            .join('');

        // Build per-query table from the metric keys of the first row, so
        // Fallout@k (which only appears when corpus_size is known) shows up
        // automatically without a hard-coded column list.
        let perQueryHtml = '';
        if (data.per_query && data.per_query.length > 0) {
            const metricNames = Object.keys(data.per_query[0].metrics);
            const headerCells = metricNames
                .map(name => `<th class="num">${escapeHtml(name)}</th>`)
                .join('');
            const bodyRows = data.per_query.map(q => {
                const numericCells = metricNames
                    .map(name => `<td class="num">${(q.metrics[name] * 100).toFixed(1)}%</td>`)
                    .join('');
                return `<tr>
                    <td>${escapeHtml(q.query_id)}</td>
                    <td>${escapeHtml(q.text)}</td>
                    <td class="num">${q.num_relevant}</td>
                    ${numericCells}
                </tr>`;
            }).join('');
            perQueryHtml = `
                <h4 style="margin-top:18px;">Desglose por consulta</h4>
                <table class="per-query-table">
                    <thead><tr>
                        <th>ID</th><th>Consulta</th><th class="num">|Rel|</th>${headerCells}
                    </tr></thead>
                    <tbody>${bodyRows}</tbody>
                </table>
            `;
        }

        container.innerHTML = `
            <div class="eval-section">
                <div class="eval-header">
                    <h3>Evaluación del sistema (k=${data.k}, ${data.num_queries} consultas, |corpus|=${data.corpus_size})</h3>
                    <button id="eval-back-btn" class="eval-back-btn">← Volver</button>
                </div>
                <table class="metrics-table">
                    <thead><tr><th>Métrica</th><th>Valor</th></tr></thead>
                    <tbody>${aggregatedRows}</tbody>
                </table>
                ${perQueryHtml}
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

async function executeQuery(queryText, options = {}) {
    if (!queryText.trim()) return;

    // Spell-correction choice: an explicit override (from the "did you mean" /
    // "search verbatim" links) wins; otherwise fall back to the toggle state.
    const spellToggle = document.getElementById('spell-correct-toggle');
    const applyCorrection = options.applyCorrection !== undefined
        ? options.applyCorrection
        : (spellToggle ? spellToggle.checked : true);
    // Keep the toggle in sync when an explicit choice was made, so the UI
    // honestly reflects what the next search will do.
    if (options.applyCorrection !== undefined && spellToggle) {
        spellToggle.checked = applyCorrection;
    }

    const searchBtn = document.getElementById('search-btn');
    const resultsContainer = document.getElementById('results-container');
    const spellHint = document.getElementById('spell-hint');
    const expansionHint = document.getElementById('expansion-hint');
    const resultsMeta = document.getElementById('results-meta');
    const webBanner = document.getElementById('web-fallback-banner');
    const ragMeta = document.getElementById('rag-meta');
    const ragQuality = document.getElementById('rag-quality');
    const historialPanel = document.getElementById('historial-panel');
    const recommendationsPanel = document.getElementById('recommendations-panel');

    // Hide auxiliary elements at the start
    spellHint.hidden = true;
    expansionHint.hidden = true;
    resultsMeta.hidden = true;
    webBanner.hidden = true;
    ragMeta.hidden = true;
    ragQuality.hidden = true;
    historialPanel.hidden = true;
    recommendationsPanel.hidden = true;
    _lastQueryText = queryText;

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
                apply_correction: applyCorrection,
            }),
        });

        if (!response.ok) throw new Error(`API error: ${response.statusText}`);

        const data = await response.json();

        // Render results
        _lastResults = data.results;
        renderResults(data.results);
        renderRagMeta(data.rag_used_llm, data.rag_model_name);
        renderRagResponse(data.rag_response);
        renderRagQuality(data.rag_quality);
        renderExpansion(data.expansion_terms);

        // Fire the recommender in parallel (it's not on the critical path —
        // results are already rendered). Uses the top-3 result doc_ids as
        // the seed and excludes everything already shown so the panel only
        // surfaces *new* documents.
        const seeds = data.results.slice(0, 3).map(r => r.doc_id).filter(Boolean);
        const excludes = data.results.map(r => r.doc_id).filter(Boolean);
        fetchRecommendations(seeds, currentProfile, excludes);

        // Query time
        if (data.query_time_ms != null) {
            resultsMeta.textContent = `${data.results.length} resultado${data.results.length !== 1 ? 's' : ''} en ${data.query_time_ms} ms`;
            resultsMeta.hidden = false;
        }

        // Spell correction hint — bidirectional (Google-style). The backend
        // reports both the available correction and whether it was applied,
        // so the user can always switch to the other interpretation.
        renderSpellHint(spellHint, queryText, data.corrected_query, data.correction_applied);

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
    loadStats();

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
