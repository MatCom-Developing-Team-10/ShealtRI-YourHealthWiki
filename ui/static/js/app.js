/**
 * ShealtRI UI — JavaScript application logic
 * - Avatar state machine (greetings → idle → responding)
 * - Search query execution
 * - Results rendering
 * - Profile selection
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

// Videos to randomly select from for "responding" state
const respondingVariants = [
    '/static/videos/responding.mp4',
    '/static/videos/responding_2.mp4',
    '/static/videos/responding_hard.mp4',
];

let currentAvatarState = null;

/**
 * Switch avatar to a specific state.
 * @param {string} state - State name: 'idle', 'responding', or 'greetings'
 */
function setAvatarState(state) {
    if (currentAvatarState === state) {
        return; // Already in this state
    }

    // Hide all videos
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

    // Set loop and autoplay based on state
    if (state === 'greetings') {
        videoEl.loop = false;
        videoEl.onended = () => {
            setAvatarState(AvatarStates.IDLE);
        };
    } else if (state === 'idle') {
        videoEl.loop = true;
    } else if (state === 'responding') {
        videoEl.loop = true;
    }

    videoEl.play().catch(err => {
        console.error(`Failed to play ${state} video:`, err);
    });

    currentAvatarState = state;
}

/**
 * Get the "hard" responding variant (always use during search phase).
 */
function getRespondingHardVariant() {
    return '/static/videos/responding_hard.mp4';
}

/**
 * Get a random responding variant (not hard) for the response phase.
 */
function getRandomRespondingVariant() {
    // Return either responding.mp4 or responding_2.mp4 (not hard)
    return [
        '/static/videos/responding.mp4',
        '/static/videos/responding_2.mp4',
    ][Math.floor(Math.random() * 2)];
}

// ─────────────────────────────────────────────────────────────────────────────
// Profile Management
// ─────────────────────────────────────────────────────────────────────────────

let currentProfile = 'paciente';

// Profile metadata: icons (SVG)
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
                // Update UI to show selection
                document.querySelectorAll('.profile-option').forEach(el => {
                    el.classList.remove('selected');
                });
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
        container.innerHTML = `
            <div class="results-placeholder">
                <p>No se encontraron resultados para esta consulta</p>
            </div>
        `;
        return;
    }

    container.innerHTML = '';

    results.forEach(result => {
        const card = document.createElement('div');
        card.className = 'result-card';

        const relevancePercent = (result.relevance * 100).toFixed(0);

        card.innerHTML = `
            <div class="result-title">
                <span>${escapeHtml(result.title)}</span>
                <span class="result-relevance">${relevancePercent}%</span>
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
        container.textContent = text;
    }
}

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;',
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Query Execution
// ─────────────────────────────────────────────────────────────────────────────

async function executeQuery(queryText) {
    if (!queryText.trim()) {
        return;
    }

    const searchBtn = document.getElementById('search-btn');
    searchBtn.classList.add('loading');
    searchBtn.disabled = true;
    searchBtn.textContent = 'Buscando...';

    // Phase 1: Switch to responding_hard during search
    const respondingHardUrl = getRespondingHardVariant();
    const respondingSource = avatarVideos.responding.querySelector('source');
    if (respondingSource) {
        respondingSource.src = respondingHardUrl;
        avatarVideos.responding.load();
        avatarVideos.responding.play().catch(err => {
            console.error('Failed to play responding_hard:', err);
        });
    }
    setAvatarState(AvatarStates.RESPONDING);

    // Show loading state
    const resultsContainer = document.getElementById('results-container');
    resultsContainer.innerHTML = `
        <div class="results-placeholder">
            <p>Buscando información...</p>
        </div>
    `;
    renderRagResponse('Generando respuesta...');

    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: queryText,
                profile: currentProfile,
                top_k: 5,
            }),
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }

        const data = await response.json();

        // Render results
        renderResults(data.results);
        renderRagResponse(data.rag_response);

        // Phase 2: Switch to responding.mp4 after response is ready
        const respondingUrl = '/static/videos/responding.mp4';
        if (respondingSource) {
            respondingSource.src = respondingUrl;
            avatarVideos.responding.load();
            avatarVideos.responding.play().catch(err => {
                console.error('Failed to play responding:', err);
            });
        }
        // Avatar stays in responding state for 5 seconds

        // Phase 3: After 5 seconds, switch back to idle
        setTimeout(() => {
            setAvatarState(AvatarStates.IDLE);
        }, 5000);
    } catch (err) {
        console.error('Query execution failed:', err);
        const resultsContainer = document.getElementById('results-container');
        resultsContainer.innerHTML = `
            <div class="results-placeholder">
                <p>Error al ejecutar la búsqueda: ${escapeHtml(err.message)}</p>
            </div>
        `;
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
    // Load profiles
    await loadProfiles();

    // Set up search input
    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');

    searchBtn.addEventListener('click', () => {
        executeQuery(searchInput.value);
    });

    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            executeQuery(searchInput.value);
        }
    });

    // Start with greetings animation
    setAvatarState(AvatarStates.GREETINGS);
});
