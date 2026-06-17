/* ============================================================
   AI Research Assistant — Client-Side Application Logic
   ============================================================ */

(function () {
  'use strict';

  // ── State ──────────────────────────────────────────────────
  const state = {
    mode: 'moderate',           // 'quick' | 'moderate' | 'deep'
    provider: 'deepseek',
    currentSearchId: null,
    isStreaming: false,
    eventSource: null,
    contentBuffer: '',
    thinkingBuffer: '',
    renderTimer: null,
    activeHistoryId: null,
    sources: [],                // current sources for citation tooltips
    images: [],
    includeImages: true,
    cancelRequested: false,
  };

  const ICON_SEARCH = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
      <line x1="5" y1="12" x2="19" y2="12"></line>
      <polyline points="12 5 19 12 12 19"></polyline>
    </svg>
  `;
  const ICON_CANCEL = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
      <line x1="6" y1="6" x2="18" y2="18"></line>
      <line x1="18" y1="6" x2="6" y2="18"></line>
    </svg>
  `;

  // ── DOM References ─────────────────────────────────────────
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const dom = {
    searchInput:      $('#search-input'),
    btnSearch:        $('#btn-search'),
    btnNewSearch:     $('#btn-new-search'),
    modelSelector:    $('#model-selector'),
    imagesToggle:     $('#images-toggle'),
    modeSlider:       $('#mode-slider'),
    modeQuick:        $('#mode-quick'),
    modeModerate:     $('#mode-moderate'),
    modeDeep:         $('#mode-deep'),
    resultsArea:      $('#results-area'),
    emptyState:       $('#empty-state'),
    pipelinePanel:    $('#pipeline-panel'),
    pipelineSteps:    $('#pipeline-steps'),
    thinkingPanel:    $('#thinking-panel'),
    thinkingHeader:   $('#thinking-header'),
    thinkingChevron:  $('#thinking-chevron'),
    thinkingBody:     $('#thinking-body'),
    thinkingContent:  $('#thinking-content'),
    thinkingTokenCt:  $('#thinking-token-count'),
    responsePanel:    $('#response-panel'),
    responseContent:  $('#response-content'),
    imagesPanel:      $('#images-panel'),
    imagesGrid:       $('#images-grid'),
    sourcesPanel:     $('#sources-panel'),
    sourcesList:      $('#sources-list'),
    historyList:      $('#history-list'),
    toastContainer:   $('#toast-container'),
  };

  // ── Configure marked.js ────────────────────────────────────
  marked.setOptions({
    highlight: function (code, lang) {
      if (lang && hljs.getLanguage(lang)) {
        try { return hljs.highlight(code, { language: lang }).value; } catch (_) {}
      }
      try { return hljs.highlightAuto(code).value; } catch (_) {}
      return code;
    },
    breaks: false,
    gfm: true,
  });

  // ── Pipeline stage definitions ─────────────────────────────
  const PIPELINE_STAGES = [
    { key: 'plan',          label: 'Planning search strategy' },
    { key: 'search',        label: 'Searching the web' },
    { key: 'gap_analysis',  label: 'Analyzing information gaps' },
    { key: 'synthesis',     label: 'Synthesizing response' },
  ];

  // ── Initialization ─────────────────────────────────────────
  async function init() {
    bindEvents();
    await Promise.all([loadProviders(), loadHistory()]);
    updateModeSlider();
  }

  function bindEvents() {
    // Search
    dom.searchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submitSearch(); }
    });
    dom.btnSearch.addEventListener('click', () => {
      if (state.isStreaming) cancelSearch();
      else submitSearch();
    });

    // New search
    dom.btnNewSearch.addEventListener('click', resetToEmpty);

    // Mode toggle
    dom.modeQuick.addEventListener('click', () => setMode('quick'));
    dom.modeModerate.addEventListener('click', () => setMode('moderate'));
    dom.modeDeep.addEventListener('click', () => setMode('deep'));

    // Thinking panel collapse
    dom.thinkingHeader.addEventListener('click', toggleThinkingPanel);

    // Model selector
    dom.modelSelector.addEventListener('change', () => {
      state.provider = dom.modelSelector.value;
    });

    dom.imagesToggle.addEventListener('change', () => {
      state.includeImages = dom.imagesToggle.checked;
    });

    // Re-align mode toggle slider on window resize, page load, or when custom fonts load
    window.addEventListener('resize', updateModeSlider);
    window.addEventListener('load', updateModeSlider);
    if (document.fonts) {
      document.fonts.ready.then(updateModeSlider);
    }
  }

  // ── Providers ──────────────────────────────────────────────
  async function loadProviders() {
    try {
      const res = await fetch('/api/providers');
      const data = await res.json();
      dom.modelSelector.innerHTML = '';
      for (const [key, info] of Object.entries(data.providers)) {
        const opt = document.createElement('option');
        opt.value = key;
        opt.textContent = info.name;
        dom.modelSelector.appendChild(opt);
      }
      if (data.default) {
        dom.modelSelector.value = data.default;
        state.provider = data.default;
      }
    } catch (err) {
      console.error('Failed to load providers:', err);
      showToast('Could not load model providers.');
    }
  }

  // ── History ────────────────────────────────────────────────
  async function loadHistory() {
    try {
      const res = await fetch('/api/history');
      const data = await res.json();
      renderHistory(data.searches || []);
    } catch (err) {
      console.error('Failed to load history:', err);
    }
  }

  function renderHistory(searches) {
    dom.historyList.innerHTML = '';
    if (!searches.length) {
      dom.historyList.innerHTML = '<div style="padding:12px 8px;font-size:12px;color:var(--text-muted);">No searches yet</div>';
      return;
    }
    searches.forEach((item) => {
      const el = document.createElement('div');
      el.className = 'history-item' + (item.id === state.activeHistoryId ? ' active' : '');
      el.dataset.id = item.id;

      const date = new Date(item.date);
      const timeStr = formatRelativeTime(date);
      const title = item.title || makeFallbackTitle(item.question || '');

      el.innerHTML = `
        <div class="history-item-top">
          <div class="history-item-text">
            <div class="history-item-title">${escapeHtml(title)}</div>
            <div class="history-item-question">${escapeHtml(item.question)}</div>
          </div>
          <button class="history-delete-btn" data-id="${item.id}" type="button" aria-label="Delete search" title="Delete">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="3 6 5 6 21 6"></polyline>
              <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"></path>
              <path d="M10 11v6"></path>
              <path d="M14 11v6"></path>
            </svg>
          </button>
        </div>
        <div class="history-item-meta">
          <span class="history-badge ${item.mode}">${item.mode.charAt(0).toUpperCase() + item.mode.slice(1)}</span>
          <span class="history-item-date">${timeStr}</span>
        </div>
      `;

      // Click on the item to load it
      el.addEventListener('click', (e) => {
        // Don't load if the delete button was clicked
        if (e.target.closest('.history-delete-btn')) return;
        loadHistoryItem(item.id);
      });

      // Delete button
      const deleteBtn = el.querySelector('.history-delete-btn');
      deleteBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        deleteHistoryItem(item.id);
      });

      dom.historyList.appendChild(el);
    });
  }

  async function deleteHistoryItem(id) {
    try {
      const res = await fetch(`/api/history/${id}`, { method: 'DELETE' });
      if (!res.ok) throw new Error('Delete failed');

      // If the deleted item is currently displayed, reset the view
      if (state.activeHistoryId === id) {
        resetToEmpty();
      }

      await loadHistory();
    } catch (err) {
      console.error('Failed to delete history item:', err);
      showToast('Could not delete search.');
    }
  }

  async function loadHistoryItem(id) {
    // Close any active stream
    closeStream();

    state.activeHistoryId = id;
    highlightActiveHistory(id);

    try {
      const res = await fetch(`/api/history/${id}`);
      const data = await res.json();

      dom.searchInput.value = data.question || '';
      hideElement(dom.emptyState);
      hideElement(dom.pipelinePanel);

      // Thinking
      if (data.thinking) {
        showElement(dom.thinkingPanel);
        dom.thinkingContent.textContent = data.thinking;
        const tokenCount = data.thinking.split(/\s+/).length;
        dom.thinkingTokenCt.textContent = `~${tokenCount} tokens`;
        dom.thinkingPanel.classList.remove('collapsed');
      } else {
        hideElement(dom.thinkingPanel);
      }

      // Content
      if (data.content) {
        showElement(dom.responsePanel);
        state.sources = data.sources || [];
        const html = marked.parse(data.content);
        dom.responseContent.innerHTML = html;
        dom.responseContent.querySelectorAll('pre code').forEach((block) => {
          hljs.highlightElement(block);
        });
        linkifyCitations(dom.responseContent);
      } else {
        hideElement(dom.responsePanel);
      }

      // Sources
      state.images = data.images || [];
      if (state.images.length) {
        renderImages(state.images);
        showElement(dom.imagesPanel);
      } else {
        hideElement(dom.imagesPanel);
      }

      if (data.sources && data.sources.length) {
        renderSources(data.sources);
        showElement(dom.sourcesPanel);
      } else {
        hideElement(dom.sourcesPanel);
      }

      renderHistoryMetadata(data.metadata || {});

      // Scroll top
      dom.resultsArea.scrollTop = 0;
    } catch (err) {
      console.error('Failed to load history item:', err);
      showToast('Could not load search result.');
    }
  }

  function highlightActiveHistory(id) {
    $$('.history-item').forEach((el) => {
      el.classList.toggle('active', el.dataset.id === id);
    });
  }

  // ── Mode Toggle ────────────────────────────────────────────
  function setMode(mode) {
    state.mode = mode;
    dom.modeQuick.classList.toggle('active', mode === 'quick');
    dom.modeModerate.classList.toggle('active', mode === 'moderate');
    dom.modeDeep.classList.toggle('active', mode === 'deep');
    updateModeSlider();
  }

  function updateModeSlider() {
    const slider = dom.modeSlider;
    const quickBtn = dom.modeQuick;
    const moderateBtn = dom.modeModerate;
    const deepBtn = dom.modeDeep;

    if (state.mode === 'quick') {
      slider.style.left = quickBtn.offsetLeft + 'px';
      slider.style.width = quickBtn.offsetWidth + 'px';
      slider.className = 'mode-slider quick';
    } else if (state.mode === 'moderate') {
      slider.style.left = moderateBtn.offsetLeft + 'px';
      slider.style.width = moderateBtn.offsetWidth + 'px';
      slider.className = 'mode-slider moderate';
    } else {
      slider.style.left = deepBtn.offsetLeft + 'px';
      slider.style.width = deepBtn.offsetWidth + 'px';
      slider.className = 'mode-slider deep';
    }
  }

  // ── Thinking Panel Toggle ──────────────────────────────────
  function toggleThinkingPanel() {
    dom.thinkingPanel.classList.toggle('collapsed');
  }

  // ── Submit Search ──────────────────────────────────────────
  async function submitSearch() {
    const question = dom.searchInput.value.trim();
    if (!question || state.isStreaming) return;

    state.isStreaming = true;
    state.cancelRequested = false;
    state.contentBuffer = '';
    state.thinkingBuffer = '';
    state.sources = [];
    state.images = [];
    state.activeHistoryId = null;
    setSearchButtonState('cancel');
    dom.imagesToggle.disabled = true;

    // Reset panels
    hideElement(dom.emptyState);
    resetPipeline();
    showElement(dom.pipelinePanel);
    hideElement(dom.thinkingPanel);
    hideElement(dom.responsePanel);
    hideElement(dom.imagesPanel);
    hideElement(dom.sourcesPanel);

    dom.thinkingContent.textContent = '';
    dom.responseContent.innerHTML = '';
    dom.imagesGrid.innerHTML = '';
    dom.sourcesList.innerHTML = '';
    dom.thinkingTokenCt.textContent = '';

    // Scroll to top
    dom.resultsArea.scrollTop = 0;

    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          provider: state.provider,
          mode: state.mode,
          thinking: true,
          include_images: state.includeImages,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `HTTP ${res.status}`);
      }

      const data = await res.json();
      state.currentSearchId = data.search_id;
      openStream(data.search_id);
    } catch (err) {
      console.error('Search submission failed:', err);
      showToast('Search failed: ' + err.message);
      finishStream();
    }
  }

  async function cancelSearch() {
    if (!state.currentSearchId || state.cancelRequested) return;
    state.cancelRequested = true;
    setSearchButtonState('cancel-pending');
    try {
      const res = await fetch(`/api/search/${state.currentSearchId}/cancel`, {
        method: 'POST',
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `HTTP ${res.status}`);
      }
    } catch (err) {
      console.error('Cancel failed:', err);
      showToast('Could not cancel search: ' + err.message);
      state.cancelRequested = false;
      if (state.isStreaming) setSearchButtonState('cancel');
    }
  }

  // ── SSE Stream ─────────────────────────────────────────────
  function openStream(searchId) {
    closeStream();

    const es = new EventSource(`/api/search/${searchId}/stream`);
    state.eventSource = es;

    es.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleStreamMessage(msg);
      } catch (err) {
        console.warn('Bad SSE message:', event.data, err);
      }
    };

    es.onerror = () => {
      // EventSource will try to reconnect by default; we close on terminal errors
      if (es.readyState === EventSource.CLOSED) {
        if (state.isStreaming) {
          showToast('Connection to server was lost.');
          finishStream();
        }
      }
    };
  }

  function closeStream() {
    if (state.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
    if (state.renderTimer) {
      clearTimeout(state.renderTimer);
      state.renderTimer = null;
    }
  }

  function finishStream() {
    state.isStreaming = false;
    state.cancelRequested = false;
    dom.btnSearch.disabled = false;
    dom.imagesToggle.disabled = false;
    setSearchButtonState('search');
    closeStream();
    removeCursor();
  }

  function handleStreamMessage(msg) {
    switch (msg.type) {
      case 'status':
        updatePipelineStage(msg.stage, msg.message);
        break;

      case 'queries':
        showQueries(msg);
        break;

      case 'thinking':
        handleThinkingDelta(msg.delta);
        break;

      case 'content':
        handleContentDelta(msg.delta);
        break;

      case 'sources':
        renderSources(msg.sources);
        showElement(dom.sourcesPanel);
        break;

      case 'images':
        renderImages(msg.images || []);
        if (msg.images && msg.images.length) showElement(dom.imagesPanel);
        break;

      case 'gap_analysis':
        showGapAnalysis(msg);
        break;

      case 'source_fetch':
        showSourceFetch(msg);
        break;

      case 'done':
        handleDone(msg);
        break;

      case 'error':
        showToast(msg.message || 'An error occurred.');
        finishStream();
        break;

      case 'cancelled':
        showToast(msg.message || 'Search cancelled.');
        finishStream();
        break;

      default:
        break;
    }
  }

  // ── Pipeline ───────────────────────────────────────────────
  function resetPipeline() {
    dom.pipelineSteps.innerHTML = '';
    PIPELINE_STAGES.forEach((stage) => {
      const step = document.createElement('div');
      step.className = 'pipeline-step';
      step.id = `step-${stage.key}`;
      step.innerHTML = `
        <div class="step-indicator pending"></div>
        <div class="step-content">
          <div class="step-label">${stage.label}</div>
        </div>
      `;
      dom.pipelineSteps.appendChild(step);
    });
  }

  function updatePipelineStage(stageKey, message) {
    const isValidStage = PIPELINE_STAGES.some((stage) => stage.key === stageKey);
    if (!isValidStage) return;

    // Complete all stages before current
    let reached = false;
    PIPELINE_STAGES.forEach((stage) => {
      const stepEl = $(`#step-${stage.key}`);
      if (!stepEl) return;
      const indicator = stepEl.querySelector('.step-indicator');
      const label = stepEl.querySelector('.step-label');

      if (stage.key === stageKey) {
        reached = true;
        stepEl.className = 'pipeline-step active';
        indicator.className = 'step-indicator active';
        label.textContent = message || stage.label;
      } else if (!reached) {
        // Mark previous stages as completed
        stepEl.className = 'pipeline-step completed';
        indicator.className = 'step-indicator completed';
        indicator.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';
      }
    });
  }

  function showQueries(msg) {
    const container = ensurePipelineContainer('search', 'queries-list');
    if (!container) return;
    const queries = Array.isArray(msg) ? msg : (msg.queries || []);
    const defaultLabel = queries.length === 1 ? 'Query' : 'Queries';
    const label = (msg && msg.label) ? msg.label : defaultLabel;
    appendPipelineCard(container, 'gap-analysis-card query-event-card', `
      <div class="gap-pass">${escapeHtml(label)}</div>
      <div class="gap-followups">
        ${queries.map((q) => `<span class="query-chip">${escapeHtml(q)}</span>`).join('')}
      </div>
    `);
  }

  function showGapAnalysis(msg) {
    const container = ensurePipelineContainer('gap_analysis', 'gap-analysis-list');
    if (!container || !msg.result) return;

    const result = msg.result;
    const followups = Array.isArray(result.followup_queries)
      ? result.followup_queries
      : [];

    appendPipelineCard(container, 'gap-analysis-card', `
      <div class="gap-pass">Checking collected sources</div>
      ${renderFollowups(followups)}
    `);
  }

  function showSourceFetch(msg) {
    const container = ensurePipelineContainer('gap_analysis', 'source-fetch-list');
    if (!container) return;

    const sources = Array.isArray(msg.sources) ? msg.sources : [];
    if (!sources.length && !msg.summary) return;
    const readCount = sources.filter((source) => source.has_page_excerpt).length;
    const countLabel = sources.length
      ? `${readCount}/${sources.length} full text`
      : (msg.summary || '');

    const headerTitle = msg.phase === 'followup' ? 'Follow-up sources read' : 'Sources read';

    appendPipelineCard(container, 'source-fetch-card', `
      <div class="source-fetch-header">
        <span>${escapeHtml(headerTitle)}</span>
        ${countLabel ? `<span class="source-fetch-count">${escapeHtml(countLabel)}</span>` : ''}
      </div>
      ${renderSourceFetchList(sources)}
    `);
  }

  function renderSourceFetchList(sources) {
    if (!sources.length) return '';
    return `
      <div class="source-fetch-items">
        ${sources.map((source) => {
          const title = source.title || source.url || `Source ${source.id || ''}`.trim();
          const domain = source.domain || source.url || '';
          const url = source.url || '#';
          const status = sourceFetchLabel(source);
          const statusClass = source.has_page_excerpt ? 'ready' : 'muted';
          return `
            <a class="source-fetch-item" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">
              <span class="source-fetch-main">
                <span class="source-fetch-title">${escapeHtml(title)}</span>
                ${domain ? `<span class="source-fetch-domain">${escapeHtml(domain)}</span>` : ''}
              </span>
              ${status ? `<span class="source-fetch-badge ${statusClass}">${escapeHtml(status)}</span>` : ''}
            </a>
          `;
        }).join('')}
      </div>
    `;
  }

  function sourceFetchLabel(source) {
    if (source.has_page_excerpt) return 'Full text';
    const status = (source.page_fetch_status || '').toLowerCase();
    if (!status) return '';
    if (status.includes('anti-bot') || status.includes('blocked')) return 'Blocked';
    if (status.includes('too short')) return 'No text';
    if (status.includes('unsupported content type')) return 'Unsupported';
    if (status.includes('timeout')) return 'Timeout';
    if (status.includes('fetch failed') || status.includes('extract failed')) return 'Error';
    if (status.includes('http 401') || status.includes('http 403')) return 'Blocked';
    if (status.includes('http 429')) return 'Rate limit';
    return 'Skipped';
  }

  function ensurePipelineContainer(stageKey, className) {
    const step = $(`#step-${stageKey}`);
    if (!step) return null;
    let container = step.querySelector(`.${className}`);
    if (!container) {
      container = document.createElement('div');
      container.className = className;
      step.querySelector('.step-content').appendChild(container);
    }
    return container;
  }

  function appendPipelineCard(container, className, html) {
    const card = document.createElement('div');
    card.className = className;
    card.innerHTML = html;
    container.appendChild(card);
    return card;
  }

  function renderHistoryMetadata(metadata) {
    const queryEvents = Array.isArray(metadata.query_events)
      ? metadata.query_events
      : [];
    const analyses = Array.isArray(metadata.gap_analyses)
      ? metadata.gap_analyses
      : [];
    const sourceFetchEvents = Array.isArray(metadata.source_fetch_events)
      ? metadata.source_fetch_events
      : Array.isArray(metadata.directed_source_fetches)
      ? metadata.directed_source_fetches
      : [];
    if (!queryEvents.length && !analyses.length && !sourceFetchEvents.length) return;

    resetPipeline();
    showElement(dom.pipelinePanel);
    queryEvents.forEach((event) => {
      showQueries({
        phase: event.phase,
        pass: event.pass,
        label: event.label,
        queries: event.queries || [],
      });
    });
    analyses.forEach((analysis) => {
      showGapAnalysis({
        type: 'gap_analysis',
        mode: analysis.mode,
        pass: analysis.pass,
        result: analysis.result || {},
      });
      sourceFetchEvents
        .filter((fetchEvent) => fetchEvent.pass === analysis.pass)
        .forEach((fetchEvent) => {
          showSourceFetch({
            type: 'source_fetch',
            mode: fetchEvent.mode,
            pass: fetchEvent.pass,
            phase: fetchEvent.phase,
            summary: fetchEvent.summary,
            sources: fetchEvent.sources || [],
          });
        });
    });
    sourceFetchEvents
      .filter((fetchEvent) => !analyses.some(a => a.pass === fetchEvent.pass))
      .forEach((fetchEvent) => {
        showSourceFetch({
          type: 'source_fetch',
          mode: fetchEvent.mode,
          pass: fetchEvent.pass,
          phase: fetchEvent.phase,
          summary: fetchEvent.summary,
          sources: fetchEvent.sources || [],
        });
      });
    PIPELINE_STAGES.forEach((stage) => {
      const stepEl = $(`#step-${stage.key}`);
      if (!stepEl) return;
      const indicator = stepEl.querySelector('.step-indicator');
      stepEl.className = 'pipeline-step completed';
      indicator.className = 'step-indicator completed';
      indicator.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';
    });
  }

  function renderFollowups(followups) {
    if (!followups.length) return '';
    return `
      <div class="gap-followups">
        ${followups.map((q) => `<span class="query-chip">${escapeHtml(q)}</span>`).join('')}
      </div>
    `;
  }

  // ── Thinking ───────────────────────────────────────────────
  function handleThinkingDelta(delta) {
    if (!delta) return;
    state.thinkingBuffer += delta;
    showElement(dom.thinkingPanel);
    dom.thinkingPanel.classList.remove('collapsed');
    dom.thinkingContent.textContent = state.thinkingBuffer;

    // Token count approximation
    const tokens = state.thinkingBuffer.split(/\s+/).filter(Boolean).length;
    dom.thinkingTokenCt.textContent = `~${tokens} tokens`;

    // Auto-scroll thinking panel
    const inner = dom.thinkingBody.querySelector('.thinking-body-inner');
    inner.scrollTop = inner.scrollHeight;
  }

  // ── Content ────────────────────────────────────────────────
  function handleContentDelta(delta) {
    if (!delta) return;
    state.contentBuffer += delta;
    showElement(dom.responsePanel);

    // Throttled markdown render
    if (!state.renderTimer) {
      state.renderTimer = setTimeout(() => {
        renderMarkdown();
        state.renderTimer = null;
      }, 150);
    }
  }

  function renderMarkdown() {
    const html = marked.parse(state.contentBuffer);
    dom.responseContent.innerHTML = html + '<span class="streaming-cursor"></span>';
    // Apply highlight to code blocks
    dom.responseContent.querySelectorAll('pre code:not(.hljs)').forEach((block) => {
      hljs.highlightElement(block);
    });
    linkifyCitations(dom.responseContent);
  }

  function removeCursor() {
    const cursor = dom.responseContent.querySelector('.streaming-cursor');
    if (cursor) cursor.remove();
  }

  // ── Done ───────────────────────────────────────────────────
  function handleDone(msg) {
    // Final content render
    if (msg.content) {
      state.contentBuffer = msg.content;
    }
    // Store final cited sources for citation tooltips.
    state.sources = msg.sources || [];
    if (state.sources.length) {
      renderSources(state.sources);
      showElement(dom.sourcesPanel);
    } else {
      dom.sourcesList.innerHTML = '';
      hideElement(dom.sourcesPanel);
    }
    if (msg.images && msg.images.length) {
      state.images = msg.images;
      renderImages(msg.images);
      showElement(dom.imagesPanel);
    }
    if (state.contentBuffer) {
      showElement(dom.responsePanel);
      dom.responseContent.innerHTML = marked.parse(state.contentBuffer);
      dom.responseContent.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightElement(block);
      });
      linkifyCitations(dom.responseContent);
    }

    // Final thinking
    if (msg.thinking) {
      state.thinkingBuffer = msg.thinking;
      dom.thinkingContent.textContent = state.thinkingBuffer;
      showElement(dom.thinkingPanel);
      const tokens = state.thinkingBuffer.split(/\s+/).filter(Boolean).length;
      dom.thinkingTokenCt.textContent = `~${tokens} tokens`;
    }

    // Complete all pipeline stages and restore default labels
    PIPELINE_STAGES.forEach((stage) => {
      const stepEl = $(`#step-${stage.key}`);
      if (!stepEl) return;
      const indicator = stepEl.querySelector('.step-indicator');
      const label = stepEl.querySelector('.step-label');
      stepEl.className = 'pipeline-step completed';
      indicator.className = 'step-indicator completed';
      indicator.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';
      if (label) {
        label.textContent = stage.label;
      }
    });

    // Wrap up
    state.activeHistoryId = state.currentSearchId;
    finishStream();
    loadHistory(); // refresh sidebar
  }

  // ── Sources ────────────────────────────────────────────────
  function renderSources(sources) {
    state.sources = sources;
    dom.sourcesList.innerHTML = '';
    sources.forEach((src) => {
      const el = document.createElement('a');
      el.className = 'source-item';
      el.href = src.url;
      el.target = '_blank';
      el.rel = 'noopener noreferrer';

      const domain = src.domain || extractDomain(src.url);
      const faviconUrl = `https://www.google.com/s2/favicons?domain=${encodeURIComponent(domain)}&sz=16`;

      el.innerHTML = `
        <div class="source-number">${src.id}</div>
        <div class="source-info">
          <div class="source-title">
            <img src="${faviconUrl}" alt="" loading="lazy" />
            ${escapeHtml(src.title)}
          </div>
          <div class="source-url">${escapeHtml(src.url)}</div>
          ${renderSourceFetchState(src)}
        </div>
        <div class="source-domain-badge">${escapeHtml(domain)}</div>
      `;

      dom.sourcesList.appendChild(el);
    });
  }

  function renderSourceFetchState(src) {
    if (src.has_page_excerpt) {
      return '<div class="source-fetch-state">FULL TEXT</div>';
    }
    if (src.page_fetch_status) {
      return `<div class="source-fetch-state muted">${escapeHtml(src.page_fetch_status)}</div>`;
    }
    return '';
  }

  // ── Image Results ─────────────────────────────────────────
  function renderImages(images) {
    state.images = images || [];
    dom.imagesGrid.innerHTML = '';
    if (!state.images.length) {
      hideElement(dom.imagesPanel);
      return;
    }

    state.images.forEach((image) => {
      const link = document.createElement('a');
      link.className = 'image-result';
      link.href = image.url || image.image_url || image.thumbnail_url || '#';
      link.target = '_blank';
      link.rel = 'noopener noreferrer';

      const displayUrl = image.thumbnail_url || image.image_url || '';
      const domain = image.source_domain || extractDomain(image.url || '');
      const dimensions = formatDimensions(image.width, image.height);

      link.innerHTML = `
        <div class="image-result-media">
          ${displayUrl
            ? `<img src="${escapeAttr(displayUrl)}" alt="${escapeAttr(image.title || 'Image result')}" loading="lazy" />`
            : '<div class="image-result-placeholder">NO IMAGE</div>'}
        </div>
        <div class="image-result-meta">
          <div class="image-result-title">${escapeHtml(image.title || 'Image result')}</div>
          <div class="image-result-source">
            <span>${escapeHtml(domain || 'source')}</span>
            ${dimensions ? `<span>${escapeHtml(dimensions)}</span>` : ''}
          </div>
        </div>
      `;

      const img = link.querySelector('img');
      if (img) {
        img.addEventListener('error', () => {
          const media = link.querySelector('.image-result-media');
          media.innerHTML = '<div class="image-result-placeholder">IMAGE UNAVAILABLE</div>';
        });
      }

      dom.imagesGrid.appendChild(link);
    });
  }

  function formatDimensions(width, height) {
    const w = parseInt(width, 10);
    const h = parseInt(height, 10);
    if (!w || !h) return '';
    return `${w}×${h}`;
  }

  // ── Citation Helpers ───────────────────────────────────────

  /**
   * Convert [N] citation references in rendered HTML into interactive
   * tooltip spans that show source details on hover.
   */
  function linkifyCitations(container) {
    // Walk all text nodes and replace [N] with interactive spans
    const walker = document.createTreeWalker(
      container, NodeFilter.SHOW_TEXT, null
    );
    const textNodes = [];
    while (walker.nextNode()) textNodes.push(walker.currentNode);

    textNodes.forEach((node) => {
      const text = node.textContent;
      if (!/\[\d+\]/.test(text)) return;

      // Don't process text inside <pre> or <code>
      if (node.parentElement && node.parentElement.closest('pre, code')) return;

      const frag = document.createDocumentFragment();
      let lastIndex = 0;
      const regex = /\[(\d+)\]/g;
      let match;

      while ((match = regex.exec(text)) !== null) {
        // Add text before this match
        if (match.index > lastIndex) {
          frag.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
        }

        const num = parseInt(match[1], 10);
        const source = state.sources.find((s) => s.id === num);

        if (source) {
          const span = document.createElement('span');
          span.className = 'citation-ref';
          span.textContent = `[${num}]`;
          span.dataset.sourceId = num;
          span.addEventListener('mouseenter', showCitationTooltip);
          span.addEventListener('mouseleave', hideCitationTooltip);
          span.addEventListener('click', (e) => {
            e.preventDefault();
            window.open(source.url, '_blank', 'noopener,noreferrer');
          });
          frag.appendChild(span);
        } else {
          frag.appendChild(document.createTextNode(match[0]));
        }

        lastIndex = match.index + match[0].length;
      }

      // Add remaining text
      if (lastIndex < text.length) {
        frag.appendChild(document.createTextNode(text.slice(lastIndex)));
      }

      node.parentNode.replaceChild(frag, node);
    });
  }

  // ── Citation Tooltip ───────────────────────────────────────
  let tooltipEl = null;

  function ensureTooltip() {
    if (!tooltipEl) {
      tooltipEl = document.createElement('div');
      tooltipEl.className = 'citation-tooltip';
      tooltipEl.id = 'citation-tooltip';
      document.body.appendChild(tooltipEl);
    }
    return tooltipEl;
  }

  function showCitationTooltip(e) {
    const id = parseInt(e.target.dataset.sourceId, 10);
    const source = state.sources.find((s) => s.id === id);
    if (!source) return;

    const tip = ensureTooltip();
    const domain = source.domain || extractDomain(source.url);
    const faviconUrl = `https://www.google.com/s2/favicons?domain=${encodeURIComponent(domain)}&sz=16`;

    tip.innerHTML = `
      <div class="citation-tooltip-title">
        <img src="${faviconUrl}" alt="" />
        ${escapeHtml(source.title)}
      </div>
      <div class="citation-tooltip-url">${escapeHtml(source.url)}</div>
      <div class="citation-tooltip-domain">${escapeHtml(domain)}</div>
    `;

    // Position above the citation ref
    const rect = e.target.getBoundingClientRect();
    tip.style.display = 'block';
    tip.style.opacity = '0';

    // Measure tooltip size
    const tipRect = tip.getBoundingClientRect();
    let left = rect.left + rect.width / 2 - tipRect.width / 2;
    let top = rect.top - tipRect.height - 8;

    // Keep within viewport
    if (left < 8) left = 8;
    if (left + tipRect.width > window.innerWidth - 8) left = window.innerWidth - tipRect.width - 8;
    if (top < 8) {
      // Show below instead
      top = rect.bottom + 8;
    }

    tip.style.left = left + 'px';
    tip.style.top = top + 'px';
    tip.style.opacity = '1';
  }

  function hideCitationTooltip() {
    if (tooltipEl) {
      tooltipEl.style.opacity = '0';
      setTimeout(() => { if (tooltipEl) tooltipEl.style.display = 'none'; }, 150);
    }
  }

  // ── Reset / New Search ─────────────────────────────────────
  function resetToEmpty() {
    closeStream();
    state.isStreaming = false;
    state.currentSearchId = null;
    state.activeHistoryId = null;
    state.contentBuffer = '';
    state.thinkingBuffer = '';
    state.sources = [];
    state.images = [];
    state.cancelRequested = false;
    dom.btnSearch.disabled = false;
    dom.imagesToggle.disabled = false;
    setSearchButtonState('search');
    dom.searchInput.value = '';

    hideElement(dom.pipelinePanel);
    hideElement(dom.thinkingPanel);
    hideElement(dom.responsePanel);
    hideElement(dom.imagesPanel);
    hideElement(dom.sourcesPanel);
    showElement(dom.emptyState);

    highlightActiveHistory(null);
    dom.searchInput.focus();
  }

  // ── Toast Notifications ────────────────────────────────────
  function showToast(message) {
    const toast = document.createElement('div');
    toast.className = 'error-toast';
    toast.innerHTML = `
      <span class="error-toast-icon">!</span>
      <span class="error-toast-message">${escapeHtml(message)}</span>
      <button class="error-toast-close" type="button">&times;</button>
    `;

    toast.querySelector('.error-toast-close').addEventListener('click', () => dismissToast(toast));
    dom.toastContainer.appendChild(toast);

    // Auto-dismiss after 6 seconds
    setTimeout(() => dismissToast(toast), 6000);
  }

  function dismissToast(toast) {
    if (!toast.parentNode) return;
    toast.classList.add('hiding');
    toast.addEventListener('animationend', () => toast.remove());
  }

  // ── Utility ────────────────────────────────────────────────
  function showElement(el) { el.classList.remove('hidden'); }
  function hideElement(el) { el.classList.add('hidden'); }

  function setSearchButtonState(mode) {
    dom.btnSearch.classList.toggle('cancel-mode', mode !== 'search');
    dom.btnSearch.classList.toggle('cancel-pending', mode === 'cancel-pending');
    dom.btnSearch.disabled = false;
    if (mode === 'search') {
      dom.btnSearch.innerHTML = ICON_SEARCH;
      dom.btnSearch.setAttribute('aria-label', 'Search');
      dom.btnSearch.title = 'Search';
    } else {
      dom.btnSearch.innerHTML = ICON_CANCEL;
      dom.btnSearch.setAttribute('aria-label', 'Cancel search');
      dom.btnSearch.title = mode === 'cancel-pending'
        ? 'Cancelling search'
        : 'Cancel search';
    }
  }

  function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function escapeAttr(str) {
    return escapeHtml(str).replace(/"/g, '&quot;');
  }

  function extractDomain(url) {
    try { return new URL(url).hostname.replace(/^www\./, ''); } catch (_) { return url; }
  }

  function formatRelativeTime(date) {
    const now = new Date();
    const diffMs = now - date;
    const diffMin = Math.floor(diffMs / 60000);
    const diffHr = Math.floor(diffMs / 3600000);
    const diffDay = Math.floor(diffMs / 86400000);

    if (diffMin < 1) return 'Just now';
    if (diffMin < 60) return `${diffMin}m ago`;
    if (diffHr < 24) return `${diffHr}h ago`;
    if (diffDay < 7) return `${diffDay}d ago`;
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  }

  function makeFallbackTitle(question) {
    if (!question) return 'Untitled Search';
    return question
      .replace(/[?.!,;:\s]+$/g, '')
      .split(/\s+/)
      .slice(0, 7)
      .map((word) => word.toUpperCase() === word
        ? word
        : word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ') || 'Untitled Search';
  }

  // ── Boot ───────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', init);
})();
