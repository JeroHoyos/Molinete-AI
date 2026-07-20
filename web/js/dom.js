'use strict';

// ─────────────────────────────────────────────────
// Referencias al DOM
// ─────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const $homeView       = $('home-view');
const $moduleView     = $('module-view');
const $hdrHome        = $('hdr-home');
const $hdrModule      = $('hdr-module');
const $hdrTaskName    = $('hdr-task-name');
const $hdrTaskSub     = $('hdr-task-sub');
const $hdrRunDots     = $('hdr-run-dots');
const $stopBtn        = $('stop-btn');
const $clearBtn       = $('clear-btn');
const $backBtn        = $('back-btn');
const $statusDot      = $('status-dot');
const $statusText     = $('status-text');

// Views
const $chatView       = $('chat-view');
const $trainView      = $('train-view');
const $learnView      = $('learn-view');

// Chat
const $chatMessages   = $('chat-messages');
const $chatBubbleList = $('chat-bubble-list');
const $chatWelcome    = $('chat-welcome');
const $chatWelcomeText= $('chat-welcome-text');
const $chatLoading    = $('chat-loading');
const $chatThinking   = $('chat-thinking');
const $inputBar       = $('input-bar');
const $userInput      = $('user-input');
const $sendBtn        = $('send-btn');
const $chatSuggest    = $('chat-suggest');
const $chatToolbar    = $('chat-toolbar');
const $chatModelName  = $('chat-model-name');
const $tempValue      = $('temp-value');
const $maxValue       = $('max-value');

// Training
const $trainContent   = $('train-right-scroll');
const $trainRightDots = $('train-right-dots');
const $progressFill   = $('progress-fill');
const $trainStepLabel = $('train-step-label');
const $trainPctLabel  = $('train-pct-label');
const $metricTrain    = $('metric-train');
const $metricVal      = $('metric-val');
const $metricValBest  = $('metric-val-best');
const $metricLr       = $('metric-lr');
const $metricPerp     = $('metric-perp');
const $samplesList    = $('samples-list');
const $checkpointBadge= $('checkpoint-badge');
const $checkpointPath = $('checkpoint-path');
const $lossChart      = $('loss-chart');
const $lossChartBody  = $('loss-chart-body');
const $lossChartEmpty = $('loss-chart-empty');
const $lcTip          = $('lc-tip');
const $lcTipStep      = $('lc-tip-step');
const $lcTipTrain     = $('lc-tip-train');
const $lcTipVal       = $('lc-tip-val');
const $tmModel        = $('tm-model');
const $tmParams       = $('tm-params');
const $tmSteps        = $('tm-steps');
const $tmLr           = $('tm-lr');
const $tmVocab        = $('tm-vocab');
const $tmCorpus       = $('tm-corpus');
const $ttElapsed      = $('tt-elapsed');
const $ttSpeed        = $('tt-speed');
const $ttEta          = $('tt-eta');
const $prepCard       = $('prep-card');
const $prepList       = $('prep-list');
const $prepRaw        = $('prep-raw');
const $tokExamples    = $('tok-examples');
const $tokExamplesList= $('tok-examples-list');
const $vocabLog       = $('vocab-log');

// Learn
const $outputPre      = $('output-pre');
const $terminalNote   = $('terminal-note');
const $terminalRunDots= $('terminal-run-dots');
const $archTableWrap  = $('arch-table-wrap');
const $archTableBody  = $('arch-table-body');

// Descarga del corpus (módulo 11)
const $dlCard         = $('dl-card');
const $dlDest         = $('dl-dest');
const $dlConfirm      = $('dl-confirm');
const $dlConfirmText  = $('dl-confirm-text');
const $dlList         = $('dl-list');
const $dlProgress     = $('dl-progress');
const $dlProgressFill = $('dl-progress-fill');
const $dlDone         = $('dl-done');
const $dlDoneText     = $('dl-done-text');
const $dlNote         = $('dl-note');
const $dlRaw          = $('dl-raw');
const $dlLog          = $('dl-log');
