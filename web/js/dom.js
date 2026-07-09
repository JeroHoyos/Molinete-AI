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
const $modHeaderName  = $('mod-header-name');
const $modHeaderDesc  = $('mod-header-desc');

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

// Training
const $trainContent   = $('train-right-scroll');
const $trainRightDots = $('train-right-dots');
const $progressFill   = $('progress-fill');
const $trainStepLabel = $('train-step-label');
const $trainPctLabel  = $('train-pct-label');
const $metricTrain    = $('metric-train');
const $metricVal      = $('metric-val');
const $metricLr       = $('metric-lr');
const $samplesList    = $('samples-list');
const $samplesSection = $('samples-section');
const $checkpointBadge= $('checkpoint-badge');
const $checkpointPath = $('checkpoint-path');
const $lossChart      = $('loss-chart');
const $lossChartEmpty = $('loss-chart-empty');
const $tmModel        = $('tm-model');
const $tmParams       = $('tm-params');
const $tmSteps        = $('tm-steps');
const $tmLr           = $('tm-lr');
const $tmVocab        = $('tm-vocab');
const $metricPerp     = $('metric-perp');
const $vocabLog       = $('vocab-log');

// Learn
const $outputPre      = $('output-pre');
const $terminalNote   = $('terminal-note');
const $terminalRunDots= $('terminal-run-dots');
const $archTableWrap  = $('arch-table-wrap');
const $archTableBody  = $('arch-table-body');
