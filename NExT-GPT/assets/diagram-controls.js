(() => {
  const diagrams = document.querySelectorAll('figure.diagram[data-cycle-ms][data-step-count]');
  if (!diagrams.length) return;

  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  diagrams.forEach((figure) => {
    const cycleMs = Number(figure.dataset.cycleMs);
    const stepCount = Number(figure.dataset.stepCount);
    if (!Number.isFinite(cycleMs) || !Number.isFinite(stepCount) || cycleMs <= 0 || stepCount <= 0) return;

    const controls = document.createElement('div');
    controls.className = 'diagram-controls';

    const autoplayButton = document.createElement('button');
    autoplayButton.type = 'button';
    autoplayButton.textContent = '自動再生: ON';

    const stepButton = document.createElement('button');
    stepButton.type = 'button';
    stepButton.textContent = '1ステップ進む';

    controls.appendChild(autoplayButton);
    controls.appendChild(stepButton);
    figure.insertBefore(controls, figure.firstChild);

    const state = {
      playing: !prefersReducedMotion,
      steppingTimer: null,
    };

    const stopStepTimer = () => {
      if (!state.steppingTimer) return;
      clearTimeout(state.steppingTimer);
      state.steppingTimer = null;
    };

    const applyState = () => {
      figure.classList.toggle('is-paused', !state.playing);
      autoplayButton.textContent = state.playing ? '自動再生: ON' : '自動再生: OFF';
      stepButton.disabled = state.playing;
    };

    const startOneStep = () => {
      stopStepTimer();
      state.playing = false;
      figure.classList.remove('is-paused');
      autoplayButton.textContent = '自動再生: OFF';
      stepButton.disabled = true;

      const stepMs = Math.max(120, Math.round(cycleMs / stepCount));
      state.steppingTimer = window.setTimeout(() => {
        figure.classList.add('is-paused');
        stepButton.disabled = false;
        state.steppingTimer = null;
      }, stepMs);
    };

    autoplayButton.addEventListener('click', () => {
      stopStepTimer();
      state.playing = !state.playing;
      applyState();
    });

    stepButton.addEventListener('click', () => {
      if (state.playing) {
        state.playing = false;
      }
      startOneStep();
    });

    if (prefersReducedMotion) {
      autoplayButton.disabled = true;
      stepButton.disabled = true;
      autoplayButton.textContent = '自動再生: OFF';
      figure.classList.add('is-paused');
      return;
    }

    applyState();
  });
})();
