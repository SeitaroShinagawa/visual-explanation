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
      stepButton.disabled = false; // stepping is always available; it pauses autoplay first
    };

    // Stepping seeks every animation in the figure to the middle of the next stage
    // (the highlighted part of each stage's keyframes) and pauses it there, instead of
    // letting time run for one stage length: that could stop on a transition keyframe
    // where every panel is dim. Falls back to the timed step without the Web Animations API.
    const stepMs = Math.max(120, Math.round(cycleMs / stepCount));
    const figureAnimations = () =>
      (typeof figure.getAnimations === 'function' ? figure.getAnimations({ subtree: true }) : []);

    const seekToNextStage = () => {
      const anims = figureAnimations();
      if (!anims.length) return false;
      const ref = anims.find((a) => a.effect && a.effect.getTiming && a.effect.getTiming().duration === cycleMs) || anims[0];
      const t = Number(ref.currentTime) || 0;
      const cycleStart = Math.floor(t / cycleMs) * cycleMs;
      let next = Math.floor((t - cycleStart) / stepMs) + 1;
      let base = cycleStart;
      if (next >= stepCount) { next = 0; base += cycleMs; }
      const target = base + next * stepMs + stepMs / 2;
      anims.forEach((a) => { a.pause(); a.currentTime = target; });
      return true;
    };

    const startOneStep = () => {
      stopStepTimer();
      state.playing = false;
      autoplayButton.textContent = '自動再生: OFF';
      figure.classList.add('is-paused');
      if (seekToNextStage()) return;

      figure.classList.remove('is-paused');
      stepButton.disabled = true;
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
      // animations paused by a step were paused through the Web Animations API, so resume them explicitly
      if (state.playing) figureAnimations().forEach((a) => a.play());
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
