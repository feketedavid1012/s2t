// Runs on the audio thread (not the main thread), so mic capture and PCM
// conversion never block the UI. The AudioContext is created at 16 kHz, so the
// samples arriving here are already at Whisper's rate — no resampling needed.
// Frames are batched to ~100 ms (1600 samples) to avoid flooding the socket.
class RecorderWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = new Float32Array(1600);
    this._n = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const ch = input[0]; // Float32Array, 128 samples, mono
    for (let i = 0; i < ch.length; i++) {
      this._buf[this._n++] = ch[i];
      if (this._n === this._buf.length) {
        const pcm = new Int16Array(this._buf.length);
        for (let j = 0; j < pcm.length; j++) {
          const s = Math.max(-1, Math.min(1, this._buf[j]));
          pcm[j] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        this.port.postMessage(pcm.buffer, [pcm.buffer]); // transfer, no copy
        this._n = 0;
      }
    }
    return true;
  }
}

registerProcessor('recorder-worklet', RecorderWorklet);
