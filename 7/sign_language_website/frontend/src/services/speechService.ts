export interface SpeechSettings {
  rate: number;
  pitch: number;
  enabled: boolean;
}

class SpeechService {
  private synthesis: SpeechSynthesis;
  private settings: SpeechSettings = {
    rate: 1,
    pitch: 1,
    enabled: true,
  };

  constructor() {
    this.synthesis = window.speechSynthesis;
  }

  updateSettings(newSettings: Partial<SpeechSettings>) {
    this.settings = { ...this.settings, ...newSettings };
  }

  speak(text: string): void {
    if (!this.settings.enabled || !text) return;

    // Cancel any ongoing speech
    this.synthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = this.settings.rate;
    utterance.pitch = this.settings.pitch;

    // Use the default voice
    const voices = this.synthesis.getVoices();
    if (voices.length > 0) {
      utterance.voice = voices[0];
    }

    this.synthesis.speak(utterance);
  }

  stop(): void {
    this.synthesis.cancel();
  }

  getVoices(): SpeechSynthesisVoice[] {
    return this.synthesis.getVoices();
  }
}

export const speechService = new SpeechService();
export default speechService; 