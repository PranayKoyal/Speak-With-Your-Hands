const API_BASE_URL = 'http://localhost:5000';

export interface RecognitionStatus {
  current_gesture: string | null;
  sentence: string[];
  confidence: number;
  fps: number;
}

export interface Settings {
  modelSensitivity: number;
  gestureCooldown: number;
  showLandmarks: boolean;
}

export const api = {
  async getStatus(): Promise<RecognitionStatus> {
    const response = await fetch(`${API_BASE_URL}/status`);
    if (!response.ok) {
      throw new Error('Failed to fetch recognition status');
    }
    return response.json();
  },

  async updateSettings(settings: Settings): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/update_settings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(settings),
    });

    if (!response.ok) {
      throw new Error('Failed to update settings');
    }
  },

  async startProcessing(): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/start_processing`);
    if (!response.ok) {
      throw new Error('Failed to start processing');
    }
  },

  async stopProcessing(): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/stop_processing`);
    if (!response.ok) {
      throw new Error('Failed to stop processing');
    }
  },

  getVideoFeedUrl(): string {
    return `${API_BASE_URL}/video_feed`;
  },
};

export default api; 