import { useCallback, useEffect } from 'react';
import { useSettings } from '../context/SettingsContext';
import speechService, { SpeechSettings } from '../services/speechService';

export const useSpeechSynthesis = () => {
  const { settings } = useSettings();

  useEffect(() => {
    speechService.updateSettings({
      rate: settings.voiceRate,
      pitch: settings.voicePitch,
      enabled: settings.enableVoice,
    });
  }, [settings.voiceRate, settings.voicePitch, settings.enableVoice]);

  const speak = useCallback((text: string) => {
    speechService.speak(text);
  }, []);

  const stop = useCallback(() => {
    speechService.stop();
  }, []);

  const getVoices = useCallback(() => {
    return speechService.getVoices();
  }, []);

  return {
    speak,
    stop,
    getVoices,
  };
};

export default useSpeechSynthesis; 