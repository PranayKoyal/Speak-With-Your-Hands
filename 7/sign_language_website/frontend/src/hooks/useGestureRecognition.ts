import { useEffect, useCallback, useRef } from 'react';
import { api } from '../services/api';
import { useApp } from '../context/AppContext';
import { useRecognition } from '../context/RecognitionContext';
import { useSettings } from '../context/SettingsContext';

export const useGestureRecognition = () => {
  const { isVideoActive, setError } = useApp();
  const { recognitionState, updateRecognitionState } = useRecognition();
  const { settings } = useSettings();
  const pollInterval = useRef<NodeJS.Timeout>();

  const fetchRecognitionStatus = useCallback(async () => {
    try {
      const status = await api.getStatus();
      if (status.current_gesture) {
        updateRecognitionState({
          currentGesture: status.current_gesture,
          confidence: status.confidence,
          fps: status.fps,
        });
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to fetch recognition status');
    }
  }, [setError, updateRecognitionState]);

  useEffect(() => {
    if (isVideoActive) {
      // Start polling for recognition status
      pollInterval.current = setInterval(fetchRecognitionStatus, 500); // Poll less frequently
      
      // Initial fetch
      fetchRecognitionStatus();

      return () => {
        if (pollInterval.current) {
          clearInterval(pollInterval.current);
        }
      };
    } else if (pollInterval.current) {
      clearInterval(pollInterval.current);
    }
  }, [isVideoActive, fetchRecognitionStatus]);

  return {
    currentGesture: recognitionState.currentGesture,
    confidence: recognitionState.confidence,
    fps: recognitionState.fps,
  };
};

export default useGestureRecognition; 