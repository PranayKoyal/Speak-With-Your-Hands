import { useState, useEffect, useRef } from 'react';
import { api } from '../services/api';
import { useApp } from '../context/AppContext';

export const useVideoStream = () => {
  const videoRef = useRef<HTMLImageElement | null>(null);
  const { isVideoActive, setIsVideoActive, setError } = useApp();
  const [isLoading, setIsLoading] = useState(false);
  const refreshTimerRef = useRef<NodeJS.Timeout>();

  // Auto-start the video stream when component mounts
  useEffect(() => {
    startStream();
    return () => {
      stopStream();
    };
  }, []); // Run once on mount

  useEffect(() => {
    if (!videoRef.current) return;

    const updateVideoSource = () => {
      if (videoRef.current) {
        videoRef.current.src = `${api.getVideoFeedUrl()}?t=${Date.now()}`; // Add timestamp to prevent caching
      }
    };

    if (isVideoActive) {
      setIsLoading(true);
      try {
        updateVideoSource();
        videoRef.current.onload = () => setIsLoading(false);
        videoRef.current.onerror = () => {
          setError('Failed to load video stream');
          setIsLoading(false);
          setIsVideoActive(false);
        };

        // Refresh the video source every 5 seconds to prevent stale frames
        refreshTimerRef.current = setInterval(updateVideoSource, 5000);
      } catch (error) {
        setError(error instanceof Error ? error.message : 'Failed to start video stream');
        setIsLoading(false);
        setIsVideoActive(false);
      }
    } else {
      if (videoRef.current) {
        videoRef.current.src = '';
      }
      setIsLoading(false);
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
      }
    }

    return () => {
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
      }
    };
  }, [isVideoActive, setError, setIsVideoActive]);

  const startStream = async () => {
    try {
      await api.startProcessing();
      setIsVideoActive(true);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to start processing');
    }
  };

  const stopStream = async () => {
    try {
      await api.stopProcessing();
      setIsVideoActive(false);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to stop processing');
    }
  };

  return {
    videoRef,
    isLoading,
    isVideoActive,
    startStream,
    stopStream,
  };
};

export default useVideoStream; 