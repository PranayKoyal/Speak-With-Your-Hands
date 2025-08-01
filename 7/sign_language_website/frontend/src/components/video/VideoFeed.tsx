import React from 'react';
import { useVideoStream } from '../../hooks/useVideoStream';

export const VideoFeed: React.FC = () => {
  const { videoRef, isLoading } = useVideoStream();

  return (
    <div className="relative w-full aspect-video bg-gray-900 rounded-lg overflow-hidden">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-75">
          <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent"></div>
        </div>
      )}
      <img
        ref={videoRef}
        className="w-full h-full object-contain"
        alt="Video feed"
      />
    </div>
  );
};

export default VideoFeed; 