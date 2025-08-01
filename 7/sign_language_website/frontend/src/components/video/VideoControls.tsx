import React from 'react';
import { useVideoStream } from '../../hooks/useVideoStream';

export const VideoControls: React.FC = () => {
  const { isVideoActive, startStream, stopStream } = useVideoStream();

  return (
    <div className="flex gap-4 justify-center mt-4">
      <button
        onClick={isVideoActive ? stopStream : startStream}
        className={`px-6 py-2 rounded-lg font-medium transition-colors ${
          isVideoActive
            ? 'bg-red-500 hover:bg-red-600 text-white'
            : 'bg-blue-500 hover:bg-blue-600 text-white'
        }`}
      >
        {isVideoActive ? 'Stop Camera' : 'Start Camera'}
      </button>
    </div>
  );
};

export default VideoControls; 