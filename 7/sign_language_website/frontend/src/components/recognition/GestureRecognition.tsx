import React from 'react';
import { useGestureRecognition } from '../../hooks/useGestureRecognition';

export const GestureRecognition: React.FC = () => {
  const { currentGesture, confidence, fps } = useGestureRecognition();

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <h2 className="text-xl font-semibold mb-4">Current Recognition</h2>
      <div className="space-y-4">
        <div>
          <p className="text-gray-600 mb-1">Detected Gesture</p>
          <p className="text-2xl font-bold">
            {currentGesture || 'No gesture detected'}
          </p>
        </div>
        <div>
          <p className="text-gray-600 mb-1">Confidence</p>
          <div className="relative w-full h-4 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="absolute h-full bg-blue-500 transition-all duration-300"
              style={{ width: `${(confidence || 0) * 100}%` }}
            />
          </div>
          <p className="text-sm text-gray-500 mt-1">
            {((confidence || 0) * 100).toFixed(1)}%
          </p>
        </div>
        <div>
          <p className="text-gray-600 mb-1">FPS</p>
          <p className="text-lg font-medium">{fps?.toFixed(1) || '0'}</p>
        </div>
      </div>
    </div>
  );
};

export default GestureRecognition; 