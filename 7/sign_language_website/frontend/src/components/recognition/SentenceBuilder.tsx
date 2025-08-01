import React from 'react';
import { useRecognition } from '../../context/RecognitionContext';
import { useSpeechSynthesis } from '../../hooks/useSpeechSynthesis';

export const SentenceBuilder: React.FC = () => {
  const { recognitionState, clearSentence } = useRecognition();
  const { speak } = useSpeechSynthesis();

  const handleSpeak = () => {
    if (recognitionState.sentence.length > 0) {
      speak(recognitionState.sentence.join(' '));
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold">Constructed Sentence</h2>
        <div className="flex gap-2">
          <button
            onClick={handleSpeak}
            disabled={recognitionState.sentence.length === 0}
            className={`px-4 py-2 rounded-lg font-medium ${
              recognitionState.sentence.length === 0
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-green-500 hover:bg-green-600 text-white'
            }`}
          >
            Speak
          </button>
          <button
            onClick={clearSentence}
            disabled={recognitionState.sentence.length === 0}
            className={`px-4 py-2 rounded-lg font-medium ${
              recognitionState.sentence.length === 0
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-red-500 hover:bg-red-600 text-white'
            }`}
          >
            Clear
          </button>
        </div>
      </div>
      <div className="min-h-[100px] p-4 bg-gray-50 rounded-lg">
        {recognitionState.sentence.length > 0 ? (
          <p className="text-lg">
            {recognitionState.sentence.join(' ')}
          </p>
        ) : (
          <p className="text-gray-500 italic">
            No words in sentence yet. Detected gestures will appear here.
          </p>
        )}
      </div>
    </div>
  );
};

export default SentenceBuilder; 