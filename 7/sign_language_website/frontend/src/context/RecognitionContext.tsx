import React, { createContext, useContext, useState, ReactNode } from 'react';

interface RecognitionState {
  currentGesture: string | null;
  confidence: number;
  sentence: string[];
  fps: number;
}

interface RecognitionContextType {
  recognitionState: RecognitionState;
  updateRecognitionState: (newState: Partial<RecognitionState>) => void;
  clearSentence: () => void;
  addWordToSentence: (word: string) => void;
}

const initialState: RecognitionState = {
  currentGesture: null,
  confidence: 0,
  sentence: [],
  fps: 0,
};

const RecognitionContext = createContext<RecognitionContextType | undefined>(undefined);

export const RecognitionProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [recognitionState, setRecognitionState] = useState<RecognitionState>(initialState);

  const updateRecognitionState = (newState: Partial<RecognitionState>) => {
    setRecognitionState(prev => ({ ...prev, ...newState }));
  };

  const clearSentence = () => {
    setRecognitionState(prev => ({ ...prev, sentence: [] }));
  };

  const addWordToSentence = (word: string) => {
    setRecognitionState(prev => ({
      ...prev,
      sentence: [...prev.sentence, word],
    }));
  };

  return (
    <RecognitionContext.Provider
      value={{
        recognitionState,
        updateRecognitionState,
        clearSentence,
        addWordToSentence,
      }}
    >
      {children}
    </RecognitionContext.Provider>
  );
};

export const useRecognition = () => {
  const context = useContext(RecognitionContext);
  if (context === undefined) {
    throw new Error('useRecognition must be used within a RecognitionProvider');
  }
  return context;
}; 