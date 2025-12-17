import React, { useState, useEffect } from 'react';
import { textSelectionUtils } from '../utils/textSelectionUtils';

interface SelectedTextIndicatorProps {
  onTextSelected: (selectedText: string) => void;
}

const SelectedTextIndicator: React.FC<SelectedTextIndicatorProps> = ({ onTextSelected }) => {
  const [selectedText, setSelectedText] = useState('');
  const [showIndicator, setShowIndicator] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleSelection = () => {
      const text = textSelectionUtils.getSelectedText();
      const range = textSelectionUtils.getSelectedRange();

      if (text && range) {
        // Get the bounding rectangle for the selection to position the indicator
        const rect = range.getBoundingClientRect();
        setPosition({ x: rect.left, y: rect.top - 10 });
        setSelectedText(text);
        setShowIndicator(true);
        onTextSelected(text);
      } else {
        setShowIndicator(false);
        setSelectedText('');
        onTextSelected('');
      }
    };

    // Add event listeners for text selection
    document.addEventListener('mouseup', handleSelection);
    document.addEventListener('keyup', handleSelection);

    // Clean up event listeners
    return () => {
      document.removeEventListener('mouseup', handleSelection);
      document.removeEventListener('keyup', handleSelection);
    };
  }, [onTextSelected]);

  if (!showIndicator || !selectedText) {
    return null;
  }

  // Limit the displayed text length for the indicator
  const displayText = selectedText.length > 50
    ? selectedText.substring(0, 50) + '...'
    : selectedText;

  return (
    <div
      className="selected-text-indicator fixed z-50 bg-blue-500 text-white text-xs px-2 py-1 rounded shadow-lg"
      style={{
        left: `${position.x}px`,
        top: `${position.y}px`,
      }}
    >
      <span title={selectedText}>{displayText}</span>
    </div>
  );
};

export default SelectedTextIndicator;