import { useEffect } from 'react';

const SelectedTextCapture = ({ onTextSelected }) => {
  useEffect(() => {
    const handleSelection = () => {
      const selectedText = window.getSelection().toString().trim();

      // Only trigger if there's actually selected text
      if (selectedText.length > 0) {
        // Validate selected text length according to data model rules
        if (selectedText.length >= 10 && selectedText.length <= 5000) {
          onTextSelected(selectedText);
        }
      }
    };

    // Listen for selection changes
    document.addEventListener('mouseup', handleSelection);
    document.addEventListener('keyup', handleSelection);

    // Cleanup event listeners
    return () => {
      document.removeEventListener('mouseup', handleSelection);
      document.removeEventListener('keyup', handleSelection);
    };
  }, [onTextSelected]);

  return null; // This component doesn't render anything
};

export default SelectedTextCapture;