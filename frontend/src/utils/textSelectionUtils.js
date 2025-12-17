export const textSelectionUtils = {
  getSelectedText() {
    const selection = window.getSelection();
    return selection.toString().trim();
  },

  getSelectedRange() {
    const selection = window.getSelection();
    if (selection.rangeCount > 0) {
      return selection.getRangeAt(0);
    }
    return null;
  },

  getSelectedTextWithContext(contextLength = 50) {
    const selection = window.getSelection();
    if (selection.rangeCount === 0) return { text: '', context: '' };

    const selectedText = selection.toString().trim();
    if (!selectedText) return { text: '', context: '' };

    // Get context by expanding the range
    const range = selection.getRangeAt(0).cloneRange();
    const startContainer = range.startContainer;
    const startOffset = range.startOffset;

    // Try to get more context around the selection
    try {
      range.setStart(startContainer, Math.max(0, startOffset - contextLength));
    } catch (e) {
      // If we can't set the start, continue with the original range
    }

    try {
      range.setEnd(startContainer, Math.min(startContainer.length || 0, startOffset + selectedText.length + contextLength));
    } catch (e) {
      // If we can't set the end, continue with the original range
    }

    const contextText = range.toString();

    return {
      text: selectedText,
      context: contextText
    };
  }
};