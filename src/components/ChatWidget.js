import React, { useState, useRef, useEffect } from 'react';

export default function ChatWidget() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bodyRef = useRef(null);

  const sendMessage = async () => {
    if (!input.trim()) return;

    // Add user message
    const userMsg = { sender: 'user', text: input };
    setMessages(prev => [...prev, userMsg]);
    const currentInput = input;
    setInput('');
    setLoading(true);

    try {
      // Backend call to Hugging Face RAG Agent
      const res = await fetch(
        'https://syedbabarmehmoodzaidi-book1.hf.space/ask',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: currentInput }),
        }
      );

      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const data = await res.json();

      const botReply = data.answer || 'No response from agent';
      setMessages(prev => [...prev, { sender: 'bot', text: botReply }]);
    } catch (err) {
      // Mock fallback if backend unavailable
      setMessages(prev => [
        ...prev,
        { sender: 'bot', text: `You typed: "${currentInput}"` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Auto-scroll to bottom when new messages added
  useEffect(() => {
    if (bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [messages, loading]);

  return (
    <div className="chat-container">
      <button className="chat-button" onClick={() => setOpen(true)}>
        💬 Chat
      </button>

      {open && (
        <div className="chat-box">
          <div className="chat-header">
            AI Assistant
            <button
              className="chat-close"
              onClick={() => setOpen(false)}
              title="Close"
            >
              ×
            </button>
          </div>

          <div className="chat-body" ref={bodyRef}>
            {messages.map((m, i) => (
              <div key={i} className={`bubble ${m.sender}`}>
                {m.text}
              </div>
            ))}
            {loading && <div className="bubble bot">Typing...</div>}
          </div>

          <div className="chat-input">
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Type message..."
              onKeyDown={e => e.key === 'Enter' && sendMessage()}
              disabled={loading}
            />
            <button onClick={sendMessage} disabled={loading}>
              {loading ? '...' : 'Send'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
