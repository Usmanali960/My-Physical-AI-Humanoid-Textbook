import React, { useState, useEffect, useRef } from 'react';
import styles from './styles.module.css';
import { useColorMode } from '@docusaurus/theme-common';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  citations?: Citation[];
}

interface Citation {
  section: string;
  page?: string;
  paragraph?: string;
  text_excerpt: string;
  relevance_score: number;
}

interface ChatProps {
  backendUrl?: string;
  sessionId?: string;
}

const DocumentationChatWidget: React.FC<ChatProps> = ({
  backendUrl = process.env.REACT_APP_API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000',
  sessionId: initialSessionId
}) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(initialSessionId || '');
  const [selectedText, setSelectedText] = useState<string | null>(null);
  const { colorMode } = useColorMode();
  const messagesEndRef = useRef<null | HTMLDivElement>(null);

  // Function to get selected text from the page
  useEffect(() => {
    const handleSelection = () => {
      const selectedText = window.getSelection()?.toString().trim();
      if (selectedText) {
        setSelectedText(selectedText);
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => {
      document.removeEventListener('mouseup', handleSelection);
    };
  }, []);

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    // Add user message to chat
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Prepare context based on selected text
      const userContext = {
        selected_text: selectedText || undefined,
        session_id: sessionId || undefined,
      };

      // Make API call to backend
      const response = await fetch(`${backendUrl}/chat-v2`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputValue,
          session_id: sessionId,
          user_context: userContext,
        }),
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      const data = await response.json();

      // Update session ID if new one was provided
      if (data.session_id && !sessionId) {
        setSessionId(data.session_id);
      }

      // Add assistant message to chat
      const assistantMessage: Message = {
        id: data.message_id || Date.now().toString(),
        role: 'assistant',
        content: data.response,
        timestamp: new Date(data.timestamp || Date.now()),
        citations: data.citations || [],
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: Date.now().toString(),
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setSelectedText(null); // Clear selected text after sending
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setSessionId('');
  };

  return (
    <div className={`${styles.chatWidget} ${colorMode}`}>
      <div className={styles.chatHeader}>
        <h3>AI Textbook Assistant</h3>
        <div className={styles.headerActions}>
          <button onClick={clearChat} className={styles.clearBtn} title="Clear chat">
            ✕
          </button>
        </div>
      </div>

      <div className={styles.chatMessages}>
        {messages.length === 0 ? (
          <div className={styles.welcomeMessage}>
            <p>Hello! I'm your AI assistant for the AI-Humanoid Robotics textbook.</p>
            <p>Ask me anything about the content on this page, or select text and ask questions about it.</p>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`${styles.message} ${styles[message.role]}`}
            >
              <div className={styles.messageContent}>
                <div className={styles.messageText}>
                  {message.content}
                </div>

                {message.citations && message.citations.length > 0 && (
                  <div className={styles.citations}>
                    <h4>Sources:</h4>
                    <ul>
                      {message.citations.map((citation, idx) => (
                        <li key={idx}>
                          <strong>{citation.section}</strong>
                          {citation.page && ` > Page ${citation.page}`}
                          {citation.paragraph && ` > ${citation.paragraph}`}
                          <p className={styles.citationExcerpt}>"{citation.text_excerpt}"</p>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
              <div className={styles.messageTimestamp}>
                {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className={`${styles.message} ${styles.assistant}`}>
            <div className={styles.messageContent}>
              <div className={styles.typingIndicator}>
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {selectedText && (
        <div className={styles.selectedTextPreview}>
          <p><strong>Selected text:</strong> "{selectedText}"</p>
        </div>
      )}

      <form className={styles.chatInputForm} onSubmit={handleSubmit}>
        <textarea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={selectedText
            ? "Ask about selected text..."
            : "Ask a question about the textbook content..."}
          rows={2}
          disabled={isLoading}
          className={styles.chatInput}
        />
        <button
          type="submit"
          disabled={!inputValue.trim() || isLoading}
          className={styles.sendButton}
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>

      <div className={styles.chatFooter}>
        <small>
          Powered by Advanced RAG Chatbot • Answers are based on textbook content
        </small>
      </div>
    </div>
  );
};

export default DocumentationChatWidget;