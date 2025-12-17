import React, { useState, useEffect } from 'react';
import styles from './ChatBotEmbed.module.css';

const ChatBotEmbed = ({ chainlitUrl = process.env.REACT_APP_CHAINLIT_URL || process.env.NEXT_PUBLIC_CHAINLIT_URL || 'http://localhost:8000' }) => {
  const [iframeLoaded, setIframeLoaded] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Check if the Chainlit service is available
    const checkServiceAvailability = async () => {
      try {
        const response = await fetch(`${chainlitUrl}/health`);
        if (!response.ok) {
          throw new Error(`Service responded with status ${response.status}`);
        }
      } catch (err) {
        setError(`Chainlit service is not accessible at ${chainlitUrl}. Please start the backend server.`);
        console.error('Chatbot service error:', err);
      }
    };

    checkServiceAvailability();
  }, [chainlitUrl]);

  return (
    <div className={styles.chatContainer}>
      <div className={styles.chatHeader}>
        <h3>🤖 AI-Humanoid Robotics Assistant</h3>
        <p>Ask me anything about humanoid robotics!</p>
      </div>
      
      {error ? (
        <div className={styles.errorContainer}>
          <p>{error}</p>
          <p className={styles.hint}>
            To run the chatbot, start the backend server with:
            <code>cd backend-rag && chainlit run app.py -w</code>
          </p>
        </div>
      ) : (
        <div className={styles.iframeContainer}>
          <iframe
            src={chainlitUrl}
            title="AI-Humanoid Robotics Chat Interface"
            className={styles.chatIframe}
            onLoad={() => setIframeLoaded(true)}
            onError={() => setError('Failed to load chat interface')}
            frameBorder="0"
            allow="microphone; camera"
          />
          {!iframeLoaded && (
            <div className={styles.loadingOverlay}>
              <div className={styles.spinner}></div>
              <p>Loading chat interface...</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ChatBotEmbed;