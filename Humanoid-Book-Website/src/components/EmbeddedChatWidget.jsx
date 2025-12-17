import React from 'react';
import ChatBotEmbed from '../components/ChatBotEmbed/ChatBotEmbed';

// Simple wrapper component to embed the chatbot in docs pages
const EmbeddedChatWidget = () => {
  return (
    <div style={{ marginTop: '2rem' }}>
      <h3>Need Help with Humanoid Robotics Concepts?</h3>
      <ChatBotEmbed />
    </div>
  );
};

export default EmbeddedChatWidget;