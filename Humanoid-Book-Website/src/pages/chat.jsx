import React from 'react';
import Layout from '@theme/Layout';
import ChatBotEmbed from '../components/ChatBotEmbed/ChatBotEmbed';

function ChatPage() {
  return (
    <Layout
      title="AI-Humanoid Robotics Chatbot"
      description="Interactive chatbot for AI-Humanoid Robotics knowledge">
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--12">
            <h1>🤖 AI-Humanoid Robotics Assistant</h1>
            <p>
              Interact with our AI-powered chatbot to learn more about humanoid robotics.
              Ask questions about mechanics, AI implementations, sensors, actuators, locomotion, and more!
            </p>

            <ChatBotEmbed chainlitUrl={`${window.location.protocol}//${window.location.hostname}:${window.env?.CHAINLIT_PORT || 8000}`} />
          </div>
        </div>
      </div>
    </Layout>
  );
}

export default ChatPage;