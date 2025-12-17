import React, { useState, useEffect } from 'react';
import Layout from '@theme/Layout';
import ChatWidget from '@site/src/components/ChatKit/ChatWidget';

// Wrapper component that adds the chat widget to any layout
export default function LayoutWithChatWidget(props) {
  const [showChat, setShowChat] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    // Check if we're on mobile to adjust UI accordingly
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768);
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);

    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  return (
    <Layout {...props}>
      {/* Render the original content */}
      {props.children}

      {/* Floating chat button */}
      <button
        onClick={() => setShowChat(!showChat)}
        style={{
          position: 'fixed',
          bottom: '20px',
          right: '20px',
          width: '60px',
          height: '60px',
          borderRadius: '50%',
          backgroundColor: '#005ea6',
          color: 'white',
          border: 'none',
          fontSize: '24px',
          cursor: 'pointer',
          zIndex: 1000,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 4px 8px rgba(0,0,0,0.2)',
        }}
      >
        💬
      </button>

      {/* Chat widget - only show when toggled */}
      {showChat && (
        <div
          style={{
            position: 'fixed',
            bottom: '90px',
            right: '20px',
            zIndex: 1000,
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            borderRadius: '8px',
            width: isMobile ? '90%' : '400px',
          }}
        >
          <ChatWidget />
        </div>
      )}
    </Layout>
  );
}