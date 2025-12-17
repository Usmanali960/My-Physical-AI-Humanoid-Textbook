// Global chat widget that respects Docusaurus context
import React, { useEffect } from 'react';
import { createRoot } from 'react-dom/client';

// Function to create and inject the chat widget
function createGlobalChatWidget() {
  // Create button container
  const buttonContainer = document.createElement('div');
  buttonContainer.id = 'global-chat-button';
  buttonContainer.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 10001;
  `;

  // Create the toggle button
  const toggleButton = document.createElement('button');
  toggleButton.innerHTML = '💬';
  toggleButton.style.cssText = `
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background-color: #005ea6;
    color: white;
    border: none;
    font-size: 24px;
    cursor: pointer;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    display: flex;
    align-items: center;
    justify-content: center;
  `;

  // Add button to button container
  buttonContainer.appendChild(toggleButton);
  document.body.appendChild(buttonContainer);

  // Create widget container (initially hidden)
  const widgetContainer = document.createElement('div');
  widgetContainer.id = 'global-chat-widget';
  widgetContainer.style.cssText = `
    position: fixed;
    bottom: 90px;
    right: 20px;
    z-index: 10000;
    display: none;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    border-radius: 8px;
    width: 400px;
  `;
  document.body.appendChild(widgetContainer);

  // Add toggle functionality
  let isChatOpen = false;
  let chatWidgetRoot = null;

  toggleButton.addEventListener('click', () => {
    isChatOpen = !isChatOpen;
    buttonContainer.style.display = isChatOpen ? 'none' : 'block';
    widgetContainer.style.display = isChatOpen ? 'block' : 'none';

    if (isChatOpen && !chatWidgetRoot) {
      // Use dynamic import to load ChatWidget inside the React ecosystem
      import('@site/src/components/ChatKit/ChatWidget').then((module) => {
        const ChatWidget = module.default;
        chatWidgetRoot = createRoot(widgetContainer);
        chatWidgetRoot.render(
          React.createElement(ChatWidget, { backendUrl: 'http://localhost:8000' })
        );
      });
    }
  });
}

// Execute when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', createGlobalChatWidget);
} else {
  // DOM is already ready
  createGlobalChatWidget();
}

export default {};