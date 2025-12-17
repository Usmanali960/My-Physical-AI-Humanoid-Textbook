import React, { useState, useEffect } from 'react';

// Component to embed Chainlit UI in Docusaurus
const ChainlitEmbed = () => {
  const [chainlitUrl] = useState(process.env.REACT_APP_CHAINLIT_URL || process.env.NEXT_PUBLIC_CHAINLIT_URL || 'http://localhost:8001');
  const [isChainlitAvailable, setIsChainlitAvailable] = useState(true);

  // Check if Chainlit is available
  useEffect(() => {
    const checkChainlit = async () => {
      try {
        const response = await fetch(`${chainlitUrl}/health`, { method: 'GET' });
        setIsChainlitAvailable(response.ok);
      } catch (error) {
        setIsChainlitAvailable(false);
      }
    };

    checkChainlit();

    // Check periodically if the service is available
    const interval = setInterval(checkChainlit, 10000); // Check every 10 seconds

    return () => clearInterval(interval);
  }, [chainlitUrl]);

  if (!isChainlitAvailable) {
    return (
      <div className="chainlit-container" style={{ padding: '20px', textAlign: 'center' }}>
        <div style={{ 
          backgroundColor: '#f8f9fa', 
          padding: '30px', 
          borderRadius: '8px', 
          border: '2px dashed #ccc',
          maxWidth: '600px',
          margin: '0 auto'
        }}>
          <h3>Chainlit UI Not Available</h3>
          <p>To use the AI assistant, please ensure both servers are running:</p>
          <ol style={{ textAlign: 'left' }}>
            <li>The Backend API server on port 8000: <code>uvicorn main:app --reload --port 8000</code></li>
            <li>The Chainlit UI server on port 8001: <code>chainlit run chainlit_app.py --port 8001</code></li>
          </ol>
          <p>Once both services are running, this interface will connect automatically.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="chainlit-container">
      <iframe
        src={chainlitUrl}
        width="100%"
        height="600px"
        style={{ border: '1px solid #ccc', borderRadius: '8px' }}
        title="Chainlit AI Assistant"
      >
        <p>Your browser does not support iframes. The Chainlit UI cannot be displayed.</p>
      </iframe>
    </div>
  );
};

export default ChainlitEmbed;