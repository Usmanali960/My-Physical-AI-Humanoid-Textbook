import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';
import { createRoot } from 'react-dom/client';

import Personalize from './components/Personalize';
import LanguageToggle from './components/LanguageToggle';

if (ExecutionEnvironment.canUseDOM) {
  // Render Personalize component
  window.addEventListener('load', () => {
    const personalizeContainer = document.getElementById('navbar-personalize');
    if (personalizeContainer) {
      const root = createRoot(personalizeContainer);
      root.render(<Personalize />);
    }

    // Render LanguageToggle component
    const languageContainer = document.getElementById('navbar-language-toggle');
    if (languageContainer) {
      const root2 = createRoot(languageContainer);
      root2.render(<LanguageToggle />);
    }
  });
}