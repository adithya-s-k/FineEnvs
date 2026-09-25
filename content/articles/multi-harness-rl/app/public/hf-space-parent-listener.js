/**
 * Script for Hugging Face Spaces parent window
 * This script listens to iframe messages and updates the parent window URL
 * 
 * Usage instructions:
 * 1. Add this script to your Hugging Face Space in app.py or in a Gradio component
 * 2. Or use it in an HTML page that contains your iframe
 */

(function () {
    'use strict';

    // Listen to iframe messages
    window.addEventListener('message', function (event) {

        // Check message type
        if (event.data && event.data.type) {
            switch (event.data.type) {
                case 'urlChange':
                case 'anchorChange':
                case 'HF_SPACE_URL_UPDATE':
                    handleUrlChange(event.data);
                    break;
                default:
                // Unknown message type, ignore
            }
        }
    });

    function handleUrlChange(data) {
        try {
            const hash = data.hash || data.anchorId;

            if (hash) {
                // Update URL with new anchor
                const newUrl = new URL(window.location);
                newUrl.hash = hash;

                // Use replaceState to avoid adding an entry to history
                window.history.replaceState(null, '', newUrl.toString());
            }
        } catch (error) {
            // Silent error when updating URL
        }
    }

    // Utility function to test communication
    window.testIframeCommunication = function () {
        const iframe = document.querySelector('iframe');
        if (iframe) {
            iframe.contentWindow.postMessage({ type: 'test' }, '*');
        }
    };

})();
