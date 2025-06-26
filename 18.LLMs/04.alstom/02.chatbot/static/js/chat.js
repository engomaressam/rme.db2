document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.querySelector('.chat-messages');
    const chatForm = document.querySelector('.chat-form');
    const chatInput = document.querySelector('.chat-input input');
    const sendButton = document.querySelector('.chat-input button');

    // Add initial welcome message
    addBotMessage("Welcome to the Alstom Project Assistant! I can answer questions about the project documents. How can I help you today?");

    // Handle form submission
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const message = chatInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        addUserMessage(message);
        
        // Clear input
        chatInput.value = '';
        
        // Show loading indicator
        showLoading();
        
        // Disable send button during processing
        sendButton.disabled = true;
        
        // Send message to server
        fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: message })
        })
        .then(response => response.json())
        .then(data => {
            // Remove loading indicator
            removeLoading();
            
            // Add bot response to chat
            addBotMessage(data.answer, data.sources);
            
            // Re-enable send button
            sendButton.disabled = false;
            
            // Scroll to bottom
            scrollToBottom();
        })
        .catch(error => {
            console.error('Error:', error);
            removeLoading();
            addBotMessage("I'm sorry, there was an error processing your request. Please try again.");
            sendButton.disabled = false;
        });
        
        // Scroll to bottom
        scrollToBottom();
    });

    // Function to add user message to chat
    function addUserMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', 'message-user');
        messageElement.textContent = message;
        chatMessages.appendChild(messageElement);
    }

    // Function to add bot message to chat
    function addBotMessage(message, sources = []) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', 'message-bot');
        messageElement.textContent = message;
        chatMessages.appendChild(messageElement);
        
        // Add sources if available
        if (sources && sources.length > 0) {
            const sourcesElement = document.createElement('div');
            sourcesElement.classList.add('message-sources');
            
            const sourcesText = document.createElement('span');
            sourcesText.textContent = 'Sources:';
            sourcesElement.appendChild(sourcesText);
            
            const sourcesList = document.createElement('ul');
            sources.forEach(source => {
                const sourceItem = document.createElement('li');
                sourceItem.textContent = source;
                sourcesList.appendChild(sourceItem);
            });
            
            sourcesElement.appendChild(sourcesList);
            chatMessages.appendChild(sourcesElement);
        }
    }

    // Function to show loading indicator
    function showLoading() {
        const loadingElement = document.createElement('div');
        loadingElement.classList.add('loading');
        
        const dotsContainer = document.createElement('div');
        dotsContainer.classList.add('loading-dots');
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            dotsContainer.appendChild(dot);
        }
        
        loadingElement.appendChild(dotsContainer);
        chatMessages.appendChild(loadingElement);
        
        scrollToBottom();
    }

    // Function to remove loading indicator
    function removeLoading() {
        const loadingElement = document.querySelector('.loading');
        if (loadingElement) {
            loadingElement.remove();
        }
    }

    // Function to scroll chat to bottom
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Focus input on page load
    chatInput.focus();
});
