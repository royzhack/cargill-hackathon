// Voyage Optimization Chatbot Frontend

class ChatApp {
    constructor() {
        this.apiBase = '';
        this.messages = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkStatus();
    }

    setupEventListeners() {
        const input = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');
        const resetBtn = document.getElementById('resetBtn');

        // Auto-resize textarea
        input.addEventListener('input', () => {
            input.style.height = 'auto';
            input.style.height = input.scrollHeight + 'px';
            sendBtn.disabled = input.value.trim() === '';
        });

        // Send on Enter (Shift+Enter for new line)
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!sendBtn.disabled) {
                    this.sendMessage();
                }
            }
        });

        // Send button
        sendBtn.addEventListener('click', () => {
            this.sendMessage();
        });

        // Reset button
        resetBtn.addEventListener('click', () => {
            this.resetConversation();
        });
    }

    async checkStatus() {
        try {
            const response = await fetch(`${this.apiBase}/api/status`);
            const data = await response.json();
            
            const statusIndicator = document.getElementById('statusIndicator');
            const statusText = statusIndicator.querySelector('.status-text');
            const statusDot = statusIndicator.querySelector('.status-dot');
            
            if (data.initialized) {
                statusText.textContent = 'Ready';
                statusDot.style.backgroundColor = '#10a37f';
            } else {
                statusText.textContent = 'Not Initialized';
                statusDot.style.backgroundColor = '#ef4444';
            }
        } catch (error) {
            console.error('Status check failed:', error);
        }
    }

    async sendMessage() {
        const input = document.getElementById('messageInput');
        const message = input.value.trim();
        
        if (!message) return;

        // Clear input
        input.value = '';
        input.style.height = 'auto';
        document.getElementById('sendBtn').disabled = true;

        // Add user message
        this.addMessage('user', message);

        // Show loading
        const loadingId = this.addLoadingMessage();

        try {
            const response = await fetch(`${this.apiBase}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message }),
            });

            const data = await response.json();

            // Remove loading
            this.removeMessage(loadingId);

            if (data.error) {
                this.addErrorMessage(data.error);
            } else {
                // Add assistant response
                if (data.response) {
                    this.addMessage('assistant', data.response);
                } else {
                    this.addErrorMessage('No response received from assistant');
                }

                // Add visualizations if any
                if (data.visualizations && data.visualizations.length > 0) {
                    setTimeout(() => {
                        data.visualizations.forEach(viz => {
                            this.addVisualization(viz);
                        });
                    }, 100);
                }
            }
        } catch (error) {
            this.removeMessage(loadingId);
            this.addErrorMessage('Failed to send message. Please try again.');
            console.error('Error:', error);
        }
    }

    addMessage(role, content) {
        const messagesContainer = document.getElementById('chatMessages');
        
        // Remove welcome message if present
        const welcomeMsg = messagesContainer.querySelector('.welcome-message');
        if (welcomeMsg) {
            welcomeMsg.remove();
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `message message-${role}`;
        messageDiv.id = `msg-${Date.now()}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        if (role === 'user') {
            avatar.textContent = 'U';
        } else {
            avatar.innerHTML = '<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M8 0C3.58 0 0 3.58 0 8s3.58 8 8 8 8-3.58 8-8-3.58-8-8-8zm0 14c-3.31 0-6-2.69-6-6S4.69 2 8 2s6 2.69 6 6-2.69 6-6 6z" fill="currentColor"/></svg>';
        }

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Format content
        const formatted = this.formatMessage(content);
        contentDiv.innerHTML = formatted;
        
        // Ensure proper spacing
        if (!contentDiv.querySelector('p, h1, h2, h3, ul, ol, pre, table')) {
            const text = contentDiv.textContent || contentDiv.innerText;
            if (text.trim()) {
                contentDiv.innerHTML = `<p>${formatted}</p>`;
            }
        }

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);
        messagesContainer.appendChild(messageDiv);

        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        return messageDiv.id;
    }

    formatMessage(content) {
        if (!content) return '';
        
        const escapeHtml = (text) => {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        };
        
        const codeBlockRegex = /```(\w+)?\n?([\s\S]*?)```/g;
        const parts = [];
        let lastIndex = 0;
        let match;
        
        while ((match = codeBlockRegex.exec(content)) !== null) {
            if (match.index > lastIndex) {
                parts.push({ type: 'text', content: content.substring(lastIndex, match.index) });
            }
            parts.push({ type: 'code', lang: match[1] || '', content: match[2].trim() });
            lastIndex = codeBlockRegex.lastIndex;
        }
        
        if (lastIndex < content.length) {
            parts.push({ type: 'text', content: content.substring(lastIndex) });
        }
        
        if (parts.length === 0) {
            parts.push({ type: 'text', content: content });
        }
        
        let formatted = '';
        
        for (const part of parts) {
            if (part.type === 'code') {
                formatted += `<pre><code>${escapeHtml(part.content)}</code></pre>`;
            } else {
                let text = escapeHtml(part.content);
                
                text = text.replace(/^### (.*$)/gim, '<h3>$1</h3>');
                text = text.replace(/^## (.*$)/gim, '<h2>$1</h2>');
                text = text.replace(/^# (.*$)/gim, '<h1>$1</h1>');
                
                text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                text = text.replace(/\*(?!\*)([^*]+?)\*/g, '<em>$1</em>');
                text = text.replace(/`([^`\n]+)`/g, '<code>$1</code>');
                
                const lines = text.split('\n');
                const processedLines = [];
                let inList = false;
                let listItems = [];
                
                for (const line of lines) {
                    const trimmed = line.trim();
                    const listMatch = trimmed.match(/^[-*•]\s+(.+)$/) || trimmed.match(/^\d+\.\s+(.+)$/);
                    
                    if (listMatch) {
                        if (!inList) {
                            inList = true;
                            if (listItems.length > 0) {
                                processedLines.push(listItems.join(''));
                                listItems = [];
                            }
                        }
                        listItems.push(`<li>${listMatch[1]}</li>`);
                    } else {
                        if (inList) {
                            processedLines.push(`<ul>${listItems.join('')}</ul>`);
                            listItems = [];
                            inList = false;
                        }
                        if (trimmed) {
                            processedLines.push(trimmed);
                        } else if (processedLines.length > 0) {
                            processedLines.push('');
                        }
                    }
                }
                
                if (inList && listItems.length > 0) {
                    processedLines.push(`<ul>${listItems.join('')}</ul>`);
                }
                
                text = processedLines.join('\n');
                
                const paragraphs = text.split(/\n\n+/);
                text = paragraphs.map(p => {
                    p = p.trim();
                    if (!p) return '';
                    if (/^<(h[1-6]|ul|ol|pre)/.test(p)) {
                        return p;
                    }
                    if (!p.replace(/<[^>]*>/g, '').trim()) {
                        return p;
                    }
                    return `<p>${p}</p>`;
                }).join('');
                
                text = text.replace(/(<p>.*?<\/p>)/gs, (match) => {
                    return match.replace(/\n/g, '<br>');
                });
                
                formatted += text;
            }
        }
        
        formatted = formatted.replace(/(<br>\s*){3,}/g, '<br><br>');
        formatted = formatted.replace(/<p><\/p>/g, '');
        
        return formatted;
    }

    addLoadingMessage() {
        const messagesContainer = document.getElementById('chatMessages');
        const welcomeMsg = messagesContainer.querySelector('.welcome-message');
        if (welcomeMsg) {
            welcomeMsg.remove();
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = 'message message-assistant';
        messageDiv.id = `loading-${Date.now()}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = '<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M8 0C3.58 0 0 3.58 0 8s3.58 8 8 8 8-3.58 8-8-3.58-8-8-8zm0 14c-3.31 0-6-2.69-6-6S4.69 2 8 2s6 2.69 6 6-2.69 6-6 6z" fill="currentColor"/></svg>';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'loading-indicator';
        loadingDiv.innerHTML = '<div class="loading-dot"></div><div class="loading-dot"></div><div class="loading-dot"></div>';
        
        const typingText = document.createElement('div');
        typingText.className = 'typing-indicator';
        typingText.textContent = 'Thinking...';
        
        contentDiv.appendChild(loadingDiv);
        contentDiv.appendChild(typingText);
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);
        messagesContainer.appendChild(messageDiv);

        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        return messageDiv.id;
    }

    removeMessage(messageId) {
        const message = document.getElementById(messageId);
        if (message) {
            message.remove();
        }
    }

    addErrorMessage(message) {
        const messagesContainer = document.getElementById('chatMessages');
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        messagesContainer.appendChild(errorDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    addVisualization(viz) {
        const messagesContainer = document.getElementById('chatMessages');

        const vizContainer = document.createElement('div');
        vizContainer.className = 'visualization-container';
        vizContainer.style.marginTop = '20px';

        const title = document.createElement('div');
        title.className = 'visualization-title';
        title.textContent = viz.title;

        // Handle HTML maps differently from images
        if (viz.is_html) {
            const mapContainer = document.createElement('div');
            mapContainer.className = 'map-container';
            mapContainer.style.width = '100%';
            mapContainer.style.height = '600px';
            mapContainer.style.border = '1px solid var(--claude-border)';
            mapContainer.style.borderRadius = '8px';
            mapContainer.style.overflow = 'hidden';
            mapContainer.style.background = '#f7f7f8';
            
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'visualization-loading';
            loadingDiv.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; gap: 8px; height: 600px;"><div class="loading-dot"></div><div class="loading-dot"></div><div class="loading-dot"></div><span style="margin-left: 8px;">Loading map...</span></div>';
            
            mapContainer.appendChild(loadingDiv);
            
            // Load map in iframe
            const iframe = document.createElement('iframe');
            iframe.src = `${this.apiBase}${viz.url}?t=${Date.now()}`;
            iframe.style.width = '100%';
            iframe.style.height = '100%';
            iframe.style.border = 'none';
            iframe.style.display = 'none';
            
            iframe.onload = () => {
                loadingDiv.remove();
                iframe.style.display = 'block';
                setTimeout(() => {
                    mapContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }, 100);
            };
            
            iframe.onerror = () => {
                loadingDiv.innerHTML = '<div style="color: #ef4444; padding: 20px; text-align: center;">Failed to load map. Please try refreshing.</div>';
            };
            
            mapContainer.appendChild(iframe);
            vizContainer.appendChild(title);
            vizContainer.appendChild(mapContainer);
        } else {
            // Regular image visualization
            const imageContainer = document.createElement('div');
            imageContainer.className = 'visualization-loading';
            imageContainer.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; gap: 8px;"><div class="loading-dot"></div><div class="loading-dot"></div><div class="loading-dot"></div><span style="margin-left: 8px;">Loading visualization...</span></div>';

            const img = document.createElement('img');
            img.className = 'visualization-image';
            img.src = `${this.apiBase}${viz.url}?t=${Date.now()}`;
            img.alt = viz.title;
            img.style.display = 'none';
            
            img.onload = () => {
                imageContainer.replaceWith(img);
                img.style.display = 'block';
                setTimeout(() => {
                    img.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }, 100);
            };
            
            img.onerror = () => {
                imageContainer.innerHTML = '<div style="color: #ef4444; padding: 20px;">Failed to load visualization. Please try refreshing.</div>';
            };

            vizContainer.appendChild(title);
            vizContainer.appendChild(imageContainer);
            vizContainer.appendChild(img);
        }

        const lastMessage = messagesContainer.querySelector('.message-assistant:last-child');
        if (lastMessage) {
            const content = lastMessage.querySelector('.message-content');
            if (content) {
                content.appendChild(vizContainer);
            }
        } else {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message message-assistant';
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.innerHTML = '<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M8 0C3.58 0 0 3.58 0 8s3.58 8 8 8 8-3.58 8-8-3.58-8-8-8zm0 14c-3.31 0-6-2.69-6-6S4.69 2 8 2s6 2.69 6 6-2.69 6-6 6z" fill="currentColor"/></svg>';
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.appendChild(vizContainer);
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(contentDiv);
            messagesContainer.appendChild(messageDiv);
        }

        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    async resetConversation() {
        if (!confirm('Start a new conversation? This will clear the current chat history.')) {
            return;
        }

        try {
            const response = await fetch(`${this.apiBase}/api/reset`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (response.ok) {
                const messagesContainer = document.getElementById('chatMessages');
                messagesContainer.innerHTML = `
                    <div class="welcome-message">
                        <h3>Welcome to Voyage Optimization Assistant</h3>
                        <p>I can help you understand:</p>
                        <ul>
                            <li>Optimal vessel-cargo assignments</li>
                            <li>Profit and TCE analysis</li>
                            <li>Risk-adjusted voyage economics</li>
                            <li>Scenario analysis and thresholds</li>
                            <li>Trade-offs between alternatives</li>
                        </ul>
                        <p class="example-prompts">Try asking: "What is the best voyage for PACIFIC GLORY?" or "Compare the top 3 most profitable voyages"</p>
                    </div>
                `;
                this.messages = [];
            }
        } catch (error) {
            console.error('Reset error:', error);
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.chatApp = new ChatApp();
});
