import os
from flask import Flask, render_template_string, request, jsonify
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not found in the environment. Please set it in your .env file")

# Initialize the Groq client
client = Groq(api_key=GROQ_API_KEY)

app = Flask(__name__)

# HTML template with inline CSS and JS for chatbot UI (including markdown renderer and code styling)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Simple Chatbot</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter&family=Material+Icons" rel="stylesheet" />
  <style>
    /* Reset and base */
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      font-family: 'Inter', sans-serif;
      background: linear-gradient(135deg, #1e293b, #0f172a);
      color: #e0e7ff;
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
      padding: 16px;
    }
    #app {
      display: flex;
      flex-direction: column;
      max-width: 900px; /* Increased width from 720px */
      width: 100%;
      height: 90vh;
      background: rgba(15, 23, 42, 0.8);
      border-radius: 16px;
      backdrop-filter: blur(10px);
      box-shadow: 0 8px 32px rgba(0,0,0,0.6);
      overflow: hidden;
    }
    header {
      padding: 16px 24px;
      border-bottom: 1px solid rgba(255,255,255,0.1);
      font-size: 1.5rem;
      font-weight: 700;
      color: #7dd3fc;
      text-align: center;
      user-select: none;
    }
    main {
      flex: 1;
      overflow-y: auto;
      padding: 16px 24px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .message {
      max-width: 85%; /* Increased from 75% */
      padding: 12px 16px;
      border-radius: 16px;
      line-height: 1.4;
      font-size: 1rem;
      word-wrap: break-word;
      display: inline-block;
      white-space: pre-wrap;
      user-select: text;
    }
    .user-message {
      align-self: flex-end;
      background: linear-gradient(135deg, #0284c7, #0369a1);
      color: white;
      border-bottom-right-radius: 4px;
    }
    .bot-message {
      align-self: flex-start;
      background: linear-gradient(135deg, #0ea5e9, #0284c7);
      color: #f0f9ff;
      border-bottom-left-radius: 4px;
      position: relative;
      /* For code block scroll */
      overflow-x: auto;
    }
    .timestamp {
      font-size: 0.7rem;
      opacity: 0.6;
      margin-top: 4px;
      user-select: none;
    }
    form {
      display: flex;
      border-top: 1px solid rgba(255,255,255,0.1);
      padding: 8px 16px;
      background: rgba(255,255,255,0.05);
      backdrop-filter: blur(8px);
    }
    input[type="text"] {
      flex: 1;
      border: none;
      background: transparent;
      padding: 12px;
      font-size: 1rem;
      color: #e0e7ff;
      outline-offset: 4px;
      border-radius: 12px;
    }
    input[type="text"]::placeholder {
      color: #94a3b8;
    }
    button {
      background: #0284c7;
      border: none;
      color: white;
      font-weight: 700;
      padding: 0 16px;
      margin-left: 12px;
      border-radius: 12px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.4rem;
      transition: background-color 0.3s ease;
    }
    button:hover {
      background: #0369a1;
    }
    button:disabled {
      background: #94a3b8;
      cursor: default;
    }
    /* Scrollbar styling */
    main::-webkit-scrollbar {
      width: 8px;
    }
    main::-webkit-scrollbar-thumb {
      background: rgba(66, 153, 225, 0.6);
      border-radius: 12px;
    }
    main::-webkit-scrollbar-track {
      background: transparent;
    }

    /* Code blocks styling - no syntax highlight, just plain monospace */
    pre {
      background-color: rgba(0,0,0,0.3);
      padding: 12px;
      border-radius: 12px;
      overflow-x: auto;
      font-family: monospace, Consolas, 'Courier New', monospace;
      color: inherit; /* No colored syntax highlight */
      margin: 1em 0;
      white-space: pre-wrap;
      word-break: break-word;
    }
    code {
      background-color: rgba(0,0,0,0.3);
      padding: 2px 6px;
      border-radius: 6px;
      font-family: monospace, Consolas, 'Courier New', monospace;
      color: inherit;
      white-space: pre-wrap;
      word-break: break-word;
    }
    /* Styling for bold */
    strong {
      font-weight: 700;
      color: #a5f3fc;
    }
    /* Styling for italic */
    em {
      font-style: italic;
      color: #bae6fd;
    }

    @media (max-width: 640px) {
      #app {
        border-radius: 0;
        height: 100vh;
      }
      header {
        font-size: 1.25rem;
        padding: 12px;
      }
      form {
        padding: 8px 12px;
      }
      input[type="text"] {
        padding: 10px;
      }
      button {
        padding: 0 12px;
        font-size: 1.2rem;
      }
    }
  </style>
  <!-- Load marked.js from CDN for markdown parsing -->
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
  <div id="app" role="main" aria-label="Chatbot application">
    <header>Simple Chatbot</header>
    <main id="chat-window" aria-live="polite" aria-relevant="additions"></main>
    <form id="chat-form" aria-label="Send message form">
      <input type="text" id="message-input" autocomplete="off" placeholder="Ask me anything..." aria-label="Message input" required />
      <button type="submit" aria-label="Send message">
        <span class="material-icons">send</span>
      </button>
    </form>
  </div>
  <script>
    const chatWindow = document.getElementById('chat-window');
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');

    // Helper: Format timestamp as HH:mm
    function formatTime() {
      const now = new Date();
      return now.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
    }

    // Escape HTML to safely display user messages as text (avoid injection)
    function escapeHtml(unsafe) {
      return unsafe.replace(/[&<>"']/g, function(m) {
        switch (m) {
          case '&': return '&amp;';
          case '<': return '&lt;';
          case '>': return '&gt;';
          case '"': return '&quot;';
          case "'": return '&#039;';
          default: return m;
        }
      });
    }

    // Add a message bubble to chat
    // sender: 'user' or 'bot'
    // For bot messages, render markdown to HTML using marked.js
    function addMessage(text, sender = 'bot') {
      const messageDiv = document.createElement('div');
      messageDiv.className = 'message ' + (sender === 'user' ? 'user-message' : 'bot-message');

      if (sender === 'user') {
        // Escape user input to prevent HTML render/injection
        messageDiv.textContent = text;
      } else {
        // Render bot response markdown as HTML
        messageDiv.innerHTML = marked.parse(text);
      }

      const timestampDiv = document.createElement('div');
      timestampDiv.className = 'timestamp';
      timestampDiv.textContent = formatTime();

      messageDiv.appendChild(timestampDiv);
      chatWindow.appendChild(messageDiv);
      chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    // Disable input while waiting
    function toggleInput(disabled) {
      messageInput.disabled = disabled;
      chatForm.querySelector('button').disabled = disabled;
    }

    chatForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const userMessage = messageInput.value.trim();
      if (!userMessage) return;

      addMessage(userMessage, 'user');
      toggleInput(true);
      messageInput.value = '';

      try {
        const response = await fetch('/chat', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({message: userMessage})
        });
        if (!response.ok) throw new Error('Network response was not ok');

        const data = await response.json();
        addMessage(data.reply, 'bot');
      } catch (err) {
        addMessage('Sorry, there was an error processing your request.', 'bot');
        console.error(err);
      } finally {
        toggleInput(false);
        messageInput.focus();
      }
    });

    // Focus input on load
    window.onload = () => {
      messageInput.focus();
    };
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_msg = data.get('message', '').strip()
    if not user_msg:
        return jsonify(reply="Please send a valid message.")

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": user_msg}
            ],
            model="llama-3.1-8b-instant"
        )
        reply_text = chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        reply_text = "Sorry, I'm having trouble generating a response right now."

    return jsonify(reply=reply_text)


if __name__ == '__main__':
    app.run(debug=True)


