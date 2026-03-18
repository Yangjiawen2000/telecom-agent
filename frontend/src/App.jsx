import React, { useState, useEffect, useRef } from 'react';

const App = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [status, setStatus] = useState('');
  const [anchors, setAnchors] = useState([]);
  const [sessionId, setSessionId] = useState(
    localStorage.getItem('telecom_session_id') || `session_${Math.random().toString(36).substr(2, 9)}`
  );

  const scrollRef = useRef(null);

  useEffect(() => {
    localStorage.setItem('telecom_session_id', sessionId);
    fetchHistory();
    fetchAnchors();
  }, [sessionId]);

  const fetchHistory = async () => {
    try {
      const response = await fetch(`http://localhost:8000/chat/history/${sessionId}`);
      const data = await response.json();
      if (data.history) {
        setMessages(data.history.map(m => ({
          role: m.role,
          content: m.content,
          intent: m.metadata?.intent || '',
          type: 'text'
        })));
      }
    } catch (err) {
      console.error("Failed to fetch history:", err);
    }
  };

  const fetchAnchors = async () => {
    try {
      const response = await fetch(`http://localhost:8000/chat/anchors/${sessionId}`);
      const data = await response.json();
      if (data.anchors) {
        setAnchors(data.anchors);
      }
    } catch (err) {
      console.error("Failed to fetch anchors:", err);
    }
  };

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, status]);

  const sendMessage = async (textOverride) => {
    const messageText = textOverride || input;
    if (!messageText.trim()) return;

    const userMessage = { role: 'user', content: messageText };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setStatus('正在加载上下文与识别意图...');

    try {
      const response = await fetch('http://localhost:8000/chat/message', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          user_id: 'user_01', // 假定用户 ID
          message: messageText,
        }),
      });

      if (!response.ok) throw new Error('网络请求失败');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let aiMessage = { role: 'assistant', content: '', intent: '', type: 'text' };
      
      setMessages(prev => [...prev, aiMessage]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));
            
            if (data.type === 'thinking') {
              setStatus(data.content);
            } else if (data.type === 'token') {
              aiMessage.content += data.content;
              setStatus('生成回复中...');
              setMessages(prev => {
                const newMsgs = [...prev];
                newMsgs[newMsgs.length - 1] = { ...aiMessage };
                return newMsgs;
              });
            } else if (data.type === 'done') {
              aiMessage.intent = data.intent;
              setMessages(prev => {
                const newMsgs = [...prev];
                newMsgs[newMsgs.length - 1] = { ...aiMessage };
                return newMsgs;
              });
              setStatus('');
              // Update anchors
              fetchAnchors();
            } else if (data.type === 'card') {
              // Mocking card behavior if backend returns it
              aiMessage.type = 'card';
              aiMessage.cardData = data.content;
              setMessages(prev => {
                const newMsgs = [...prev];
                newMsgs[newMsgs.length - 1] = { ...aiMessage };
                return newMsgs;
              });
            } else if (data.type === 'error') {
               setStatus(`错误: ${data.content}`);
            }
          }
        }
      }
    } catch (err) {
      setStatus(`连接异常: ${err.message}`);
    }
  };

  const startNewChat = async () => {
    try {
      // 1. 调用后端删除接口
      await fetch(`http://localhost:8000/chat/session/${sessionId}?user_id=user_01`, {
        method: 'DELETE'
      });
      
      // 2. 生成并设置新的 Session ID
      const newSessionId = `session_${Math.random().toString(36).substr(2, 9)}`;
      setSessionId(newSessionId);
      
      // 3. 重置前端状态
      setMessages([]);
      setAnchors([]);
      setStatus('新会话已开启');
      setTimeout(() => setStatus(''), 2000);
    } catch (err) {
      console.error("Failed to start new chat:", err);
      setStatus("开启新会话失败");
    }
  };

  return (
    <div className="flex h-screen bg-gray-100 text-gray-900 overflow-hidden font-sans">
      {/* Sidebar - Context Panel */}
      <div className="w-64 bg-slate-800 text-white flex flex-col p-4 shadow-xl border-r border-slate-700">
        <h2 className="text-xl font-bold mb-6 flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></span>
            上下文记忆
          </div>
          <button 
            onClick={startNewChat}
            title="开启新对话"
            className="p-1.5 hover:bg-slate-700 rounded-lg transition-colors text-slate-400 hover:text-white"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4"></path>
            </svg>
          </button>
        </h2>
        <div className="flex-1 space-y-3 overflow-y-auto">
          {anchors.map((anchor, i) => (
            <div key={i} className="bg-slate-700/50 p-3 rounded-lg border border-slate-600 hover:bg-slate-700 transition-all cursor-default text-sm">
              <span className="text-blue-400 font-mono mr-2">#</span> {anchor}
            </div>
          ))}
          {anchors.length === 0 && (
            <div className="text-slate-400 text-sm italic">暂无记忆锚点</div>
          )}
        </div>
        <div className="mt-4 pt-4 border-t border-slate-700">
           <p className="text-xs text-slate-500 mb-1">Session ID:</p>
           <p className="text-[10px] font-mono break-all opacity-60">{sessionId}</p>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col relative max-w-4xl mx-auto w-full shadow-2xl bg-white">
        {/* Header */}
        <div className="p-4 border-b flex justify-between items-center bg-white/80 backdrop-blur-md sticky top-0 z-10">
          <h1 className="text-lg font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
            电信业务全能助手
          </h1>
          {status && (
            <div className="flex items-center gap-2 text-blue-500 text-sm font-medium animate-fadeIn">
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
              {status}
            </div>
          )}
        </div>

        {/* Messages */}
        <div ref={scrollRef} className="flex-1 p-6 overflow-y-auto space-y-6 scroll-smooth">
          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-slideInUp`}>
              <div className={`max-w-[80%] flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                {msg.intent && (
                  <span className="text-[10px] font-bold uppercase tracking-wider text-blue-500 bg-blue-50 px-2 py-0.5 rounded-full mb-1 border border-blue-100 shadow-sm">
                    {msg.intent}
                  </span>
                )}
                <div className={`p-4 rounded-2xl shadow-sm border ${
                  msg.role === 'user' 
                    ? 'bg-gradient-to-br from-blue-600 to-indigo-700 text-white rounded-tr-none border-blue-500' 
                    : 'bg-white text-gray-800 rounded-tl-none border-gray-100'
                }`}>
                  <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                  
                  {msg.type === 'card' && msg.cardData && (
                    <div className="mt-4 grid grid-cols-1 gap-3">
                      {msg.cardData.map((choice, idx) => (
                        <button 
                          key={idx} 
                          onClick={() => sendMessage(choice)}
                          className="bg-blue-50 border border-blue-100 p-3 rounded-xl hover:bg-blue-100 transition-all text-left text-sm font-medium text-blue-700 flex justify-between items-center group"
                        >
                          {choice}
                          <span className="opacity-0 group-hover:opacity-100 transition-opacity">点击办理 →</span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
          {messages.length === 0 && (
             <div className="h-full flex flex-col items-center justify-center text-gray-300 space-y-4">
                <div className="w-20 h-20 bg-gray-50 rounded-full flex items-center justify-center border-4 border-dashed border-gray-100">
                   <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"></path></svg>
                </div>
                <p className="font-medium">您好！我是您的智慧电信助手，请问有什么可以帮您？</p>
                <div className="flex gap-2">
                   <button onClick={() => sendMessage('推荐个套餐')} className="px-3 py-1 bg-white border border-gray-200 rounded-full text-xs text-gray-500 hover:border-blue-400 hover:text-blue-500 transition-all">套餐推荐</button>
                   <button onClick={() => sendMessage('我想办宽带')} className="px-3 py-1 bg-white border border-gray-200 rounded-full text-xs text-gray-500 hover:border-blue-400 hover:text-blue-500 transition-all">办理业务</button>
                   <button onClick={() => sendMessage('查询上月账单')} className="px-3 py-1 bg-white border border-gray-200 rounded-full text-xs text-gray-500 hover:border-blue-400 hover:text-blue-500 transition-all">查账单</button>
                </div>
             </div>
          )}
        </div>

        {/* Input */}
        <div className="p-4 border-t bg-gray-50/50 backdrop-blur-sm">
          <div className="flex gap-3 bg-white p-2 rounded-2xl shadow-inner border focus-within:ring-2 focus-within:ring-blue-500/20 transition-all">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
              placeholder="请输入您的问题..."
              className="flex-1 px-4 py-2 border-none focus:outline-none bg-transparent"
            />
            <button
              onClick={() => sendMessage()}
              className="px-6 py-2 bg-blue-600 text-white rounded-xl font-bold hover:bg-blue-700 transition-all active:scale-95 shadow-lg shadow-blue-500/30"
            >
              发送
            </button>
          </div>
          <p className="text-[10px] text-center text-gray-400 mt-2">
             输入“推荐办卡并查账单”体验复合意图处理
          </p>
        </div>
      </div>
      
      {/* CSS Animations */}
      <style dangerouslySetInnerHTML={{ __html: `
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes slideInUp { from { transform: translateY(20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
        .animate-fadeIn { animation: fadeIn 0.3s ease-out; }
        .animate-slideInUp { animation: slideInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1); }
      `}} />
    </div>
  );
};

export default App;
