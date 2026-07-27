import { useState, useRef, useEffect } from 'react'
import axios from 'axios'

const TextGenerator = ({ apiUrl, status }) => {
  const savedPrompt = localStorage.getItem('shannon_prompt') || 'The '
  const savedSystemPrompt = localStorage.getItem('shannon_system_prompt') || ''
  const savedMaxTokens = parseInt(localStorage.getItem('shannon_max_tokens')) || 100
  const savedTemperature = parseFloat(localStorage.getItem('shannon_temperature')) || 0.85
  const savedTopK = parseInt(localStorage.getItem('shannon_top_k')) || 40
  const savedRepetitionPenalty = parseFloat(localStorage.getItem('shannon_repetition_penalty')) || 1.15
  const savedUseStreaming = localStorage.getItem('shannon_use_streaming') !== 'false'
  const savedMode = localStorage.getItem('shannon_mode') || 'chat'

  // Single generate mode
  const [prompt, setPrompt] = useState(savedPrompt)
  const [systemPrompt, setSystemPrompt] = useState(savedSystemPrompt)
  const [maxTokens, setMaxTokens] = useState(savedMaxTokens)
  const [temperature, setTemperature] = useState(savedTemperature)
  const [topK, setTopK] = useState(savedTopK)
  const [repetitionPenalty, setRepetitionPenalty] = useState(savedRepetitionPenalty)
  const [generated, setGenerated] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [useStreaming, setUseStreaming] = useState(savedUseStreaming)
  const [generationStats, setGenerationStats] = useState(null)
  const [mode, setMode] = useState(savedMode)
  const abortControllerRef = useRef(null)
  const generatedTextRef = useRef('')

  // Chat mode
  const [convId, setConvId] = useState(null)
  const [convTemplate, setConvTemplate] = useState('simple')
  const [messages, setMessages] = useState([])
  const [chatInput, setChatInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  const [chatStreamText, setChatStreamText] = useState('')
  const chatAbortRef = useRef(null)
  const messagesEndRef = useRef(null)
  const chatInputRef = useRef(null)
  const streamTextRef = useRef('')

  // Persist to localStorage
  useEffect(() => { localStorage.setItem('shannon_prompt', prompt) }, [prompt])
  useEffect(() => { localStorage.setItem('shannon_system_prompt', systemPrompt) }, [systemPrompt])
  useEffect(() => { localStorage.setItem('shannon_max_tokens', maxTokens.toString()) }, [maxTokens])
  useEffect(() => { localStorage.setItem('shannon_temperature', temperature.toString()) }, [temperature])
  useEffect(() => { localStorage.setItem('shannon_top_k', topK.toString()) }, [topK])
  useEffect(() => { localStorage.setItem('shannon_repetition_penalty', repetitionPenalty.toString()) }, [repetitionPenalty])
  useEffect(() => { localStorage.setItem('shannon_use_streaming', useStreaming.toString()) }, [useStreaming])
  useEffect(() => { localStorage.setItem('shannon_mode', mode) }, [mode])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, chatStreamText])

  useEffect(() => { setError('') }, [mode])

  // Chat: create conversation
  const createConversation = async () => {
    if (!status.model_loaded) { setError('请先加载模型'); return }
    setError('')
    try {
      const res = await axios.post(`${apiUrl}/conv/create`, null, {
        params: { system_prompt: systemPrompt || undefined, template: convTemplate, max_context: 4096 }
      })
      setConvId(res.data.conversation_id)
      setMessages([])
      setChatStreamText('')
      streamTextRef.current = ''
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    }
  }

  // Chat: clear history
  const clearConversation = async () => {
    if (!convId) return
    try { await axios.post(`${apiUrl}/conv/${convId}/clear`, null, { params: { keep_system: true } }) } catch (_) {}
    setMessages([])
    setChatStreamText('')
    streamTextRef.current = ''
  }

  // Chat: delete conversation
  const deleteConversation = async () => {
    if (!convId) return
    try { await axios.delete(`${apiUrl}/conv/${convId}`) } catch (_) {}
    setConvId(null)
    setMessages([])
    setChatStreamText('')
    streamTextRef.current = ''
  }

  // Chat: send message (streaming SSE)
  const sendChatMessage = async () => {
    const text = chatInput.trim()
    if (!text || !status.model_loaded) return

    setError('')
    setChatInput('')

    let currentConvId = convId
    if (!currentConvId) {
      try {
        const res = await axios.post(`${apiUrl}/conv/create`, null, {
          params: { system_prompt: systemPrompt || undefined, template: convTemplate, max_context: 4096 }
        })
        currentConvId = res.data.conversation_id
        setConvId(currentConvId)
      } catch (err) {
        setError(err.response?.data?.detail || err.message)
        return
      }
    }

    setMessages(prev => [...prev, { role: 'user', content: text, timestamp: new Date().toISOString() }])
    setChatLoading(true)
    setChatStreamText('')
    streamTextRef.current = ''

    chatAbortRef.current = new AbortController()
    const signal = chatAbortRef.current.signal

    try {
      const response = await fetch(`${apiUrl}/conv/generate/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: text, conversation_id: currentConvId,
          max_tokens: maxTokens, temperature, top_k: topK, top_p: 0.9, repetition_penalty: repetitionPenalty,
        }),
        signal,
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)

      const reader = response.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || ''
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const data = JSON.parse(line.slice(6))
            if (data.type === 'error') { setError(data.error); setChatLoading(false); break }
            if (data.type === 'complete') {
              const finalReply = data.assistant_reply || streamTextRef.current || ''
              setMessages(prev => {
                const last = prev[prev.length - 1]
                if (last && last.role === 'assistant' && last.content === finalReply) return prev
                return [...prev, { role: 'assistant', content: finalReply, timestamp: new Date().toISOString() }]
              })
              setChatStreamText('')
              streamTextRef.current = ''
              setChatLoading(false)
              break
            }
            if (data.text !== undefined && !data.is_complete) {
              streamTextRef.current = data.text
              setChatStreamText(data.text)
            }
          } catch (_) {}
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') setError(err.message || '流式生成失败')
    } finally {
      setChatLoading(false)
      chatAbortRef.current = null
    }
  }

  // Chat: stop generation
  const stopChatGeneration = () => {
    chatAbortRef.current?.abort()
    if (streamTextRef.current) {
      setMessages(prev => [...prev, { role: 'assistant', content: streamTextRef.current + ' [中断]', timestamp: new Date().toISOString() }])
      setChatStreamText('')
      streamTextRef.current = ''
    }
    setChatLoading(false)
  }

  // Single: non-streaming
  const handleGenerate = async () => {
    if (!status.model_loaded) { setError('请先加载模型'); return }
    setLoading(true); setError(''); setGenerated(''); setGenerationStats(null)
    const startTime = Date.now()
    try {
      const res = await axios.post(`${apiUrl}/generate`, {
        prompt, system_prompt: systemPrompt || undefined,
        max_tokens: maxTokens, temperature, top_k: topK, repetition_penalty: repetitionPenalty
      })
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(2)
      const tokens = res.data.generated_text?.split(/\s+/)?.length || 0
      setGenerated(res.data.generated_text)
      setGenerationStats({ time: elapsed, tokens, speed: (tokens / parseFloat(elapsed)).toFixed(2) })
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally { setLoading(false) }
  }

  // Single: streaming SSE
  const handleStreamGenerate = async () => {
    if (!status.model_loaded) { setError('请先加载模型'); return }
    setLoading(true); setError(''); setGenerated(''); setGenerationStats(null)
    generatedTextRef.current = ''
    abortControllerRef.current = new AbortController()
    const signal = abortControllerRef.current.signal
    const startTime = Date.now()
    let tokenCount = 0
    try {
      const response = await fetch(`${apiUrl}/generate/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt, system_prompt: systemPrompt || undefined,
          max_tokens: maxTokens, temperature, top_k: topK, repetition_penalty: repetitionPenalty
        }),
        signal
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const reader = response.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || ''
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const data = JSON.parse(line.slice(6))
            if (data.type === 'error') { setError(data.error); setLoading(false); break }
            if (data.type === 'complete') {
              setGenerationStats({ time: ((Date.now() - startTime) / 1000).toFixed(2), tokens: tokenCount, speed: (tokenCount / ((Date.now() - startTime) / 1000)).toFixed(2) })
              setLoading(false); break
            }
            if (data.text !== undefined) {
              tokenCount++
              generatedTextRef.current = data.text
              setGenerated(data.text)
            }
          } catch (_) {}
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') setError(err.message || '流式生成失败')
    } finally { setLoading(false); abortControllerRef.current = null }
  }

  const handleStop = () => { abortControllerRef.current?.abort(); setLoading(false) }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-6 border border-gray-700 shadow-lg backdrop-blur-sm">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent flex items-center gap-2">
            <span className="text-3xl">✍️</span> 文本生成工作站
          </h2>
          <div className="flex items-center gap-3">
            {!status.model_loaded && (
              <div className="px-4 py-2 bg-yellow-500/20 border border-yellow-500/50 rounded-full">
                <p className="text-yellow-300 text-sm font-medium">⚠️ 未加载模型</p>
              </div>
            )}
            <div className="flex rounded-lg overflow-hidden border border-gray-600">
              <button onClick={() => setMode('single')} className={`px-3 py-1.5 text-sm font-medium transition-colors ${mode === 'single' ? 'bg-purple-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}>📝 单次生成</button>
              <button onClick={() => setMode('chat')} className={`px-3 py-1.5 text-sm font-medium transition-colors ${mode === 'chat' ? 'bg-purple-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}>💬 多轮对话</button>
            </div>
          </div>
        </div>

        {/* System prompt + template */}
        <div className="mb-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2 flex items-center gap-2">
              <span>🤖</span> 系统提示词 <span className="text-xs text-gray-500 ml-1">(可选)</span>
            </label>
            <textarea value={systemPrompt} onChange={e => setSystemPrompt(e.target.value)} className="w-full px-4 py-3 bg-gray-700/50 border border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent text-white placeholder-gray-500 transition-all resize-none" rows={2} placeholder="例如：你是一个专业的编程助手..." />
          </div>
          {mode === 'chat' && (
            <div className="flex items-center gap-4">
              <label className="text-sm font-medium text-gray-300">📋 模板:</label>
              <select value={convTemplate} onChange={e => setConvTemplate(e.target.value)} disabled={!!convId} className="px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white text-sm focus:ring-2 focus:ring-purple-500 disabled:opacity-50">
                <option value="simple">Simple [默认]</option>
                <option value="chatml">ChatML (OpenAI 风格)</option>
                <option value="llama3">Llama3</option>
              </select>
              {convId && <span className="text-xs text-gray-500">(创建后不可更改)</span>}
            </div>
          )}
        </div>

        {/* ========== SINGLE MODE ========== */}
        {mode === 'single' && (
          <>
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-300 mb-2"><span>📝</span> 用户提示词</label>
              <textarea value={prompt} onChange={e => setPrompt(e.target.value)} className="w-full px-4 py-3 bg-gray-700/50 border border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent text-white placeholder-gray-500 transition-all resize-none" rows={4} placeholder="输入你的创意提示词..." />
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              <div><label className="block text-xs text-gray-400 mb-1.5">最大 Token</label><input type="number" value={maxTokens} onChange={e => setMaxTokens(parseInt(e.target.value))} className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-purple-500" min={10} max={500} /></div>
              <div><label className="block text-xs text-gray-400 mb-1.5 flex justify-between"><span>温度</span><span className="text-purple-400 font-mono text-sm">{temperature.toFixed(2)}</span></label><input type="range" value={temperature} onChange={e => setTemperature(parseFloat(e.target.value))} className="w-full accent-purple-500 h-2" min={0.1} max={1.5} step={0.05} /></div>
              <div><label className="block text-xs text-gray-400 mb-1.5">Top-K</label><input type="number" value={topK} onChange={e => setTopK(parseInt(e.target.value))} className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-purple-500" min={1} max={100} /></div>
              <div><label className="block text-xs text-gray-400 mb-1.5 flex justify-between"><span>重复惩罚</span><span className="text-purple-400 font-mono text-sm">{repetitionPenalty.toFixed(2)}</span></label><input type="range" value={repetitionPenalty} onChange={e => setRepetitionPenalty(parseFloat(e.target.value))} className="w-full accent-purple-500 h-2" min={1.0} max={1.5} step={0.05} /></div>
            </div>
            <div className="mb-6 p-4 bg-gray-700/30 rounded-lg border border-gray-600/50">
              <label className="flex items-center space-x-3 cursor-pointer">
                <div className="relative"><input type="checkbox" checked={useStreaming} onChange={e => setUseStreaming(e.target.checked)} className="sr-only peer" /><div className="w-11 h-6 bg-gray-600 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-purple-500/50 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div></div>
                <div><span className="text-sm font-medium text-gray-300">启用流式输出</span><p className="text-xs text-gray-500 mt-0.5">{useStreaming ? '✨ 实时显示生成过程' : '⏳ 等待完成后一次性显示'}</p></div>
              </label>
            </div>
            <div className="flex space-x-3">
              <button onClick={useStreaming ? handleStreamGenerate : handleGenerate} disabled={loading || !status.model_loaded} className="flex-1 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-lg transition-all transform hover:scale-[1.02] active:scale-[0.98] shadow-lg">
                {loading ? (<span className="flex items-center justify-center gap-2"><svg className="animate-spin h-5 w-5" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>生成中...</span>) : (<span className="flex items-center justify-center gap-2">🚀 开始生成</span>)}
              </button>
              {loading && useStreaming && (<button onClick={handleStop} className="px-6 py-3 bg-gradient-to-r from-red-600 to-rose-600 text-white font-semibold rounded-lg">⏹ 停止</button>)}
            </div>
          </>
        )}

        {/* ========== CHAT MODE ========== */}
        {mode === 'chat' && (
          <>
            <div className="flex items-center justify-between mb-4 p-3 bg-gray-700/30 rounded-lg border border-gray-600/50">
              <span className="text-sm text-gray-300">
                {convId ? (<span className="flex items-center gap-2"><span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>会话: <code className="text-purple-400">{convId}</code></span>) : '未创建会话 — 发送消息自动创建'}
              </span>
              <div className="flex items-center gap-2">
                {convId && (<><button onClick={clearConversation} className="px-3 py-1.5 text-xs bg-gray-600 hover:bg-gray-500 text-gray-200 rounded-lg" title="清空对话历史">🗑️ 清空</button><button onClick={deleteConversation} className="px-3 py-1.5 text-xs bg-red-900/50 hover:bg-red-800/50 text-red-300 rounded-lg" title="删除会话">❌ 删除</button></>)}
                <button onClick={createConversation} disabled={!status.model_loaded || chatLoading} className="px-3 py-1.5 text-xs bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white rounded-lg">+ 新建</button>
              </div>
            </div>

            {/* Messages area */}
            <div className="mb-4 bg-gray-950/60 rounded-xl border border-gray-700 overflow-hidden">
              <div className="h-[400px] overflow-y-auto p-4 space-y-4">
                {messages.length === 0 && !chatStreamText && !chatLoading && (
                  <div className="flex items-center justify-center h-full text-gray-500 text-sm"><div className="text-center"><p className="text-3xl mb-2">💬</p><p>开始多轮对话</p><p className="text-xs mt-1">发送消息自动创建会话</p></div></div>
                )}

                {messages.map((msg, idx) => (
                  <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[80%] px-4 py-3 rounded-2xl ${msg.role === 'user' ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-br-md' : 'bg-gray-700 text-gray-200 rounded-bl-md'}`}>
                      <span className={`text-xs font-medium block mb-1 ${msg.role === 'user' ? 'text-pink-300' : 'text-purple-400'}`}>{msg.role === 'user' ? '🧑 你' : '🤖 助手'}</span>
                      <p className="whitespace-pre-wrap text-sm leading-relaxed">{msg.content}</p>
                    </div>
                  </div>
                ))}

                {/* Streaming bubble */}
                {(chatStreamText || chatLoading) && (
                  <div className="flex justify-start">
                    <div className="max-w-[80%] px-4 py-3 rounded-2xl bg-gray-700 text-gray-200 rounded-bl-md">
                      <span className="text-xs text-purple-400 font-medium block mb-1">🤖 助手</span>
                      {streamTextRef.current ? (
                        <p className="whitespace-pre-wrap text-sm leading-relaxed">
                          {streamTextRef.current}
                          <span className="inline-block w-2 h-4 ml-0.5 bg-gradient-to-b from-purple-500 to-pink-500 animate-pulse align-middle"></span>
                        </p>
                      ) : (
                        <p className="text-sm text-gray-400 flex items-center gap-2">
                          <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path></svg>
                          思考中...
                        </p>
                      )}
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>
            </div>

            {/* Params */}
            <div className="grid grid-cols-4 gap-3 mb-4">
              <div><label className="block text-xs text-gray-500 mb-1">最大Token</label><input type="number" value={maxTokens} onChange={e => setMaxTokens(parseInt(e.target.value))} className="w-full px-2 py-1.5 bg-gray-700/50 border border-gray-600 rounded-lg text-white text-sm" min={10} max={500} /></div>
              <div><label className="block text-xs text-gray-500 mb-1 flex justify-between">温度 <span className="text-purple-400">{temperature.toFixed(2)}</span></label><input type="range" value={temperature} onChange={e => setTemperature(parseFloat(e.target.value))} className="w-full accent-purple-500 h-1.5" min={0.1} max={1.5} step={0.05} /></div>
              <div><label className="block text-xs text-gray-500 mb-1">Top-K</label><input type="number" value={topK} onChange={e => setTopK(parseInt(e.target.value))} className="w-full px-2 py-1.5 bg-gray-700/50 border border-gray-600 rounded-lg text-white text-sm" min={1} max={100} /></div>
              <div><label className="block text-xs text-gray-500 mb-1 flex justify-between">重复惩罚 <span className="text-purple-400">{repetitionPenalty.toFixed(2)}</span></label><input type="range" value={repetitionPenalty} onChange={e => setRepetitionPenalty(parseFloat(e.target.value))} className="w-full accent-purple-500 h-1.5" min={1.0} max={1.5} step={0.05} /></div>
            </div>

            {/* Input */}
            <div className="flex gap-3">
              <input ref={chatInputRef} type="text" value={chatInput} onChange={e => setChatInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChatMessage() } }} className="flex-1 px-4 py-3 bg-gray-700/50 border border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 text-white placeholder-gray-500" placeholder={status.model_loaded ? "输入消息，按 Enter 发送..." : "请先加载模型..."} disabled={!status.model_loaded || chatLoading} />
              {chatLoading ? (<button onClick={stopChatGeneration} className="px-6 py-3 bg-gradient-to-r from-red-600 to-rose-600 text-white font-semibold rounded-lg shadow-lg">⏹ 停止</button>) : (<button onClick={sendChatMessage} disabled={!status.model_loaded || !chatInput.trim()} className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed text-white font-semibold rounded-lg shadow-lg">🚀 发送</button>)}
            </div>
            <p className="text-xs text-gray-500 mt-2 ml-1">💡 按 Enter 发送 · 对话历史自动上下文截断（4096字符）</p>
          </>
        )}

        {error && (<div className="mt-4 bg-red-900/30 border border-red-500/50 rounded-lg p-4"><p className="text-red-300 text-sm flex items-center gap-2"><span>❌</span>{error}</p></div>)}
      </div>

      {/* Single generate result */}
      {mode === 'single' && generated && (
        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-6 border border-gray-700 shadow-lg">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">✨ 生成结果</h3>
            <div className="flex items-center space-x-3">
              {loading && <span className="text-xs text-blue-400">生成中...</span>}
              {generationStats && (<div className="text-xs text-gray-400 flex items-center gap-3"><span>⏱️ {generationStats.time}s</span><span>🔤 {generationStats.tokens} tokens</span><span>⚡ {generationStats.speed} t/s</span></div>)}
              <button onClick={() => navigator.clipboard.writeText(generated)} className="text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 py-1.5 px-3 rounded-lg">📋 复制</button>
            </div>
          </div>
          <div className="bg-gray-950/80 rounded-xl p-5 border border-gray-800 whitespace-pre-wrap min-h-[150px] text-gray-200 font-mono text-sm">{generated}</div>
        </div>
      )}
    </div>
  )
}

export default TextGenerator