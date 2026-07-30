import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import {
  IconEdit,
  IconRobot,
  IconLightning,
  IconRocket,
  IconStop,
  IconClipboard,
  IconClock,
  IconType,
  IconLog,
  IconComment,
  IconTrash,
  IconX,
  IconCheck,
  IconWarning,
} from './Icons'

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
    if (!status.model_loaded) { setError('Please load a model first'); return }
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
      if (err.name !== 'AbortError') setError(err.message || 'Streaming generation failed')
    } finally {
      setChatLoading(false)
      chatAbortRef.current = null
    }
  }

  // Chat: stop generation
  const stopChatGeneration = () => {
    chatAbortRef.current?.abort()
    if (streamTextRef.current) {
      setMessages(prev => [...prev, { role: 'assistant', content: streamTextRef.current + ' [interrupted]', timestamp: new Date().toISOString() }])
      setChatStreamText('')
      streamTextRef.current = ''
    }
    setChatLoading(false)
  }

  // Single: non-streaming
  const handleGenerate = async () => {
    if (!status.model_loaded) { setError('Please load a model first'); return }
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
    if (!status.model_loaded) { setError('Please load a model first'); return }
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
      if (err.name !== 'AbortError') setError(err.message || 'Streaming generation failed')
    } finally { setLoading(false); abortControllerRef.current = null }
  }

  const handleStop = () => { abortControllerRef.current?.abort(); setLoading(false) }

  return (
    <div className="space-y-4">
      <div className="bg-[#0d1117] rounded-lg p-4 border border-[#21262d]">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-4 gap-2">
          <h2 className="text-lg font-semibold text-[#e6edf3] flex items-center gap-2">
            <IconEdit className="w-5 h-5 text-[#8b949e]" />
            Text Generation
          </h2>
          <div className="flex items-center gap-3">
            {!status.model_loaded && (
              <div className="px-3 py-1.5 bg-[#9e6a03]/15 border border-[#d29922]/30 rounded-md">
                <p className="text-[#d29922] text-xs font-medium">Model not loaded</p>
              </div>
            )}
            <div className="flex rounded-md overflow-hidden border border-[#30363d]">
              <button
                onClick={() => setMode('single')}
                className={`px-3 py-1.5 text-xs font-medium transition-base ${
                  mode === 'single'
                    ? 'bg-[#1f6feb] text-white'
                    : 'bg-[#21262d] text-[#c9d1d9] hover:bg-[#30363d]'
                }`}
              >
                Single
              </button>
              <button
                onClick={() => setMode('chat')}
                className={`px-3 py-1.5 text-xs font-medium transition-base ${
                  mode === 'chat'
                    ? 'bg-[#1f6feb] text-white'
                    : 'bg-[#21262d] text-[#c9d1d9] hover:bg-[#30363d]'
                }`}
              >
                Chat
              </button>
            </div>
          </div>
        </div>

        {/* System prompt */}
        <div className="mb-4 space-y-3">
          <div>
            <label className="block text-[11px] font-medium text-[#c9d1d9] mb-1.5 flex items-center gap-1.5">
              <IconRobot className="w-3.5 h-3.5 text-[#8b949e]" />
              System Prompt
              <span className="text-[10px] text-[#484f58]">(optional)</span>
            </label>
            <textarea
              value={systemPrompt}
              onChange={e => setSystemPrompt(e.target.value)}
              className="w-full px-2.5 py-1.5 bg-[#0d1117] border border-[#30363d] rounded-md focus:ring-2 focus:ring-[#1f6feb] focus:border-transparent text-[#e6edf3] placeholder-[#484f58] text-xs transition-base resize-none"
              rows={1}
              placeholder="e.g. You are a professional coding assistant..."
            />
          </div>
          {mode === 'chat' && (
            <div className="flex items-center gap-4">
              <label className="text-xs font-medium text-[#c9d1d9] flex items-center gap-1.5">
                <IconLog className="w-3.5 h-3.5 text-[#8b949e]" />
                Template:
              </label>
              <select
                value={convTemplate}
                onChange={e => setConvTemplate(e.target.value)}
                disabled={!!convId}
                className="px-3 py-1.5 bg-[#161b22] border border-[#30363d] rounded-md text-[#e6edf3] text-xs focus:ring-2 focus:ring-[#1f6feb] disabled:opacity-50"
              >
                <option value="simple">Simple [default]</option>
                <option value="chatml">ChatML (OpenAI-style)</option>
                <option value="llama3">Llama3</option>
              </select>
              {convId && <span className="text-[10px] text-[#484f58]">(locked after creation)</span>}
            </div>
          )}
        </div>

            {/* ========== SINGLE MODE ========== */}
        {mode === 'single' && (
          <>
            <div className="mb-3">
              <label className="block text-[11px] font-medium text-[#c9d1d9] mb-1.5">
                <IconEdit className="w-3.5 h-3.5 text-[#8b949e] inline mr-1" />
                User Prompt
              </label>
              <textarea
                value={prompt}
                onChange={e => setPrompt(e.target.value)}
                className="w-full px-2.5 py-1.5 bg-[#0d1117] border border-[#30363d] rounded-md focus:ring-2 focus:ring-[#1f6feb] focus:border-transparent text-[#e6edf3] placeholder-[#484f58] text-xs transition-base resize-none"
                rows={3}
                placeholder="Enter your prompt..."
              />
            </div>

            {/* Parameter grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-3">
              <div>
                <label className="block text-[11px] text-[#8b949e] mb-1">Max Tokens</label>
                <input
                  type="number" value={maxTokens} onChange={e => setMaxTokens(parseInt(e.target.value))}
                  className="w-full px-2.5 py-1 bg-[#161b22] border border-[#30363d] rounded-md text-[#e6edf3] text-xs focus:ring-2 focus:ring-[#1f6feb]" min={10} max={500}
                />
              </div>
              <div>
                <label className="block text-[11px] text-[#8b949e] mb-1 flex justify-between">
                  <span>Temperature</span>
                  <span className="text-[#58a6ff] font-mono">{temperature.toFixed(2)}</span>
                </label>
                <input
                  type="range" value={temperature} onChange={e => setTemperature(parseFloat(e.target.value))}
                  className="w-full accent-[#58a6ff] h-1.5" min={0.1} max={1.5} step={0.05}
                />
              </div>
              <div>
                <label className="block text-[11px] text-[#8b949e] mb-1">Top-K</label>
                <input
                  type="number" value={topK} onChange={e => setTopK(parseInt(e.target.value))}
                  className="w-full px-3 py-1.5 bg-[#161b22] border border-[#30363d] rounded-md text-[#e6edf3] text-sm focus:ring-2 focus:ring-[#1f6feb]" min={1} max={100}
                />
              </div>
              <div>
                <label className="block text-[11px] text-[#8b949e] mb-1 flex justify-between">
                  <span>Rep. Penalty</span>
                  <span className="text-[#58a6ff] font-mono">{repetitionPenalty.toFixed(2)}</span>
                </label>
                <input
                  type="range" value={repetitionPenalty} onChange={e => setRepetitionPenalty(parseFloat(e.target.value))}
                  className="w-full accent-[#58a6ff] h-1.5" min={1.0} max={1.5} step={0.05}
                />
              </div>
            </div>

            {/* Streaming toggle */}
            <div className="mb-3 p-3 bg-[#161b22] rounded-md border border-[#21262d]">
              <label className="flex items-center space-x-3 cursor-pointer">
                <div className="relative">
                  <input type="checkbox" checked={useStreaming} onChange={e => setUseStreaming(e.target.checked)} className="sr-only peer" />
                  <div className="w-9 h-5 bg-[#30363d] peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-[#1f6feb] rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-[#30363d] after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-[#1f6feb]"></div>
                </div>
                <div>
                  <span className="text-xs font-medium text-[#c9d1d9]">Enable streaming output</span>
                  <p className="text-[10px] text-[#484f58] mt-0.5">
                    {useStreaming ? 'Real-time display of generation' : 'Display result after completion'}
                  </p>
                </div>
              </label>
            </div>

            {/* Action buttons */}
            <div className="flex gap-3">
              <button
                onClick={useStreaming ? handleStreamGenerate : handleGenerate}
                disabled={loading || !status.model_loaded}
              className="flex-1 flex items-center justify-center gap-1.5 bg-[#1f6feb] hover:bg-[#388bfd] disabled:bg-[#21262d] disabled:cursor-not-allowed text-white font-semibold py-2 px-3 rounded-md transition-base text-xs"
              >
                {loading ? (
                  <>
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Generating...
                  </>
                ) : (
                  <>
                    <IconRocket className="w-4 h-4" /> Generate
                  </>
                )}
              </button>
              {loading && useStreaming && (
                <button onClick={handleStop} className="flex items-center gap-1 px-3 py-2 bg-[#da3633] hover:bg-[#f85149] text-white font-semibold rounded-md text-xs transition-base">
                  <IconStop className="w-4 h-4" /> Stop
                </button>
              )}
            </div>
          </>
        )}

        {/* ========== CHAT MODE ========== */}
        {mode === 'chat' && (
          <>
            {/* Conversation bar */}
            <div className="flex items-center justify-between mb-3 p-2.5 bg-[#161b22] rounded-md border border-[#21262d]">
              <span className="text-xs text-[#c9d1d9]">
                {convId ? (
                  <span className="flex items-center gap-2">
                    <span className="w-2 h-2 bg-[#3fb950] rounded-full"></span>
                    Session: <code className="text-[#58a6ff] text-xs">{convId}</code>
                  </span>
                ) : (
                  'No session -- auto-created on first message'
                )}
              </span>
              <div className="flex items-center gap-2">
                {convId && (
                  <>
                    <button onClick={clearConversation} className="flex items-center gap-1 px-2.5 py-1 text-[10px] bg-[#21262d] hover:bg-[#30363d] text-[#c9d1d9] rounded-md transition-base" title="Clear history">
                      <IconTrash className="w-3 h-3" /> Clear
                    </button>
                    <button onClick={deleteConversation} className="flex items-center gap-1 px-2.5 py-1 text-[10px] bg-[#da3633]/20 hover:bg-[#da3633]/40 text-[#f85149] rounded-md transition-base" title="Delete session">
                      <IconX className="w-3 h-3" /> Delete
                    </button>
                  </>
                )}
                <button
                  onClick={createConversation}
                  disabled={!status.model_loaded || chatLoading}
                  className="flex items-center gap-1 px-2.5 py-1 text-[10px] bg-[#1f6feb] hover:bg-[#388bfd] disabled:bg-[#21262d] text-white rounded-md transition-base"
                >
                  + New
                </button>
              </div>
            </div>

            {/* Messages area */}
            <div className="mb-3 bg-[#0d1117] rounded-md border border-[#21262d] overflow-hidden">
              <div className="h-[280px] overflow-y-auto p-3 space-y-3">
                {messages.length === 0 && !chatStreamText && !chatLoading && (
                  <div className="flex items-center justify-center h-full text-[#8b949e]">
                    <div className="text-center">
                      <IconComment className="w-10 h-10 mx-auto mb-2 opacity-30" />
                      <p className="text-sm">Start a conversation</p>
                      <p className="text-[10px] mt-1 text-[#484f58]">Send a message to auto-create a session</p>
                    </div>
                  </div>
                )}

                {messages.map((msg, idx) => (
                  <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[80%] px-3 py-2 rounded-lg ${
                      msg.role === 'user'
                        ? 'bg-[#1f6feb] text-white rounded-br-md'
                        : 'bg-[#161b22] text-[#e6edf3] rounded-bl-md border border-[#21262d]'
                    }`}>
                      <span className={`text-[10px] font-semibold block mb-1 ${msg.role === 'user' ? 'text-[#f0883e]' : 'text-[#58a6ff]'}`}>
                        {msg.role === 'user' ? 'You' : 'Assistant'}
                      </span>
                      <p className="whitespace-pre-wrap text-sm leading-relaxed">{msg.content}</p>
                    </div>
                  </div>
                ))}

                {/* Streaming bubble */}
                {(chatStreamText || chatLoading) && (
                  <div className="flex justify-start">
                    <div className="max-w-[80%] px-4 py-3 rounded-lg bg-[#161b22] text-[#e6edf3] rounded-bl-md border border-[#21262d]">
                      <span className="text-[10px] text-[#58a6ff] font-semibold block mb-1">Assistant</span>
                      {streamTextRef.current ? (
                        <p className="whitespace-pre-wrap text-sm leading-relaxed">
                          {streamTextRef.current}
                          <span className="inline-block w-2 h-4 ml-0.5 bg-[#58a6ff] animate-pulse align-middle rounded-sm"></span>
                        </p>
                      ) : (
                        <p className="text-sm text-[#8b949e] flex items-center gap-2">
                          <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
                          </svg>
                          Thinking...
                        </p>
                      )}
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>
            </div>

            {/* Chat params */}
            <div className="grid grid-cols-4 gap-2 mb-3">
              <div>
                <label className="block text-[10px] text-[#484f58] mb-1">Max Tokens</label>
                <input type="number" value={maxTokens} onChange={e => setMaxTokens(parseInt(e.target.value))} className="w-full px-2 py-1.5 bg-[#161b22] border border-[#30363d] rounded-md text-[#e6edf3] text-xs" min={10} max={500} />
              </div>
              <div>
                <label className="block text-[10px] text-[#484f58] mb-1 flex justify-between">
                  Temp <span className="text-[#58a6ff]">{temperature.toFixed(2)}</span>
                </label>
                <input type="range" value={temperature} onChange={e => setTemperature(parseFloat(e.target.value))} className="w-full accent-[#58a6ff] h-1" min={0.1} max={1.5} step={0.05} />
              </div>
              <div>
                <label className="block text-[10px] text-[#484f58] mb-1">Top-K</label>
                <input type="number" value={topK} onChange={e => setTopK(parseInt(e.target.value))} className="w-full px-2 py-1.5 bg-[#161b22] border border-[#30363d] rounded-md text-[#e6edf3] text-xs" min={1} max={100} />
              </div>
              <div>
                <label className="block text-[10px] text-[#484f58] mb-1 flex justify-between">
                  Rep. P <span className="text-[#58a6ff]">{repetitionPenalty.toFixed(2)}</span>
                </label>
                <input type="range" value={repetitionPenalty} onChange={e => setRepetitionPenalty(parseFloat(e.target.value))} className="w-full accent-[#58a6ff] h-1" min={1.0} max={1.5} step={0.05} />
              </div>
            </div>

            {/* Chat input */}
            <div className="flex gap-3">
              <input
                ref={chatInputRef}
                type="text" value={chatInput}
                onChange={e => setChatInput(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChatMessage() } }}
                className="flex-1 px-3 py-2 bg-[#161b22] border border-[#30363d] rounded-md focus:ring-2 focus:ring-[#1f6feb] text-[#e6edf3] placeholder-[#484f58] text-sm"
                placeholder={status.model_loaded ? "Type a message, press Enter..." : "Load a model first..."}
                disabled={!status.model_loaded || chatLoading}
              />
              {chatLoading ? (
                <button onClick={stopChatGeneration} className="flex items-center gap-1 px-3 py-2 bg-[#da3633] hover:bg-[#f85149] text-white font-semibold rounded-md text-xs transition-base">
                  <IconStop className="w-4 h-4" /> Stop
                </button>
              ) : (
                <button
                  onClick={sendChatMessage}
                  disabled={!status.model_loaded || !chatInput.trim()}
                  className="flex items-center gap-1 px-3 py-2 bg-[#1f6feb] hover:bg-[#388bfd] disabled:bg-[#21262d] disabled:cursor-not-allowed text-white font-semibold rounded-md text-xs transition-base"
                >
                  <IconRocket className="w-4 h-4" /> Send
                </button>
              )}
            </div>
            <p className="text-[10px] text-[#484f58] mt-1.5 ml-1">
              Press Enter to send - Context auto-truncates (4096 chars)
            </p>
          </>
        )}

        {/* Error display */}
        {error && (
          <div className="mt-3 bg-[#da3633]/10 border border-[#f85149]/20 rounded-md p-2.5">
            <p className="text-[#f85149] text-xs flex items-center gap-2">
              <IconX className="w-3.5 h-3.5 flex-shrink-0" />
              {error}
            </p>
          </div>
        )}
      </div>

      {/* Single generate result */}
      {mode === 'single' && generated && (
        <div className="bg-[#0d1117] rounded-lg p-4 border border-[#21262d]">
          <div className="flex justify-between items-center mb-3">
            <h3 className="text-sm font-semibold text-[#e6edf3]">Generated Output</h3>
            <div className="flex items-center gap-3">
              {loading && <span className="text-xs text-[#58a6ff]">Generating...</span>}
              {generationStats && (
                <div className="text-[10px] text-[#8b949e] flex items-center gap-3">
                  <span className="flex items-center gap-1"><IconClock className="w-3 h-3" /> {generationStats.time}s</span>
                  <span className="flex items-center gap-1"><IconType className="w-3 h-3" /> {generationStats.tokens} tokens</span>
                  <span className="flex items-center gap-1"><IconLightning className="w-3 h-3" /> {generationStats.speed} t/s</span>
                </div>
              )}
              <button
                onClick={() => navigator.clipboard.writeText(generated)}
                className="flex items-center gap-1 text-[10px] bg-[#21262d] hover:bg-[#30363d] text-[#c9d1d9] py-1 px-2.5 rounded-md transition-base"
              >
                <IconClipboard className="w-3 h-3" /> Copy
              </button>
            </div>
          </div>
          <div className="bg-[#0d1117] rounded-md p-3 border border-[#21262d] whitespace-pre-wrap min-h-[80px] max-h-[200px] overflow-y-auto text-[#e6edf3] font-mono text-xs">
            {generated}
          </div>
        </div>
      )}
    </div>
  )
}

export default TextGenerator