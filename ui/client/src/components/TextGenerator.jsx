import { useState, useRef, useEffect } from 'react'
import axios from 'axios'

/**
 * 文本生成组件 - 提供AI文本生成功能界面和交互逻辑
 * @param {Object} props - 组件属性对象
 * @param {string} props.apiUrl - API服务端点地址
 * @param {Object} props.status - 应用状态对象，包含模型加载状态信息
 * @returns {JSX.Element} 文本生成界面组件
 */
const TextGenerator = ({ apiUrl, status }) => {
  // 从localStorage读取保存的参数设置
  const savedPrompt = localStorage.getItem('shannon_prompt') || 'The '
  const savedMaxTokens = parseInt(localStorage.getItem('shannon_max_tokens')) || 100
  const savedTemperature = parseFloat(localStorage.getItem('shannon_temperature')) || 0.85
  const savedTopK = parseInt(localStorage.getItem('shannon_top_k')) || 40
  const savedRepetitionPenalty = parseFloat(localStorage.getItem('shannon_repetition_penalty')) || 1.15
  const savedUseStreaming = localStorage.getItem('shannon_use_streaming') !== 'false' // 默认true
  
  // 初始化文本生成相关状态变量
  const [prompt, setPrompt] = useState(savedPrompt) // 提示词输入
  const [maxTokens, setMaxTokens] = useState(savedMaxTokens) // 最大token数
  const [temperature, setTemperature] = useState(savedTemperature) // 温度参数，控制随机性
  const [topK, setTopK] = useState(savedTopK) // Top-K采样参数
  const [repetitionPenalty, setRepetitionPenalty] = useState(savedRepetitionPenalty) // 重复惩罚系数
  const [generated, setGenerated] = useState('') // 生成的文本结果
  const [loading, setLoading] = useState(false) // 加载状态
  const [error, setError] = useState('') // 错误信息
  const [useStreaming, setUseStreaming] = useState(savedUseStreaming) // 是否使用流式输出
  const [generationStats, setGenerationStats] = useState(null) // 生成统计信息
  const abortControllerRef = useRef(null) // 用于取消流式请求
  const generatedTextRef = useRef('') // 用于立即更新显示的ref

  // 当参数变化时，保存到localStorage
  useEffect(() => {
    localStorage.setItem('shannon_prompt', prompt)
  }, [prompt])

  useEffect(() => {
    localStorage.setItem('shannon_max_tokens', maxTokens.toString())
  }, [maxTokens])

  useEffect(() => {
    localStorage.setItem('shannon_temperature', temperature.toString())
  }, [temperature])

  useEffect(() => {
    localStorage.setItem('shannon_top_k', topK.toString())
  }, [topK])

  useEffect(() => {
    localStorage.setItem('shannon_repetition_penalty', repetitionPenalty.toString())
  }, [repetitionPenalty])

  useEffect(() => {
    localStorage.setItem('shannon_use_streaming', useStreaming.toString())
  }, [useStreaming])

  /**
   * 处理文本生成请求的异步函数（非流式）
   * 向后端API发送生成请求并处理响应
   */
  const handleGenerate = async () => {
    if (!status.model_loaded) {
      setError('请先加载模型')
      return
    }

    setLoading(true)
    setError('')
    setGenerated('')
    setGenerationStats(null)

    const startTime = Date.now()

    try {
      const res = await axios.post(`${apiUrl}/generate`, {
        prompt,
        max_tokens: maxTokens,
        temperature,
        top_k: topK,
        repetition_penalty: repetitionPenalty
      })
      
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(2)
      const tokens = res.data.generated_text.split(/\s+/).length
      
      setGenerated(res.data.generated_text)
      setGenerationStats({
        time: elapsed,
        tokens: tokens,
        speed: (tokens / parseFloat(elapsed)).toFixed(2)
      })
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  /**
   * 处理流式文本生成请求
   * 使用SSE（Server-Sent Events）接收实时生成的文本
   */
  const handleStreamGenerate = async () => {
    if (!status.model_loaded) {
      setError('请先加载模型')
      return
    }

    setLoading(true)
    setError('')
    setGenerated('')
    setGenerationStats(null)
    generatedTextRef.current = '' // 重置ref

    // 创建AbortController用于取消请求
    abortControllerRef.current = new AbortController()
    const signal = abortControllerRef.current.signal

    // 记录开始时间
    const startTime = Date.now()
    let tokenCount = 0

    try {
      console.log('[Streaming] Starting stream generation...')
      
      const response = await fetch(`${apiUrl}/generate/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt,
          max_tokens: maxTokens,
          temperature,
          top_k: topK,
          repetition_penalty: repetitionPenalty
        }),
        signal
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      console.log('[Streaming] Connection established, reading stream...')

      const reader = response.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        
        if (done) {
          console.log('[Streaming] Stream completed')
          break
        }

        // 解码接收到的数据
        buffer += decoder.decode(value, { stream: true })
        
        // 处理SSE格式的数据
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || '' // 保留不完整的数据在buffer中

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              
              // 处理错误
              if (data.type === 'error') {
                console.error('[Streaming] Error:', data.error)
                setError(data.error)
                setLoading(false)
                break
              }
              
              // 处理完成信号
              if (data.type === 'complete') {
                const elapsed = ((Date.now() - startTime) / 1000).toFixed(2)
                const words = generatedTextRef.current.split(/\s+/).length
                console.log(`[Streaming] Generation complete in ${elapsed}s, ${tokenCount} tokens`)
                
                setGenerationStats({
                  time: elapsed,
                  tokens: tokenCount,
                  speed: (tokenCount / parseFloat(elapsed)).toFixed(2)
                })
                
                setLoading(false)
                break
              }
              
              // 更新生成的文本 - 同时更新state和ref
              if (data.text !== undefined) {
                tokenCount++
                const elapsed = ((Date.now() - startTime) / 1000).toFixed(3)
                
                generatedTextRef.current = data.text
                
                // 强制立即更新状态
                setGenerated(data.text)
                
                // 详细的调试日志
                console.log(
                  `[${elapsed}s] Token #${tokenCount} | ` +
                  `ID: ${data.token_id} | ` +
                  `Prob: ${data.probability?.toFixed(4)} | ` +
                  `Length: ${data.text.length} chars`
                )
              }
            } catch (e) {
              console.error('[Streaming] Failed to parse SSE data:', e)
              console.error('[Streaming] Raw line:', line)
            }
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        console.log('[Streaming] Generation aborted by user')
      } else {
        console.error('[Streaming] Error:', err)
        setError(err.message || '流式生成失败')
      }
    } finally {
      setLoading(false)
      abortControllerRef.current = null
      
      const totalTime = ((Date.now() - startTime) / 1000).toFixed(2)
      console.log(`[Streaming] Finished. Total time: ${totalTime}s, Tokens: ${tokenCount}`)
    }
  }

  /**
   * 停止当前的流式生成
   */
  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* 主生成卡片 */}
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-6 border border-gray-700 shadow-lg backdrop-blur-sm">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent flex items-center gap-2">
            <span className="text-3xl">✍️</span>
            文本生成工作站
          </h2>
          {!status.model_loaded && (
            <div className="px-4 py-2 bg-yellow-500/20 border border-yellow-500/50 rounded-full">
              <p className="text-yellow-300 text-sm font-medium">⚠️ 未加载模型</p>
            </div>
          )}
        </div>

        {/* 输入区 */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-300 mb-2 flex items-center gap-2">
            <span>📝</span>
            提示词 (Prompt)
          </label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="w-full px-4 py-3 bg-gray-700/50 border border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent text-white placeholder-gray-500 transition-all resize-none"
            rows={4}
            placeholder="输入你的创意提示词..."
          />
        </div>

        {/* 参数配置区 */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <div className="group">
            <label className="block text-xs text-gray-400 mb-1.5 group-hover:text-purple-400 transition-colors">最大 Token</label>
            <input
              type="number"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value))}
              className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all hover:bg-gray-700"
              min={10}
              max={500}
            />
          </div>
          <div className="group">
            <label className="block text-xs text-gray-400 mb-1.5 group-hover:text-purple-400 transition-colors flex justify-between items-center">
              <span>温度 (Temperature)</span>
              <span className="text-purple-400 font-mono text-sm">{temperature.toFixed(2)}</span>
            </label>
            <input
              type="range"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full accent-purple-500 h-2"
              min={0.1}
              max={1.5}
              step={0.05}
            />
          </div>
          <div className="group">
            <label className="block text-xs text-gray-400 mb-1.5 group-hover:text-purple-400 transition-colors">Top-K</label>
            <input
              type="number"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all hover:bg-gray-700"
              min={1}
              max={100}
            />
          </div>
          <div className="group">
            <label className="block text-xs text-gray-400 mb-1.5 group-hover:text-purple-400 transition-colors flex justify-between items-center">
              <span>重复惩罚</span>
              <span className="text-purple-400 font-mono text-sm">{repetitionPenalty.toFixed(2)}</span>
            </label>
            <input
              type="range"
              value={repetitionPenalty}
              onChange={(e) => setRepetitionPenalty(parseFloat(e.target.value))}
              className="w-full accent-purple-500 h-2"
              min={1.0}
              max={1.5}
              step={0.05}
            />
          </div>
        </div>

        {/* 流式输出选项 */}
        <div className="mb-6 p-4 bg-gray-700/30 rounded-lg border border-gray-600/50">
          <label className="flex items-center space-x-3 cursor-pointer">
            <div className="relative">
              <input
                type="checkbox"
                checked={useStreaming}
                onChange={(e) => setUseStreaming(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-purple-500/50 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
            </div>
            <div>
              <span className="text-sm font-medium text-gray-300">启用流式输出</span>
              <p className="text-xs text-gray-500 mt-0.5">
                {useStreaming ? '✨ 实时显示生成过程，获得更好的交互体验' : '⏳ 等待完成后一次性显示全部结果'}
              </p>
            </div>
          </label>
        </div>

        {/* 生成按钮 */}
        <div className="flex space-x-3">
          <button
            onClick={useStreaming ? handleStreamGenerate : handleGenerate}
            disabled={loading || !status.model_loaded}
            className="flex-1 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-lg transition-all transform hover:scale-[1.02] active:scale-[0.98] shadow-lg disabled:shadow-none"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                {useStreaming ? '生成中...' : '生成中...'}
              </span>
            ) : (
              <span className="flex items-center justify-center gap-2">
                🚀 开始生成
              </span>
            )}
          </button>
          
          {loading && useStreaming && (
            <button
              onClick={handleStop}
              className="px-6 py-3 bg-gradient-to-r from-red-600 to-rose-600 hover:from-red-700 hover:to-rose-700 text-white font-semibold rounded-lg transition-all transform hover:scale-[1.02] active:scale-[0.98] shadow-lg"
            >
              ⏹ 停止
            </button>
          )}
        </div>

        {/* 错误提示 */}
        {error && (
          <div className="mt-4 bg-red-900/30 border border-red-500/50 rounded-lg p-4 animate-slide-in">
            <p className="text-red-300 text-sm flex items-center gap-2">
              <span>❌</span>
              {error}
            </p>
          </div>
        )}
      </div>

      {/* 输出结果区 */}
      {generated && (
        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-6 border border-gray-700 shadow-lg backdrop-blur-sm animate-fade-in">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent flex items-center gap-2">
              <span className="text-2xl">✨</span>
              生成结果
            </h3>
            <div className="flex items-center space-x-3">
              {loading && useStreaming && (
                <span className="text-xs text-blue-400 flex items-center gap-1.5">
                  <span className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></span>
                  生成中...
                </span>
              )}
              {generationStats && (
                <div className="text-xs text-gray-400 flex items-center gap-3">
                  <span className="flex items-center gap-1">
                    <span>⏱️</span>
                    {generationStats.time}s
                  </span>
                  <span className="flex items-center gap-1">
                    <span>🔤</span>
                    {generationStats.tokens} tokens
                  </span>
                  <span className="flex items-center gap-1">
                    <span>⚡</span>
                    {generationStats.speed} t/s
                  </span>
                </div>
              )}
              <button
                onClick={() => {
                  navigator.clipboard.writeText(generated)
                  // 显示复制成功提示
                  const btn = event.target
                  const originalText = btn.textContent
                  btn.textContent = '✅ 已复制'
                  setTimeout(() => btn.textContent = originalText, 2000)
                }}
                className="text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white py-1.5 px-3 rounded-lg transition-all"
              >
                📋 复制
              </button>
            </div>
          </div>
          <div className="bg-gray-950/80 rounded-xl p-5 border border-gray-800 whitespace-pre-wrap min-h-[150px] text-gray-200 leading-relaxed font-mono text-sm relative overflow-hidden">
            {generated}
            {loading && useStreaming && (
              <span className="inline-block w-2.5 h-5 ml-1 bg-gradient-to-b from-purple-500 to-pink-500 animate-pulse align-middle"></span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default TextGenerator