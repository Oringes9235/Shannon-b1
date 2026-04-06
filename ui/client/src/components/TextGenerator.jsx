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

    try {
      const res = await axios.post(`${apiUrl}/generate`, {
        prompt,
        max_tokens: maxTokens,
        temperature,
        top_k: topK,
        repetition_penalty: repetitionPenalty
      })
      setGenerated(res.data.generated_text)
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
                console.log(`[Streaming] Generation complete in ${elapsed}s, ${tokenCount} tokens`)
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
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h2 className="text-xl font-semibold mb-4">🎨 文本生成</h2>
        
        {!status.model_loaded && (
          <div className="bg-yellow-900/50 border border-yellow-700 rounded-lg p-4 mb-4">
            <p className="text-yellow-300">⚠️ 未加载模型，请在"模型管理"中加载模型</p>
          </div>
        )}

        {/* 输入区 */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-300 mb-2">提示词</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-white"
            rows={3}
            placeholder="输入提示词..."
          />
        </div>

        {/* 参数区 */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div>
            <label className="block text-xs text-gray-400 mb-1">最大 Token</label>
            <input
              type="number"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value))}
              className="w-full px-3 py-1 bg-gray-700 border border-gray-600 rounded"
              min={10}
              max={500}
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">温度</label>
            <input
              type="range"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full"
              min={0.1}
              max={1.5}
              step={0.05}
            />
            <span className="text-xs text-gray-400">{temperature}</span>
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Top-K</label>
            <input
              type="number"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              className="w-full px-3 py-1 bg-gray-700 border border-gray-600 rounded"
              min={1}
              max={100}
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">重复惩罚</label>
            <input
              type="range"
              value={repetitionPenalty}
              onChange={(e) => setRepetitionPenalty(parseFloat(e.target.value))}
              className="w-full"
              min={1.0}
              max={1.5}
              step={0.05}
            />
            <span className="text-xs text-gray-400">{repetitionPenalty}</span>
          </div>
        </div>

        {/* 流式输出选项 */}
        <div className="mb-4 flex items-center space-x-4">
          <label className="flex items-center space-x-2 cursor-pointer">
            <input
              type="checkbox"
              checked={useStreaming}
              onChange={(e) => setUseStreaming(e.target.checked)}
              className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
            />
            <span className="text-sm text-gray-300">启用流式输出</span>
          </label>
          <span className="text-xs text-gray-500">
            {useStreaming ? '实时显示生成过程' : '等待完成后一次性显示'}
          </span>
        </div>

        {/* 生成按钮 */}
        <div className="flex space-x-2">
          <button
            onClick={useStreaming ? handleStreamGenerate : handleGenerate}
            disabled={loading || !status.model_loaded}
            className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-medium py-2 px-4 rounded-lg transition-colors"
          >
            {loading ? (useStreaming ? '生成中...' : '生成中...') : '🚀 生成文本'}
          </button>
          
          {loading && useStreaming && (
            <button
              onClick={handleStop}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white font-medium rounded-lg transition-colors"
            >
              ⏹ 停止
            </button>
          )}
        </div>

        {/* 错误提示 */}
        {error && (
          <div className="mt-4 bg-red-900/50 border border-red-700 rounded-lg p-3">
            <p className="text-red-300 text-sm">{error}</p>
          </div>
        )}

        {/* 输出区 */}
        {generated && (
          <div className="mt-6">
            <div className="flex justify-between items-center mb-2">
              <label className="text-sm font-medium text-gray-300">生成结果</label>
              <div className="flex space-x-2">
                {loading && useStreaming && (
                  <span className="text-xs text-blue-400 animate-pulse">● 生成中...</span>
                )}
                <button
                  onClick={() => navigator.clipboard.writeText(generated)}
                  className="text-xs text-gray-400 hover:text-gray-300"
                >
                  📋 复制
                </button>
              </div>
            </div>
            <div className="bg-gray-900 rounded-lg p-4 border border-gray-700 whitespace-pre-wrap min-h-[100px]">
              {generated}
              {loading && useStreaming && (
                <span className="inline-block w-2 h-4 ml-1 bg-blue-500 animate-pulse"></span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default TextGenerator