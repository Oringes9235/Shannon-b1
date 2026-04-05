import React, { useState } from 'react'
import axios from 'axios'

const TextGenerator = ({ apiUrl, status }) => {
  const [prompt, setPrompt] = useState('The ')
  const [maxTokens, setMaxTokens] = useState(100)
  const [temperature, setTemperature] = useState(0.85)
  const [topK, setTopK] = useState(40)
  const [repetitionPenalty, setRepetitionPenalty] = useState(1.15)
  const [generated, setGenerated] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

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

        {/* 生成按钮 */}
        <button
          onClick={handleGenerate}
          disabled={loading || !status.model_loaded}
          className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-medium py-2 px-4 rounded-lg transition-colors"
        >
          {loading ? '生成中...' : '🚀 生成文本'}
        </button>

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
              <button
                onClick={() => navigator.clipboard.writeText(generated)}
                className="text-xs text-gray-400 hover:text-gray-300"
              >
                📋 复制
              </button>
            </div>
            <div className="bg-gray-900 rounded-lg p-4 border border-gray-700 whitespace-pre-wrap">
              {generated}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default TextGenerator