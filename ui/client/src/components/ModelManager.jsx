import React, { useState, useEffect } from 'react'
import axios from 'axios'

const ModelManager = ({ apiUrl, status }) => {
  const [checkpoints, setCheckpoints] = useState([])
  const [modelPath, setModelPath] = useState('../../checkpoints/shannon_b1.pt')
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')

  useEffect(() => {
    fetchCheckpoints()
  }, [])

  const fetchCheckpoints = async () => {
    try {
      const res = await axios.get(`${apiUrl}/checkpoints`)
      setCheckpoints(res.data)
    } catch (error) {
      console.error('Failed to fetch checkpoints:', error)
    }
  }

  const loadModel = async () => {
    setLoading(true)
    setMessage('')
    try {
      const res = await axios.post(`${apiUrl}/model/load`, null, {
        params: { model_path: modelPath }
      })
      setMessage(`✅ ${res.data.message}`)
    } catch (error) {
      setMessage(`❌ ${error.response?.data?.detail || error.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* 模型信息 */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h2 className="text-xl font-semibold mb-4">🗂️ 当前模型</h2>
        
        {status.model_loaded ? (
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-gray-400">词表大小</span>
                <p className="text-lg font-semibold">{status.model_info?.vocab_size || '-'}</p>
              </div>
              <div>
                <span className="text-gray-400">模型维度</span>
                <p className="text-lg font-semibold">{status.model_info?.d_model || '-'}</p>
              </div>
              <div>
                <span className="text-gray-400">参数量</span>
                <p className="text-lg font-semibold">{status.model_info?.parameters?.toLocaleString() || '-'}</p>
              </div>
              <div>
                <span className="text-gray-400">设备</span>
                <p className="text-lg font-semibold">{status.model_info?.device || '-'}</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-yellow-900/50 border border-yellow-700 rounded-lg p-4">
            <p className="text-yellow-300">⚠️ 未加载模型</p>
          </div>
        )}
      </div>

      {/* 加载模型 */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h2 className="text-xl font-semibold mb-4">📂 加载模型</h2>
        <div className="flex gap-3">
          <input
            type="text"
            value={modelPath}
            onChange={(e) => setModelPath(e.target.value)}
            className="flex-1 px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500"
            placeholder="模型路径"
          />
          <button
            onClick={loadModel}
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white font-medium py-2 px-6 rounded-lg transition-colors"
          >
            {loading ? '加载中...' : '📥 加载模型'}
          </button>
        </div>
        {message && (
          <div className={`mt-3 text-sm ${message.startsWith('✅') ? 'text-green-400' : 'text-red-400'}`}>
            {message}
          </div>
        )}
      </div>

      {/* 检查点列表 */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">💾 检查点列表</h2>
          <button
            onClick={fetchCheckpoints}
            className="text-sm text-gray-400 hover:text-gray-300"
          >
            🔄 刷新
          </button>
        </div>
        <div className="space-y-2">
          {checkpoints.length === 0 ? (
            <p className="text-gray-400 text-center py-4">暂无检查点</p>
          ) : (
            checkpoints.map((ckpt, idx) => (
              <div
                key={idx}
                className="flex justify-between items-center bg-gray-700 rounded-lg p-3 hover:bg-gray-650 transition-colors"
              >
                <div>
                  <p className="font-mono text-sm">{ckpt.name}</p>
                  <p className="text-xs text-gray-400">{ckpt.size_mb} MB • {new Date(ckpt.modified).toLocaleString()}</p>
                </div>
                <button
                  onClick={() => setModelPath(ckpt.path)}
                  className="text-blue-400 hover:text-blue-300 text-sm"
                >
                  选择
                </button>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

export default ModelManager