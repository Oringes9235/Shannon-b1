import React, { useState, useEffect } from 'react'
import axios from 'axios'

/**
 * 模型管理组件 - 用于管理和加载机器学习模型
 * 提供模型加载、检查点管理以及模型信息展示功能
 * 
 * @param {Object} props - 组件属性对象
 * @param {string} props.apiUrl - API服务端点地址
 * @param {Object} props.status - 包含当前模型状态的对象
 * @returns {JSX.Element} 模型管理界面组件
 */
const ModelManager = ({ apiUrl, status }) => {
  // 从localStorage读取保存的模型路径，如果没有则使用默认值
  const savedModelPath = localStorage.getItem('shannon_model_path') || '../../checkpoints/shannon_b1.pt'
  
  // 初始化组件状态：检查点列表、模型路径、加载状态和消息提示
  const [checkpoints, setCheckpoints] = useState([])
  const [modelPath, setModelPath] = useState(savedModelPath)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')

  // 组件挂载时获取检查点列表
  useEffect(() => {
    fetchCheckpoints()
  }, [])

  // 当modelPath变化时，保存到localStorage
  useEffect(() => {
    localStorage.setItem('shannon_model_path', modelPath)
  }, [modelPath])

  /**
   * 获取可用的模型检查点列表
   * 从API端点获取所有可用的模型文件信息
   * 
   * @async
   * @function fetchCheckpoints
   * @returns {void}
   */
  const fetchCheckpoints = async () => {
    try {
      const res = await axios.get(`${apiUrl}/checkpoints`)
      setCheckpoints(res.data)
    } catch (error) {
      console.error('Failed to fetch checkpoints:', error)
    }
  }

  /**
   * 向后端API发送请求加载指定路径的模型
   * 显示加载状态并在完成后显示成功或错误消息
   * 
   * @async
   * @function loadModel
   * @returns {void}
   */
  const loadModel = async () => {
    setLoading(true)
    setMessage('')
    try {
      const res = await axios.post(`${apiUrl}/model/load`, null, {
        params: { model_path: modelPath }
      })
      setMessage(`✅ ${res.data.message}`)
      
      // 3秒后清除消息
      setTimeout(() => setMessage(''), 3000)
    } catch (error) {
      setMessage(`❌ ${error.response?.data?.detail || error.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* 模型信息卡片 */}
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-6 border border-gray-700 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent flex items-center gap-2">
          <span className="text-3xl">🗂️</span>
          当前模型
        </h2>
        
        {status.model_loaded ? (
          <div className="bg-gradient-to-br from-green-900/30 to-emerald-900/30 border border-green-500/50 rounded-xl p-5">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-green-400 font-semibold">模型已加载</span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-gray-800/50 rounded-lg p-3">
                <span className="text-xs text-gray-400 block mb-1">词表大小</span>
                <p className="text-xl font-bold text-white font-mono">{status.model_info?.vocab_size || '-'}</p>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <span className="text-xs text-gray-400 block mb-1">模型维度</span>
                <p className="text-xl font-bold text-white font-mono">{status.model_info?.d_model || '-'}</p>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <span className="text-xs text-gray-400 block mb-1">参数量</span>
                <p className="text-xl font-bold text-white font-mono">{status.model_info?.parameters?.toLocaleString() || '-'}</p>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <span className="text-xs text-gray-400 block mb-1">设备</span>
                <p className="text-xl font-bold text-white font-mono">{status.model_info?.device || '-'}</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-gradient-to-br from-yellow-900/30 to-orange-900/30 border border-yellow-500/50 rounded-xl p-5">
            <div className="flex items-center gap-3">
              <span className="text-3xl">⚠️</span>
              <div>
                <p className="text-yellow-300 font-semibold">未加载模型</p>
                <p className="text-sm text-yellow-400/70 mt-1">请从下方选择或输入模型路径进行加载</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 加载模型区域 */}
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-4 sm:p-6 border border-gray-700 shadow-lg">
        <h2 className="text-xl sm:text-2xl font-bold mb-4 sm:mb-6 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent flex items-center gap-2">
          <span className="text-2xl sm:text-3xl">📂</span>
          <span className="hidden xs:inline">加载模型</span>
          <span className="xs:hidden">加载</span>
        </h2>
        <div className="flex flex-col sm:flex-row gap-3">
          <input
            type="text"
            value={modelPath}
            onChange={(e) => setModelPath(e.target.value)}
            className="flex-1 px-4 py-2.5 sm:py-3 bg-gray-700/50 border border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all placeholder-gray-500 text-sm sm:text-base"
            placeholder="输入模型文件路径..."
          />
          <button
            onClick={loadModel}
            disabled={loading}
            className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-semibold py-2.5 sm:py-3 px-4 sm:px-6 rounded-lg transition-all transform hover:scale-[1.02] active:scale-[0.98] shadow-lg disabled:shadow-none disabled:cursor-not-allowed text-sm sm:text-base whitespace-nowrap"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-4 w-4 sm:h-5 sm:w-5" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                加载中...
              </span>
            ) : (
              <span className="flex items-center justify-center gap-2">
                📥 加载模型
              </span>
            )}
          </button>
        </div>
        {message && (
          <div className={`mt-3 sm:mt-4 p-3 rounded-lg animate-slide-in ${
            message.startsWith('✅') 
              ? 'bg-green-900/30 border border-green-500/50 text-green-300' 
              : 'bg-red-900/30 border border-red-500/50 text-red-300'
          }`}>
            <p className="text-xs sm:text-sm">{message}</p>
          </div>
        )}
      </div>

      {/* 检查点列表 */}
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-4 sm:p-6 border border-gray-700 shadow-lg">
        <div className="flex justify-between items-center mb-4 sm:mb-6">
          <h2 className="text-xl sm:text-2xl font-bold bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent flex items-center gap-2">
            <span className="text-2xl sm:text-3xl">💾</span>
            <span className="hidden xs:inline">检查点列表</span>
            <span className="xs:hidden">检查点</span>
          </h2>
          <button
            onClick={fetchCheckpoints}
            className="text-xs sm:text-sm bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white py-1.5 px-3 sm:py-2 sm:px-4 rounded-lg transition-all flex items-center gap-1.5 sm:gap-2"
          >
            <span>🔄</span>
            <span className="hidden xs:inline">刷新</span>
          </button>
        </div>
        
        <div className="space-y-2 sm:space-y-3 max-h-64 sm:max-h-96 overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-gray-900">
          {checkpoints.length === 0 ? (
            <div className="text-center py-8 sm:py-12">
              <div className="text-5xl sm:text-6xl mb-3 sm:mb-4 opacity-50">📭</div>
              <p className="text-base sm:text-lg text-gray-400">暂无检查点</p>
              <p className="text-xs sm:text-sm text-gray-500 mt-2 px-4">训练模型后将在此处显示检查点文件</p>
            </div>
          ) : (
            checkpoints.map((ckpt, idx) => (
              <div
                key={idx}
                className="flex flex-col sm:flex-row sm:justify-between sm:items-center bg-gray-700/30 hover:bg-gray-700/50 rounded-xl p-3 sm:p-4 border border-gray-600/50 hover:border-purple-500/50 transition-all group cursor-pointer"
                onClick={() => setModelPath(ckpt.path)}
              >
                <div className="flex-1 mb-2 sm:mb-0">
                  <p className="font-mono text-xs sm:text-sm text-white font-medium group-hover:text-purple-400 transition-colors break-all">{ckpt.name}</p>
                  <div className="flex items-center gap-2 sm:gap-3 mt-1 flex-wrap">
                    <span className="text-[10px] sm:text-xs text-gray-400 flex items-center gap-1">
                      <span>💾</span>
                      {ckpt.size_mb} MB
                    </span>
                    <span className="text-[10px] sm:text-xs text-gray-400 flex items-center gap-1">
                      <span>📅</span>
                      {new Date(ckpt.modified).toLocaleDateString()}
                    </span>
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    setModelPath(ckpt.path)
                  }}
                  className="ml-0 sm:ml-4 px-3 sm:px-4 py-1.5 sm:py-2 bg-purple-600/20 hover:bg-purple-600/40 text-purple-400 hover:text-purple-300 rounded-lg transition-all text-xs sm:text-sm font-medium border border-purple-500/30 self-start sm:self-center"
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