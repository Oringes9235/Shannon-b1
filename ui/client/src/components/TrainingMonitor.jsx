import { useState, useEffect, useRef, useCallback } from 'react'
import axios from 'axios'
import { subscribe } from '../ws'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart, Legend } from 'recharts'

/**
 * 训练监控组件 - 用于监控和控制机器学习模型的训练过程
 * 提供训练配置、实时进度监控、损失曲线可视化和训练日志功能
 * @param {Object} props - 组件属性
 * @param {string} props.apiUrl - API服务端点地址
 * @returns {JSX.Element} 训练监控界面组件
 */
const TrainingMonitor = ({ apiUrl }) => {
  // 从localStorage读取保存的训练配置
  const savedConfig = localStorage.getItem('shannon_training_config')
  const defaultConfig = {
    tokenizer: 'char',
    vocab_size: 200,
    d_model: 128,
    num_layers: 3,
    epochs: 30,
    batch_size: 32,
    seq_len: 64,
    lr: 0.0005,
    dropout: 0.3,
    weight_decay: 0.1,
    patience: 10,
    num_heads: 8,
    grad_accum: 1,
    warmup_steps: 1000
  }
  
  // 合并默认配置和保存的配置，确保新参数有默认值
  const initialConfig = savedConfig 
    ? { ...defaultConfig, ...JSON.parse(savedConfig) }
    : defaultConfig
  
  // 初始化训练配置状态，包含模型架构和训练超参数
  const [trainingConfig, setTrainingConfig] = useState(initialConfig)

  // 当trainingConfig变化时，保存到localStorage
  useEffect(() => {
    localStorage.setItem('shannon_training_config', JSON.stringify(trainingConfig))
  }, [trainingConfig])

  // 训练状态：运行状态和进度信息
  const [trainingStatus, setTrainingStatus] = useState({ is_running: false, progress: 0 })

  // 损失历史记录，用于绘制损失曲线
  const [lossHistory, setLossHistory] = useState([])

  // 训练日志记录 - 限制最大日志数量为500条，防止内存溢出和卡顿
  const MAX_LOGS = 500
  const [logs, setLogs] = useState([])

  // WebSocket连接管理
  const [ws, setWs] = useState(null)

  // 加载状态，用于控制按钮状态
  const [loading, setLoading] = useState(false)

  // 日志容器引用，用于自动滚动
  const logsContainerRef = useRef(null)

  // 当日志更新时，自动滚动到底部（使用即时滚动，更快速）
  useEffect(() => {
    if (logsContainerRef.current && logs.length > 0) {
      // 直接设置scrollTop到最大值，实现即时滚动
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight
    }
  }, [logs])

  // 设置WebSocket订阅，用于接收实时训练消息
  useEffect(() => {
    // 使用共享 WebSocket，避免组件卸载时关闭连接导致训练监控中断
    const unsub = subscribe((data) => handleWebSocketMessage(data), apiUrl)
    return () => unsub()
  }, [])

  // 设置定时器，定期获取训练状态更新
  useEffect(() => {
    // 定时获取训练状态
    const interval = setInterval(fetchTrainingStatus, 2000)
    return () => clearInterval(interval)
  }, [])

  /**
   * 处理WebSocket接收到的消息，根据消息类型更新相应状态
   * @param {Object} data - WebSocket消息数据
   * @param {string} data.type - 消息类型（training_progress, training_epoch_complete等）
   * @param {Object} data.data - 消息携带的具体数据
   */
  const handleWebSocketMessage = useCallback((data) => {
    if (data.type === 'training_log') {
      // 处理实时训练日志 - 使用函数式更新并限制数量
      setLogs(prev => {
        const newLogs = [...prev, {
          time: new Date().toLocaleTimeString(),
          message: data.data.line
        }]
        // 如果超过最大数量，只保留最新的MAX_LOGS条
        return newLogs.length > MAX_LOGS ? newLogs.slice(-MAX_LOGS) : newLogs
      })
    } else if (data.type === 'training_progress') {
      setTrainingStatus(data.data)
    } else if (data.type === 'training_epoch_complete') {
      setLossHistory(prev => [...prev, {
        epoch: data.data.epoch,
        train_loss: data.data.train_loss,
        val_loss: data.data.val_loss
      }])
      setLogs(prev => {
        const newLogs = [...prev, {
          time: new Date().toLocaleTimeString(),
          message: `Epoch ${data.data.epoch}: train_loss=${data.data.train_loss.toFixed(4)}, val_loss=${data.data.val_loss.toFixed(4)}`
        }]
        return newLogs.length > MAX_LOGS ? newLogs.slice(-MAX_LOGS) : newLogs
      })
    } else if (data.type === 'training_completed') {
      setTrainingStatus({ is_running: false, progress: 1 })
      setLoading(false)
      setLogs(prev => {
        const newLogs = [...prev, {
          time: new Date().toLocaleTimeString(),
          message: `✅ Training completed! Best loss: ${data.data.best_loss?.toFixed(4) || 'N/A'}`
        }]
        return newLogs.length > MAX_LOGS ? newLogs.slice(-MAX_LOGS) : newLogs
      })
    } else if (data.type === 'training_error') {
      setLogs(prev => {
        const newLogs = [...prev, {
          time: new Date().toLocaleTimeString(),
          message: `❌ Error: ${data.data.error || data.data.returncode}`
        }]
        return newLogs.length > MAX_LOGS ? newLogs.slice(-MAX_LOGS) : newLogs
      })
      setLoading(false)
    } else if (data.type === 'training_started') {
      // 训练已启动
      setTrainingStatus(data.data)
    }
  }, [])

  /**
   * 异步获取当前训练状态
   * 从API端点获取最新的训练进度和状态信息
   */
  const fetchTrainingStatus = async () => {
    try {
      const res = await axios.get(`${apiUrl}/train/status`)
      setTrainingStatus(res.data)
    } catch (error) {
      console.error('Failed to fetch training status:', error)
    }
  }

  /**
   * 启动训练过程
   * 发送训练配置到后端开始训练，并初始化相关状态
   */
  const startTraining = async () => {
    setLoading(true)
    setLossHistory([])
    setLogs([])
    try {
      const res = await axios.post(`${apiUrl}/train/start`, trainingConfig, {
        headers: { 'Content-Type': 'application/json' }
      })
      if (res.data && res.data.success) {
        setLogs(prev => [{
          time: new Date().toLocaleTimeString(),
          message: '🚀 Training started...'
        }])
      } else {
        setLogs(prev => [{
          time: new Date().toLocaleTimeString(),
          message: `❌ Failed to start training: ${res.data?.message || JSON.stringify(res.data)}`
        }])
        setLoading(false)
        return
      }
    } catch (error) {
      console.error('Failed to start training:', error)
      setLogs(prev => [{
        time: new Date().toLocaleTimeString(),
        message: `❌ Failed to start training: ${error.response?.data?.detail || error.message}`
      }])
      setLoading(false)
    }
  }

  /**
   * 请求停止当前训练过程
   * 向后端发送停止训练指令
   */
  const stopTraining = async () => {
    try {
      await axios.post(`${apiUrl}/train/stop`)
      setLogs(prev => {
        const newLogs = [...prev, {
          time: new Date().toLocaleTimeString(),
          message: '⏹️ Training stop requested'
        }]
        return newLogs.length > MAX_LOGS ? newLogs.slice(-MAX_LOGS) : newLogs
      })
    } catch (error) {
      console.error('Failed to stop training:', error)
    }
  }

  return (
    <div className="space-y-6">
      {/* 训练配置部分 - 包含各种训练参数的输入控件 */}
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-4 sm:p-6 border border-gray-700 shadow-lg backdrop-blur-sm">
        <div className="flex items-center justify-between mb-4 sm:mb-6">
          <h2 className="text-xl sm:text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent flex items-center gap-2">
            <span className="text-2xl sm:text-3xl">⚙️</span>
            <span className="hidden xs:inline">训练配置</span>
            <span className="xs:hidden">配置</span>
          </h2>
          {trainingStatus.is_running && (
            <div className="flex items-center gap-1.5 sm:gap-2 px-2 sm:px-3 py-1 bg-green-500/20 border border-green-500/50 rounded-full">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-[10px] sm:text-xs text-green-400 font-medium hidden xs:inline">训练中</span>
            </div>
          )}
        </div>
        
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
          <div className="group">
            <label className="block text-[10px] sm:text-xs text-gray-400 mb-1 sm:mb-1.5 group-hover:text-blue-400 transition-colors">分词器</label>
            <select
              value={trainingConfig.tokenizer}
              onChange={(e) => setTrainingConfig({...trainingConfig, tokenizer: e.target.value})}
              className="w-full px-2 sm:px-3 py-1.5 sm:py-2 bg-gray-700/50 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all hover:bg-gray-700 text-xs sm:text-sm"
              disabled={trainingStatus.is_running}
            >
              <option value="char">字符级</option>
              <option value="bpe">BPE</option>
            </select>
          </div>
          <div className="group">
            <label className="block text-[10px] sm:text-xs text-gray-400 mb-1 sm:mb-1.5 group-hover:text-blue-400 transition-colors">词表大小</label>
            <input
              type="number"
              value={trainingConfig.vocab_size}
              onChange={(e) => setTrainingConfig({...trainingConfig, vocab_size: parseInt(e.target.value)})}
              className="w-full px-2 sm:px-3 py-1.5 sm:py-2 bg-gray-700/50 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all hover:bg-gray-700 text-xs sm:text-sm"
              disabled={trainingStatus.is_running}
            />
          </div>
          <div className="group">
            <label className="block text-[10px] sm:text-xs text-gray-400 mb-1 sm:mb-1.5 group-hover:text-blue-400 transition-colors">模型维度</label>
            <input
              type="number"
              value={trainingConfig.d_model}
              onChange={(e) => setTrainingConfig({...trainingConfig, d_model: parseInt(e.target.value)})}
              className="w-full px-2 sm:px-3 py-1.5 sm:py-2 bg-gray-700/50 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all hover:bg-gray-700 text-xs sm:text-sm"
              disabled={trainingStatus.is_running}
            />
          </div>
          <div className="group">
            <label className="block text-[10px] sm:text-xs text-gray-400 mb-1 sm:mb-1.5 group-hover:text-blue-400 transition-colors">层数</label>
            <input
              type="number"
              value={trainingConfig.num_layers}
              onChange={(e) => setTrainingConfig({...trainingConfig, num_layers: parseInt(e.target.value)})}
              className="w-full px-2 sm:px-3 py-1.5 sm:py-2 bg-gray-700/50 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all hover:bg-gray-700 text-xs sm:text-sm"
              disabled={trainingStatus.is_running}
            />
          </div>
          <div className="group">
            <label className="block text-[10px] sm:text-xs text-gray-400 mb-1 sm:mb-1.5 group-hover:text-blue-400 transition-colors">Epochs</label>
            <input
              type="number"
              value={trainingConfig.epochs}
              onChange={(e) => setTrainingConfig({...trainingConfig, epochs: parseInt(e.target.value)})}
              className="w-full px-2 sm:px-3 py-1.5 sm:py-2 bg-gray-700/50 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all hover:bg-gray-700 text-xs sm:text-sm"
              disabled={trainingStatus.is_running}
            />
          </div>
          <div className="group">
            <label className="block text-[10px] sm:text-xs text-gray-400 mb-1 sm:mb-1.5 group-hover:text-blue-400 transition-colors">Batch Size</label>
            <input
              type="number"
              value={trainingConfig.batch_size}
              onChange={(e) => setTrainingConfig({...trainingConfig, batch_size: parseInt(e.target.value)})}
              className="w-full px-2 sm:px-3 py-1.5 sm:py-2 bg-gray-700/50 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all hover:bg-gray-700 text-xs sm:text-sm"
              disabled={trainingStatus.is_running}
            />
          </div>
          <div className="group">
            <label className="block text-[10px] sm:text-xs text-gray-400 mb-1 sm:mb-1.5 group-hover:text-blue-400 transition-colors">学习率</label>
            <input
              type="number"
              step="0.0001"
              value={trainingConfig.lr}
              onChange={(e) => setTrainingConfig({...trainingConfig, lr: parseFloat(e.target.value)})}
              className="w-full px-2 sm:px-3 py-1.5 sm:py-2 bg-gray-700/50 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all hover:bg-gray-700 text-xs sm:text-sm"
              disabled={trainingStatus.is_running}
            />
          </div>
          <div className="group">
            <label className="block text-[10px] sm:text-xs text-gray-400 mb-1 sm:mb-1.5 group-hover:text-blue-400 transition-colors">Dropout</label>
            <input
              type="number"
              step="0.05"
              value={trainingConfig.dropout}
              onChange={(e) => setTrainingConfig({...trainingConfig, dropout: parseFloat(e.target.value)})}
              className="w-full px-2 sm:px-3 py-1.5 sm:py-2 bg-gray-700/50 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all hover:bg-gray-700 text-xs sm:text-sm"
              disabled={trainingStatus.is_running}
            />
          </div>
          <div className="group">
            <label className="block text-[10px] sm:text-xs text-gray-400 mb-1 sm:mb-1.5 group-hover:text-blue-400 transition-colors">Patience (早停)</label>
            <input
              type="number"
              value={trainingConfig.patience}
              onChange={(e) => setTrainingConfig({...trainingConfig, patience: parseInt(e.target.value)})}
              className="w-full px-2 sm:px-3 py-1.5 sm:py-2 bg-gray-700/50 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all hover:bg-gray-700 text-xs sm:text-sm"
              disabled={trainingStatus.is_running}
            />
          </div>
          <div className="group">
            <label className="block text-[10px] sm:text-xs text-gray-400 mb-1 sm:mb-1.5 group-hover:text-blue-400 transition-colors">注意力头数</label>
            <input
              type="number"
              value={trainingConfig.num_heads}
              onChange={(e) => setTrainingConfig({...trainingConfig, num_heads: parseInt(e.target.value)})}
              className="w-full px-2 sm:px-3 py-1.5 sm:py-2 bg-gray-700/50 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all hover:bg-gray-700 text-xs sm:text-sm"
              disabled={trainingStatus.is_running}
            />
          </div>
          <div className="group">
            <label className="block text-[10px] sm:text-xs text-gray-400 mb-1 sm:mb-1.5 group-hover:text-blue-400 transition-colors">梯度累积</label>
            <input
              type="number"
              value={trainingConfig.grad_accum}
              onChange={(e) => setTrainingConfig({...trainingConfig, grad_accum: parseInt(e.target.value)})}
              className="w-full px-2 sm:px-3 py-1.5 sm:py-2 bg-gray-700/50 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all hover:bg-gray-700 text-xs sm:text-sm"
              disabled={trainingStatus.is_running}
            />
          </div>
          <div className="group">
            <label className="block text-[10px] sm:text-xs text-gray-400 mb-1 sm:mb-1.5 group-hover:text-blue-400 transition-colors">预热步数</label>
            <input
              type="number"
              value={trainingConfig.warmup_steps}
              onChange={(e) => setTrainingConfig({...trainingConfig, warmup_steps: parseInt(e.target.value)})}
              className="w-full px-2 sm:px-3 py-1.5 sm:py-2 bg-gray-700/50 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all hover:bg-gray-700 text-xs sm:text-sm"
              disabled={trainingStatus.is_running}
            />
          </div>
        </div>

        {/* 训练控制按钮 - 根据当前训练状态显示开始或停止按钮 */}
        <div className="flex gap-3 mt-4 sm:mt-6">
          {!trainingStatus.is_running ? (
            <button
              onClick={startTraining}
              disabled={loading}
              className="flex-1 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-semibold py-2.5 sm:py-3 px-4 sm:px-6 rounded-lg transition-all transform hover:scale-[1.02] active:scale-[0.98] shadow-lg disabled:shadow-none text-sm sm:text-base"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-4 w-4 sm:h-5 sm:w-5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <span className="hidden xs:inline">启动中...</span>
                  <span className="xs:hidden">启动</span>
                </span>
              ) : (
                <span className="flex items-center justify-center gap-2">
                  🚀 <span className="hidden xs:inline">开始训练</span><span className="xs:hidden">训练</span>
                </span>
              )}
            </button>
          ) : (
            <button
              onClick={stopTraining}
              className="flex-1 bg-gradient-to-r from-red-600 to-rose-600 hover:from-red-700 hover:to-rose-700 text-white font-semibold py-2.5 sm:py-3 px-4 sm:px-6 rounded-lg transition-all transform hover:scale-[1.02] active:scale-[0.98] shadow-lg text-sm sm:text-base"
            >
              <span className="flex items-center justify-center gap-2">
                ⏹️ <span className="hidden xs:inline">停止训练</span><span className="xs:hidden">停止</span>
              </span>
            </button>
          )}
        </div>
      </div>

      {/* 训练进度显示区域 - 当训练正在运行时显示进度条和当前状态 */}
      {trainingStatus.is_running && (
        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-4 sm:p-6 border border-gray-700 shadow-lg backdrop-blur-sm">
          <h2 className="text-xl sm:text-2xl font-bold mb-4 sm:mb-6 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent flex items-center gap-2">
            <span className="text-2xl sm:text-3xl">📈</span>
            <span className="hidden xs:inline">训练进度</span>
            <span className="xs:hidden">进度</span>
          </h2>
          <div className="mb-4 sm:mb-6">
            <div className="flex justify-between text-xs sm:text-sm text-gray-400 mb-2">
              <span className="font-medium">总体进度</span>
              <span className="font-mono text-blue-400">{Math.round(trainingStatus.progress * 100)}%</span>
            </div>
            <div className="w-full bg-gray-700/50 rounded-full h-3 overflow-hidden">
              <div
                className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-500 ease-out relative"
                style={{ width: `${trainingStatus.progress * 100}%` }}
              >
                <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
              </div>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-700/30 rounded-xl p-4 border border-gray-600/50 hover:border-blue-500/50 transition-all">
              <span className="text-sm text-gray-400 block mb-2">当前 Epoch</span>
              <p className="text-3xl font-bold text-white font-mono">{trainingStatus.current_epoch || 0} <span className="text-lg text-gray-500">/ {trainingConfig.epochs}</span></p>
            </div>
            <div className="bg-gray-700/30 rounded-xl p-4 border border-gray-600/50 hover:border-green-500/50 transition-all">
              <span className="text-sm text-gray-400 block mb-2">当前 Loss</span>
              <p className="text-3xl font-bold text-green-400 font-mono">{trainingStatus.current_loss?.toFixed(4) || '-'}</p>
            </div>
          </div>
        </div>
      )}

      {/* 损失曲线图表 - 当有损失历史数据时显示训练和验证损失曲线 */}
      {lossHistory.length > 0 && (
        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-6 border border-gray-700 shadow-lg backdrop-blur-sm">
          <h2 className="text-2xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent flex items-center gap-2">
            <span className="text-3xl">📉</span>
            损失曲线
          </h2>
          <ResponsiveContainer width="100%" height={350}>
            <AreaChart data={lossHistory}>
              <defs>
                <linearGradient id="trainGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="valGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.5} />
              <XAxis dataKey="epoch" stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)' }}
                labelStyle={{ color: '#f3f4f6', fontWeight: 'bold' }}
                itemStyle={{ color: '#e5e7eb' }}
              />
              <Legend />
              <Area type="monotone" dataKey="train_loss" stroke="#3b82f6" fill="url(#trainGradient)" name="Train Loss" strokeWidth={2} />
              <Area type="monotone" dataKey="val_loss" stroke="#10b981" fill="url(#valGradient)" name="Val Loss" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* 训练日志显示区域 - 显示训练过程中的各类消息和事件 */}
      {logs.length > 0 && (
        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-4 sm:p-6 border border-gray-700 shadow-lg backdrop-blur-sm">
          <div className="flex items-center justify-between mb-3 sm:mb-4">
            <h2 className="text-xl sm:text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent flex items-center gap-2">
              <span className="text-2xl sm:text-3xl">📋</span>
              <span className="hidden xs:inline">训练日志</span>
              <span className="xs:hidden">日志</span>
            </h2>
            <div className="flex items-center gap-1.5 sm:gap-2 text-[10px] sm:text-xs text-gray-500">
              <span>{logs.length}/{MAX_LOGS}</span>
              {logs.length >= MAX_LOGS && (
                <span className="text-yellow-500 hidden sm:inline">(旧日志已清理)</span>
              )}
            </div>
          </div>
          <div 
            ref={logsContainerRef}
            className="bg-gray-950/80 rounded-xl p-3 sm:p-4 h-48 sm:h-64 overflow-y-auto font-mono text-[10px] sm:text-xs border border-gray-800 scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-gray-900"
          >
            {logs.map((log, idx) => (
              <div key={idx} className="text-gray-300 mb-1 sm:mb-1.5 hover:bg-gray-800/50 px-1.5 sm:px-2 py-0.5 sm:py-1 rounded transition-colors break-words">
                <span className="text-gray-600 font-medium">[{log.time}]</span>{' '}
                <span className={
                  log.message.includes('✅') ? 'text-green-400' :
                  log.message.includes('❌') ? 'text-red-400' :
                  log.message.includes('⚠️') ? 'text-yellow-400' :
                  log.message.includes('🚀') ? 'text-blue-400' :
                  'text-gray-300'
                }>
                  {log.message}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default TrainingMonitor