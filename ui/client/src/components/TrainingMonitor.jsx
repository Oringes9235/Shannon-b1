import React, { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import { subscribe } from '../ws'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts'

/**
 * 训练监控组件 - 用于监控和控制机器学习模型的训练过程
 * 提供训练配置、实时进度监控、损失曲线可视化和训练日志功能
 * @param {Object} props - 组件属性
 * @param {string} props.apiUrl - API服务端点地址
 * @returns {JSX.Element} 训练监控界面组件
 */
const TrainingMonitor = ({ apiUrl }) => {
  // 初始化训练配置状态，包含模型架构和训练超参数
  const [trainingConfig, setTrainingConfig] = useState({
    tokenizer: 'char',
    vocab_size: 200,
    d_model: 128,
    num_layers: 3,
    epochs: 30,
    batch_size: 32,
    seq_len: 64,
    lr: 0.0005,
    dropout: 0.3,
    weight_decay: 0.1
  })

  // 训练状态：运行状态和进度信息
  const [trainingStatus, setTrainingStatus] = useState({ is_running: false, progress: 0 })

  // 损失历史记录，用于绘制损失曲线
  const [lossHistory, setLossHistory] = useState([])

  // 训练日志记录
  const [logs, setLogs] = useState([])

  // WebSocket连接管理
  const [ws, setWs] = useState(null)

  // 加载状态，用于控制按钮状态
  const [loading, setLoading] = useState(false)

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
  const handleWebSocketMessage = (data) => {
    if (data.type === 'training_progress') {
      setTrainingStatus(data.data)
    } else if (data.type === 'training_epoch_complete') {
      setLossHistory(prev => [...prev, {
        epoch: data.data.epoch,
        train_loss: data.data.train_loss,
        val_loss: data.data.val_loss
      }])
      setLogs(prev => [...prev, {
        time: new Date().toLocaleTimeString(),
        message: `Epoch ${data.data.epoch}: train_loss=${data.data.train_loss.toFixed(4)}, val_loss=${data.data.val_loss.toFixed(4)}`
      }])
    } else if (data.type === 'training_completed') {
      setTrainingStatus({ is_running: false, progress: 1 })
      setLoading(false)
      setLogs(prev => [...prev, {
        time: new Date().toLocaleTimeString(),
        message: `✅ Training completed! Best loss: ${data.data.best_loss.toFixed(4)}`
      }])
    } else if (data.type === 'training_error') {
      setLogs(prev => [...prev, {
        time: new Date().toLocaleTimeString(),
        message: `❌ Error: ${data.data.error}`
      }])
      setLoading(false)
    }
  }

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
        setLogs(prev => [...prev, {
          time: new Date().toLocaleTimeString(),
          message: '🚀 Training started...'
        }])
      } else {
        setLogs(prev => [...prev, {
          time: new Date().toLocaleTimeString(),
          message: `❌ Failed to start training: ${res.data?.message || JSON.stringify(res.data)}`
        }])
        setLoading(false)
        return
      }
    } catch (error) {
      console.error('Failed to start training:', error)
      setLogs(prev => [...prev, {
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
      setLogs(prev => [...prev, {
        time: new Date().toLocaleTimeString(),
        message: '⏹️ Training stop requested'
      }])
    } catch (error) {
      console.error('Failed to stop training:', error)
    }
  }

  return (
    <div className="space-y-6">
      {/* 训练配置部分 - 包含各种训练参数的输入控件 */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h2 className="text-xl font-semibold mb-4">📊 训练配置</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-xs text-gray-400 mb-1">分词器</label>
            <select
              value={trainingConfig.tokenizer}
              onChange={(e) => setTrainingConfig({...trainingConfig, tokenizer: e.target.value})}
              className="w-full px-3 py-1 bg-gray-700 border border-gray-600 rounded"
              disabled={trainingStatus.is_running}
            >
              <option value="char">字符级</option>
              <option value="bpe">BPE</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">词表大小</label>
            <input
              type="number"
              value={trainingConfig.vocab_size}
              onChange={(e) => setTrainingConfig({...trainingConfig, vocab_size: parseInt(e.target.value)})}
              className="w-full px-3 py-1 bg-gray-700 border border-gray-600 rounded"
              disabled={trainingStatus.is_running}
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">模型维度</label>
            <input
              type="number"
              value={trainingConfig.d_model}
              onChange={(e) => setTrainingConfig({...trainingConfig, d_model: parseInt(e.target.value)})}
              className="w-full px-3 py-1 bg-gray-700 border border-gray-600 rounded"
              disabled={trainingStatus.is_running}
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">层数</label>
            <input
              type="number"
              value={trainingConfig.num_layers}
              onChange={(e) => setTrainingConfig({...trainingConfig, num_layers: parseInt(e.target.value)})}
              className="w-full px-3 py-1 bg-gray-700 border border-gray-600 rounded"
              disabled={trainingStatus.is_running}
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Epochs</label>
            <input
              type="number"
              value={trainingConfig.epochs}
              onChange={(e) => setTrainingConfig({...trainingConfig, epochs: parseInt(e.target.value)})}
              className="w-full px-3 py-1 bg-gray-700 border border-gray-600 rounded"
              disabled={trainingStatus.is_running}
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Batch Size</label>
            <input
              type="number"
              value={trainingConfig.batch_size}
              onChange={(e) => setTrainingConfig({...trainingConfig, batch_size: parseInt(e.target.value)})}
              className="w-full px-3 py-1 bg-gray-700 border border-gray-600 rounded"
              disabled={trainingStatus.is_running}
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">学习率</label>
            <input
              type="number"
              step="0.0001"
              value={trainingConfig.lr}
              onChange={(e) => setTrainingConfig({...trainingConfig, lr: parseFloat(e.target.value)})}
              className="w-full px-3 py-1 bg-gray-700 border border-gray-600 rounded"
              disabled={trainingStatus.is_running}
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Dropout</label>
            <input
              type="number"
              step="0.05"
              value={trainingConfig.dropout}
              onChange={(e) => setTrainingConfig({...trainingConfig, dropout: parseFloat(e.target.value)})}
              className="w-full px-3 py-1 bg-gray-700 border border-gray-600 rounded"
              disabled={trainingStatus.is_running}
            />
          </div>
        </div>

        {/* 训练控制按钮 - 根据当前训练状态显示开始或停止按钮 */}
        <div className="flex gap-3 mt-6">
          {!trainingStatus.is_running ? (
            <button
              onClick={startTraining}
              disabled={loading}
              className="bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white font-medium py-2 px-6 rounded-lg transition-colors"
            >
              {loading ? '启动中...' : '🚀 开始训练'}
            </button>
          ) : (
            <button
              onClick={stopTraining}
              className="bg-red-600 hover:bg-red-700 text-white font-medium py-2 px-6 rounded-lg transition-colors"
            >
              ⏹️ 停止训练
            </button>
          )}
        </div>
      </div>

      {/* 训练进度显示区域 - 当训练正在运行时显示进度条和当前状态 */}
      {trainingStatus.is_running && (
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h2 className="text-xl font-semibold mb-4">📈 训练进度</h2>
          <div className="mb-4">
            <div className="flex justify-between text-sm text-gray-400 mb-1">
              <span>进度</span>
              <span>{Math.round(trainingStatus.progress * 100)}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${trainingStatus.progress * 100}%` }}
              ></div>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="bg-gray-700 rounded-lg p-3">
              <span className="text-gray-400">当前 Epoch</span>
              <p className="text-xl font-semibold">{trainingStatus.current_epoch || 0} / {trainingConfig.epochs}</p>
            </div>
            <div className="bg-gray-700 rounded-lg p-3">
              <span className="text-gray-400">当前 Loss</span>
              <p className="text-xl font-semibold">{trainingStatus.current_loss?.toFixed(4) || '-'}</p>
            </div>
          </div>
        </div>
      )}

      {/* 损失曲线图表 - 当有损失历史数据时显示训练和验证损失曲线 */}
      {lossHistory.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h2 className="text-xl font-semibold mb-4">📉 损失曲线</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={lossHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="epoch" stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1f2937', border: 'none' }}
                labelStyle={{ color: '#f3f4f6' }}
              />
              <Line type="monotone" dataKey="train_loss" stroke="#3b82f6" name="Train Loss" />
              <Line type="monotone" dataKey="val_loss" stroke="#10b981" name="Val Loss" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* 训练日志显示区域 - 显示训练过程中的各类消息和事件 */}
      {logs.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h2 className="text-xl font-semibold mb-4">📋 训练日志</h2>
          <div className="bg-gray-900 rounded-lg p-3 h-48 overflow-y-auto font-mono text-xs">
            {logs.map((log, idx) => (
              <div key={idx} className="text-gray-300 mb-1">
                <span className="text-gray-500">[{log.time}]</span> {log.message}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default TrainingMonitor