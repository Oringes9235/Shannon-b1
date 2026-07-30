import { useState, useEffect, useRef, useCallback } from 'react'
import axios from 'axios'
import { subscribe } from '../ws'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart, Legend } from 'recharts'
import {
  IconGear,
  IconRocket,
  IconStop,
  IconChart,
  IconChartDown,
  IconLog,
  IconCheck,
  IconX,
  IconWarning,
} from './Icons'

const TrainingMonitor = ({ apiUrl }) => {
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

  const initialConfig = savedConfig
    ? { ...defaultConfig, ...JSON.parse(savedConfig) }
    : defaultConfig

  const [trainingConfig, setTrainingConfig] = useState(initialConfig)

  useEffect(() => {
    localStorage.setItem('shannon_training_config', JSON.stringify(trainingConfig))
  }, [trainingConfig])

  const [trainingStatus, setTrainingStatus] = useState({ is_running: false, progress: 0 })
  const [lossHistory, setLossHistory] = useState([])
  const MAX_LOGS = 500
  const [logs, setLogs] = useState([])
  const [loading, setLoading] = useState(false)
  const logsContainerRef = useRef(null)

  useEffect(() => {
    if (logsContainerRef.current && logs.length > 0) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight
    }
  }, [logs])

  useEffect(() => {
    const unsub = subscribe((data) => handleWebSocketMessage(data), apiUrl)
    return () => unsub()
  }, [])

  useEffect(() => {
    const interval = setInterval(fetchTrainingStatus, 2000)
    return () => clearInterval(interval)
  }, [])

  const handleWebSocketMessage = useCallback((data) => {
    if (data.type === 'training_log') {
      setLogs(prev => {
        const newLogs = [...prev, {
          time: new Date().toLocaleTimeString(),
          message: data.data.line
        }]
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
          message: `Training completed! Best loss: ${data.data.best_loss?.toFixed(4) || 'N/A'}`
        }]
        return newLogs.length > MAX_LOGS ? newLogs.slice(-MAX_LOGS) : newLogs
      })
    } else if (data.type === 'training_error') {
      setLogs(prev => {
        const newLogs = [...prev, {
          time: new Date().toLocaleTimeString(),
          message: `Error: ${data.data.error || data.data.returncode}`
        }]
        return newLogs.length > MAX_LOGS ? newLogs.slice(-MAX_LOGS) : newLogs
      })
      setLoading(false)
    } else if (data.type === 'training_started') {
      setTrainingStatus(data.data)
    }
  }, [])

  const fetchTrainingStatus = async () => {
    try {
      const res = await axios.get(`${apiUrl}/train/status`)
      setTrainingStatus(res.data)
    } catch (error) {
      console.error('Failed to fetch training status:', error)
    }
  }

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
          message: 'Training started...'
        }])
      } else {
        setLogs(prev => [{
          time: new Date().toLocaleTimeString(),
          message: `Failed to start training: ${res.data?.message || JSON.stringify(res.data)}`
        }])
        setLoading(false)
        return
      }
    } catch (error) {
      console.error('Failed to start training:', error)
      setLogs(prev => [{
        time: new Date().toLocaleTimeString(),
        message: `Failed to start training: ${error.response?.data?.detail || error.message}`
      }])
      setLoading(false)
    }
  }

  const stopTraining = async () => {
    try {
      await axios.post(`${apiUrl}/train/stop`)
      setLogs(prev => {
        const newLogs = [...prev, {
          time: new Date().toLocaleTimeString(),
          message: 'Training stop requested'
        }]
        return newLogs.length > MAX_LOGS ? newLogs.slice(-MAX_LOGS) : newLogs
      })
    } catch (error) {
      console.error('Failed to stop training:', error)
    }
  }

  const configFields = [
    { key: 'tokenizer', label: 'Tokenizer', type: 'select', options: [{ value: 'char', label: 'Character' }, { value: 'bpe', label: 'BPE' }] },
    { key: 'vocab_size', label: 'Vocab Size', type: 'number' },
    { key: 'd_model', label: 'Model Dim', type: 'number' },
    { key: 'num_layers', label: 'Layers', type: 'number' },
    { key: 'epochs', label: 'Epochs', type: 'number' },
    { key: 'batch_size', label: 'Batch Size', type: 'number' },
    { key: 'lr', label: 'Learning Rate', type: 'number', step: 0.0001 },
    { key: 'dropout', label: 'Dropout', type: 'number', step: 0.05 },
    { key: 'patience', label: 'Patience (Early Stop)', type: 'number' },
    { key: 'num_heads', label: 'Attention Heads', type: 'number' },
    { key: 'grad_accum', label: 'Grad Accum', type: 'number' },
    { key: 'warmup_steps', label: 'Warmup Steps', type: 'number' },
  ]

  return (
    <div className="space-y-6">
      {/* Training config */}
      <div className="bg-[#0d1117] rounded-lg p-6 border border-[#21262d]">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-[#e6edf3] flex items-center gap-2">
            <IconGear className="w-5 h-5 text-[#8b949e]" />
            Training Configuration
          </h2>
          {trainingStatus.is_running && (
            <div className="flex items-center gap-1.5 px-3 py-1 bg-[#238636]/10 border border-[#3fb950]/30 rounded-full">
              <div className="w-2 h-2 bg-[#3fb950] rounded-full animate-pulse"></div>
              <span className="text-xs text-[#3fb950] font-medium">Training</span>
            </div>
          )}
        </div>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {configFields.map((field) => (
            <div key={field.key}>
              <label className="block text-[11px] text-[#8b949e] mb-1">{field.label}</label>
              {field.type === 'select' ? (
                <select
                  value={trainingConfig[field.key]}
                  onChange={(e) => setTrainingConfig({ ...trainingConfig, [field.key]: e.target.value })}
                  className="w-full px-2.5 py-1.5 bg-[#161b22] border border-[#30363d] rounded-md focus:ring-2 focus:ring-[#1f6feb] focus:border-transparent text-[#e6edf3] text-xs transition-base"
                  disabled={trainingStatus.is_running}
                >
                  {field.options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              ) : (
                <input
                  type="number"
                  step={field.step || 1}
                  value={trainingConfig[field.key]}
                  onChange={(e) => setTrainingConfig({
                    ...trainingConfig,
                    [field.key]: field.step ? parseFloat(e.target.value) : parseInt(e.target.value)
                  })}
                  className="w-full px-2.5 py-1.5 bg-[#161b22] border border-[#30363d] rounded-md focus:ring-2 focus:ring-[#1f6feb] focus:border-transparent text-[#e6edf3] text-xs transition-base"
                  disabled={trainingStatus.is_running}
                />
              )}
            </div>
          ))}
        </div>

        {/* Control buttons */}
        <div className="flex gap-3 mt-6">
          {!trainingStatus.is_running ? (
            <button
              onClick={startTraining}
              disabled={loading}
              className="flex-1 flex items-center justify-center gap-2 bg-[#1f6feb] hover:bg-[#388bfd] disabled:bg-[#21262d] disabled:cursor-not-allowed text-white font-semibold py-2.5 px-4 rounded-md transition-base text-sm"
            >
              {loading ? (
                <>
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Starting...
                </>
              ) : (
                <>
                  <IconRocket className="w-4 h-4" /> Start Training
                </>
              )}
            </button>
          ) : (
            <button
              onClick={stopTraining}
              className="flex-1 flex items-center justify-center gap-2 bg-[#da3633] hover:bg-[#f85149] text-white font-semibold py-2.5 px-4 rounded-md transition-base text-sm"
            >
              <IconStop className="w-4 h-4" /> Stop Training
            </button>
          )}
        </div>
      </div>

      {/* Training progress */}
      {trainingStatus.is_running && (
        <div className="bg-[#0d1117] rounded-lg p-6 border border-[#21262d]">
          <h2 className="text-lg font-semibold text-[#e6edf3] mb-5 flex items-center gap-2">
            <IconChart className="w-5 h-5 text-[#8b949e]" />
            Training Progress
          </h2>
          <div className="mb-6">
            <div className="flex justify-between text-xs text-[#8b949e] mb-2">
              <span className="font-medium">Overall Progress</span>
              <span className="font-mono text-[#58a6ff]">{Math.round(trainingStatus.progress * 100)}%</span>
            </div>
            <div className="w-full bg-[#21262d] rounded-full h-2.5 overflow-hidden">
              <div
                className="h-2.5 rounded-full transition-all duration-500 ease-out"
                style={{
                  width: `${trainingStatus.progress * 100}%`,
                  backgroundColor: '#58a6ff'
                }}
              />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-[#161b22] rounded-lg p-4 border border-[#21262d]">
              <span className="text-xs text-[#8b949e] block mb-1">Current Epoch</span>
              <p className="text-2xl font-semibold text-[#e6edf3] font-mono">
                {trainingStatus.current_epoch || 0}
                <span className="text-sm text-[#484f58]"> / {trainingConfig.epochs}</span>
              </p>
            </div>
            <div className="bg-[#161b22] rounded-lg p-4 border border-[#21262d]">
              <span className="text-xs text-[#8b949e] block mb-1">Current Loss</span>
              <p className="text-2xl font-semibold text-[#3fb950] font-mono">
                {trainingStatus.current_loss?.toFixed(4) || '-'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Loss chart */}
      {lossHistory.length > 0 && (
        <div className="bg-[#0d1117] rounded-lg p-6 border border-[#21262d]">
          <h2 className="text-lg font-semibold text-[#e6edf3] mb-5 flex items-center gap-2">
            <IconChartDown className="w-5 h-5 text-[#8b949e]" />
            Loss Curve
          </h2>
          <ResponsiveContainer width="100%" height={350}>
            <AreaChart data={lossHistory}>
              <defs>
                <linearGradient id="trainGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#58a6ff" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#58a6ff" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="valGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3fb950" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#3fb950" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#21262d" opacity={0.6} />
              <XAxis dataKey="epoch" stroke="#484f58" tick={{ fontSize: 12 }} />
              <YAxis stroke="#484f58" tick={{ fontSize: 12 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#161b22',
                  border: '1px solid #30363d',
                  borderRadius: '6px',
                  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)',
                  fontSize: '12px',
                }}
                labelStyle={{ color: '#e6edf3', fontWeight: 'bold' }}
                itemStyle={{ color: '#c9d1d9' }}
              />
              <Legend wrapperStyle={{ fontSize: '12px' }} />
              <Area type="monotone" dataKey="train_loss" stroke="#58a6ff" fill="url(#trainGradient)" name="Train Loss" strokeWidth={2} />
              <Area type="monotone" dataKey="val_loss" stroke="#3fb950" fill="url(#valGradient)" name="Val Loss" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Training logs */}
      {logs.length > 0 && (
        <div className="bg-[#0d1117] rounded-lg p-6 border border-[#21262d]">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-[#e6edf3] flex items-center gap-2">
              <IconLog className="w-5 h-5 text-[#8b949e]" />
              Training Logs
            </h2>
            <div className="flex items-center gap-2 text-[10px] text-[#484f58]">
              <span>{logs.length}/{MAX_LOGS}</span>
              {logs.length >= MAX_LOGS && (
                <span className="text-[#d29922]">(old logs purged)</span>
              )}
            </div>
          </div>
          <div
            ref={logsContainerRef}
            className="bg-[#0d1117] rounded-md p-4 h-64 overflow-y-auto font-mono text-xs border border-[#21262d]"
          >
            {logs.map((log, idx) => {
              let textColor = 'text-[#c9d1d9]'
              if (log.message.includes('completed') || log.message.includes('success')) textColor = 'text-[#3fb950]'
              else if (log.message.includes('Error') || log.message.includes('Failed') || log.message.includes('error')) textColor = 'text-[#f85149]'
              else if (log.message.includes('Warning') || log.message.includes('warn')) textColor = 'text-[#d29922]'
              else if (log.message.includes('started') || log.message.includes('Started')) textColor = 'text-[#58a6ff]'

              return (
                <div key={idx} className={`${textColor} mb-1 hover:bg-[#161b22] px-2 py-0.5 rounded transition-base break-words`}>
                  <span className="text-[#484f58] font-medium">[{log.time}]</span>{' '}
                  <span>{log.message}</span>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

export default TrainingMonitor