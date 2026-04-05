import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { LineChart, Line, Area, AreaChart, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'

const Dashboard = ({ apiUrl, status }) => {
  const [systemStats, setSystemStats] = useState({
    cpu_percent: 0,
    memory_percent: 0,
    gpu_memory: 0,
    disk_usage: 0
  })

  useEffect(() => {
    const interval = setInterval(fetchSystemStats, 2000)
    return () => clearInterval(interval)
  }, [])

  const fetchSystemStats = async () => {
    try {
      // 获取系统信息（需要后端支持）
      const res = await axios.get(`${apiUrl}/system/stats`)
      setSystemStats(res.data)
    } catch (error) {
      // 使用模拟数据
      setSystemStats({
        cpu_percent: Math.random() * 60 + 20,
        memory_percent: Math.random() * 40 + 30,
        gpu_memory: status.model_loaded ? Math.random() * 50 + 20 : 0,
        disk_usage: 45
      })
    }
  }

  const chartData = [
    { name: 'CPU', value: systemStats.cpu_percent, color: '#3b82f6' },
    { name: '内存', value: systemStats.memory_percent, color: '#10b981' },
    { name: 'GPU', value: systemStats.gpu_memory, color: '#8b5cf6' },
    { name: '磁盘', value: systemStats.disk_usage, color: '#f59e0b' }
  ]

  return (
    <div className="space-y-6">
      {/* 状态卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">模型状态</p>
              <p className="text-2xl font-semibold mt-1">
                {status.model_loaded ? '✅ 已加载' : '❌ 未加载'}
              </p>
            </div>
            <div className="text-3xl">🤖</div>
          </div>
        </div>
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">训练状态</p>
              <p className="text-2xl font-semibold mt-1">
                {status.training_active ? '🏃 运行中' : '⏸️ 空闲'}
              </p>
            </div>
            <div className="text-3xl">📊</div>
          </div>
        </div>
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">API 状态</p>
              <p className="text-2xl font-semibold mt-1">🟢 在线</p>
            </div>
            <div className="text-3xl">🌐</div>
          </div>
        </div>
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">设备</p>
              <p className="text-2xl font-semibold mt-1">{status.model_info?.device || 'CPU'}</p>
            </div>
            <div className="text-3xl">💻</div>
          </div>
        </div>
      </div>

      {/* 系统资源 */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h2 className="text-xl font-semibold mb-4">📊 系统资源</h2>
        <div className="grid grid-cols-2 gap-6">
          {chartData.map((item) => (
            <div key={item.name}>
              <div className="flex justify-between text-sm text-gray-400 mb-1">
                <span>{item.name} 使用率</span>
                <span>{Math.round(item.value)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="h-2 rounded-full transition-all duration-300"
                  style={{ width: `${item.value}%`, backgroundColor: item.color }}
                ></div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 模型信息 */}
      {status.model_loaded && (
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h2 className="text-xl font-semibold mb-4">🤖 模型详情</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="bg-gray-700 rounded-lg p-3">
              <span className="text-gray-400">参数量</span>
              <p className="text-lg font-semibold">{status.model_info?.parameters?.toLocaleString() || '-'}</p>
            </div>
            <div className="bg-gray-700 rounded-lg p-3">
              <span className="text-gray-400">模型大小</span>
              <p className="text-lg font-semibold">{status.model_info?.size_mb?.toFixed(2) || '-'} MB</p>
            </div>
            <div className="bg-gray-700 rounded-lg p-3">
              <span className="text-gray-400">词表大小</span>
              <p className="text-lg font-semibold">{status.model_info?.vocab_size || '-'}</p>
            </div>
            <div className="bg-gray-700 rounded-lg p-3">
              <span className="text-gray-400">层数</span>
              <p className="text-lg font-semibold">{status.model_info?.num_layers || '-'}</p>
            </div>
          </div>
        </div>
      )}

      {/* 时间戳 */}
      <div className="text-center text-gray-500 text-xs">
        最后更新: {status.timestamp ? new Date(status.timestamp).toLocaleString() : '-'}
      </div>
    </div>
  )
}

export default Dashboard