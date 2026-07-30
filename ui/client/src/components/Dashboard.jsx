import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  IconRobot,
  IconChart,
  IconGlobe,
  IconTerminal,
  IconWarning,
  IconDatabase,
  IconFolder,
  IconGear,
  IconLog,
} from './Icons'

const Dashboard = ({ apiUrl, status }) => {
  const [systemStats, setSystemStats] = useState({
    cpu_percent: 0,
    memory_percent: 0,
    gpu_memory: 0,
    disk_usage: 0
  })

  useEffect(() => {
    const interval = setInterval(fetchSystemStats, 5000)
    fetchSystemStats()
    return () => clearInterval(interval)
  }, [apiUrl])

  const fetchSystemStats = async () => {
    try {
      const res = await axios.get(`${apiUrl}/system/stats`, { timeout: 2000 })
      setSystemStats(res.data)
    } catch (error) {
      setSystemStats({
        cpu_percent: null,
        memory_percent: null,
        gpu_memory: null,
        disk_usage: null
      })
    }
  }

  const formatValue = (value) => {
    if (value === null || value === undefined) return 'N/A'
    return `${Math.round(value)}%`
  }

  const chartData = [
    { name: 'CPU', value: systemStats.cpu_percent, color: '#58a6ff' },
    { name: 'Memory', value: systemStats.memory_percent, color: '#3fb950' },
    { name: 'GPU', value: systemStats.gpu_memory, color: '#a371f7' },
    { name: 'Disk', value: systemStats.disk_usage, color: '#d29922' }
  ].filter(item => item.value !== null)

  const statusCards = [
    {
      label: 'Model Status',
      value: status.model_loaded ? 'Loaded' : 'Not Loaded',
      color: status.model_loaded ? '#3fb950' : '#f85149',
      borderHover: '#58a6ff',
      Icon: IconRobot,
    },
    {
      label: 'Training Status',
      value: status.training_active ? 'Running' : 'Idle',
      color: status.training_active ? '#d29922' : '#8b949e',
      borderHover: '#3fb950',
      Icon: IconChart,
    },
    {
      label: 'API Status',
      value: 'Online',
      color: '#3fb950',
      borderHover: '#a371f7',
      Icon: IconGlobe,
    },
    {
      label: 'Device',
      value: status.model_info?.device || 'CPU',
      color: '#e6edf3',
      borderHover: '#d29922',
      Icon: IconTerminal,
    },
  ]

  return (
    <div className="space-y-6">
      {/* Status cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {statusCards.map((card, idx) => {
          const CardIcon = card.Icon
          return (
            <div
              key={idx}
              className="bg-[#0d1117] rounded-lg p-5 border border-[#21262d] hover:border-[#30363d] transition-base"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-[#8b949e] mb-1">{card.label}</p>
                  <p className="text-xl font-semibold" style={{ color: card.color }}>
                    {card.value}
                  </p>
                </div>
                <CardIcon className="w-8 h-8 text-[#30363d]" />
              </div>
            </div>
          )
        })}
      </div>

      {/* System resources */}
      <div className="bg-[#0d1117] rounded-lg p-6 border border-[#21262d]">
        <h2 className="text-lg font-semibold text-[#e6edf3] mb-5 flex items-center gap-2">
          <IconChart className="w-5 h-5 text-[#8b949e]" />
          System Resources
        </h2>

        {chartData.length === 0 ? (
          <div className="text-center py-12">
            <IconWarning className="w-12 h-12 mx-auto mb-3 opacity-40" />
            <p className="text-sm text-[#8b949e] font-medium">System monitoring not enabled</p>
            <p className="text-xs text-[#484f58] mt-1">Backend requires /api/system/stats endpoint for live data</p>
            <p className="text-xs text-[#484f58] mt-0.5">Currently showing model and training status only</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
            {chartData.map((item) => (
              <div key={item.name}>
                <div className="flex justify-between text-xs mb-2">
                  <span className="text-[#8b949e] font-medium">{item.name} Usage</span>
                  <span className="font-mono" style={{ color: item.color }}>{formatValue(item.value)}</span>
                </div>
                <div className="w-full bg-[#21262d] rounded-full h-2.5 overflow-hidden">
                  <div
                    className="h-2.5 rounded-full transition-all duration-500 ease-out"
                    style={{
                      width: item.value !== null ? `${item.value}%` : '0%',
                      backgroundColor: item.color
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Model details */}
      {status.model_loaded && (
        <div className="bg-[#0d1117] rounded-lg p-6 border border-[#21262d]">
          <h2 className="text-lg font-semibold text-[#e6edf3] mb-5 flex items-center gap-2">
            <IconDatabase className="w-5 h-5 text-[#8b949e]" />
            Model Details
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-[#161b22] rounded-lg p-4 border border-[#21262d]">
              <span className="text-xs text-[#8b949e] block mb-1">Parameters</span>
              <p className="text-xl font-semibold text-[#58a6ff] font-mono">
                {status.model_info?.parameters?.toLocaleString() || '-'}
              </p>
            </div>
            <div className="bg-[#161b22] rounded-lg p-4 border border-[#21262d]">
              <span className="text-xs text-[#8b949e] block mb-1">Model Size</span>
              <p className="text-xl font-semibold text-[#3fb950] font-mono">
                {status.model_info?.size_mb?.toFixed(2) || '-'} MB
              </p>
            </div>
            <div className="bg-[#161b22] rounded-lg p-4 border border-[#21262d]">
              <span className="text-xs text-[#8b949e] block mb-1">Vocab Size</span>
              <p className="text-xl font-semibold text-[#a371f7] font-mono">
                {status.model_info?.vocab_size || '-'}
              </p>
            </div>
            <div className="bg-[#161b22] rounded-lg p-4 border border-[#21262d]">
              <span className="text-xs text-[#8b949e] block mb-1">Layers</span>
              <p className="text-xl font-semibold text-[#d29922] font-mono">
                {status.model_info?.num_layers || '-'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Timestamp */}
      <div className="text-center text-[#484f58] text-xs">
        Last updated: {status.timestamp ? new Date(status.timestamp).toLocaleString() : '-'}
      </div>
    </div>
  )
}

export default Dashboard