import React, { useState, useEffect } from 'react'
import axios from 'axios'

/**
 * Dashboard组件 - 显示系统状态和监控数据
 * @param {string} apiUrl - API服务的基础URL地址
 * @param {Object} status - 包含系统状态信息的对象，包括模型加载状态、训练状态等
 * @returns {JSX.Element} 渲染的仪表板界面组件
 */
const Dashboard = ({ apiUrl, status }) => {
  /**
   * 系统统计信息状态
   * @type {Object} 包含CPU、内存、GPU内存和磁盘使用率百分比
   */
  const [systemStats, setSystemStats] = useState({
    cpu_percent: 0,
    memory_percent: 0,
    gpu_memory: 0,
    disk_usage: 0
  })

  /**
   * 设置定时器每5秒获取一次系统统计数据
   */
  useEffect(() => {
    const interval = setInterval(fetchSystemStats, 5000)
    // 立即获取一次
    fetchSystemStats()
    return () => clearInterval(interval)
  }, [apiUrl])

  /**
   * 异步获取系统统计信息
   * 尝试从API获取实时数据，如果后端不支持则显示N/A
   */
  const fetchSystemStats = async () => {
    try {
      // 获取系统信息（需要后端支持）
      const res = await axios.get(`${apiUrl}/system/stats`, { timeout: 2000 })
      setSystemStats(res.data)
    } catch (error) {
      // 后端不支持此接口时，显示N/A而不是假数据
      setSystemStats({
        cpu_percent: null,
        memory_percent: null,
        gpu_memory: null,
        disk_usage: null
      })
    }
  }

  /**
   * 格式化数据显示：处理null值显示为"N/A"
   */
  const formatValue = (value) => {
    if (value === null || value === undefined) {
      return 'N/A'
    }
    return `${Math.round(value)}%`
  }

  /**
   * 图表数据格式化：将系统统计信息转换为饼图可用的数据结构
   * 过滤掉null值
   * @type {Array} 包含名称、数值和颜色的图表数据数组
   */
  const chartData = [
    { name: 'CPU', value: systemStats.cpu_percent, color: '#3b82f6' },
    { name: '内存', value: systemStats.memory_percent, color: '#10b981' },
    { name: 'GPU', value: systemStats.gpu_memory, color: '#8b5cf6' },
    { name: '磁盘', value: systemStats.disk_usage, color: '#f59e0b' }
  ].filter(item => item.value !== null)

  return (
    <div className="space-y-6">
      {/* 状态卡片区域 - 显示模型、训练、API和设备状态 */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-4 sm:p-5 border border-gray-700 shadow-lg hover:border-blue-500/50 transition-all group">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-xs sm:text-sm group-hover:text-blue-400 transition-colors">模型状态</p>
              <p className={`text-xl sm:text-2xl font-bold mt-1 sm:mt-2 ${status.model_loaded ? 'text-green-400' : 'text-red-400'}`}>
                {status.model_loaded ? '✅ 已加载' : '❌ 未加载'}
              </p>
            </div>
            <div className="text-3xl sm:text-4xl opacity-80 group-hover:opacity-100 transition-opacity">🤖</div>
          </div>
        </div>
        
        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-4 sm:p-5 border border-gray-700 shadow-lg hover:border-green-500/50 transition-all group">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-xs sm:text-sm group-hover:text-green-400 transition-colors">训练状态</p>
              <p className="text-xl sm:text-2xl font-bold mt-1 sm:mt-2 text-yellow-400">
                {status.training_active ? '🏃 运行中' : '⏸️ 空闲'}
              </p>
            </div>
            <div className="text-3xl sm:text-4xl opacity-80 group-hover:opacity-100 transition-opacity">📊</div>
          </div>
        </div>
        
        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-4 sm:p-5 border border-gray-700 shadow-lg hover:border-purple-500/50 transition-all group">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-xs sm:text-sm group-hover:text-purple-400 transition-colors">API 状态</p>
              <p className="text-xl sm:text-2xl font-bold mt-1 sm:mt-2 text-green-400">🟢 在线</p>
            </div>
            <div className="text-3xl sm:text-4xl opacity-80 group-hover:opacity-100 transition-opacity">🌐</div>
          </div>
        </div>
        
        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-4 sm:p-5 border border-gray-700 shadow-lg hover:border-orange-500/50 transition-all group">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-xs sm:text-sm group-hover:text-orange-400 transition-colors">设备</p>
              <p className="text-xl sm:text-2xl font-bold mt-1 sm:mt-2 text-white">{status.model_info?.device || 'CPU'}</p>
            </div>
            <div className="text-3xl sm:text-4xl opacity-80 group-hover:opacity-100 transition-opacity">💻</div>
          </div>
        </div>
      </div>

      {/* 系统资源使用情况可视化区域 */}
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-4 sm:p-6 border border-gray-700 shadow-lg">
        <h2 className="text-xl sm:text-2xl font-bold mb-4 sm:mb-6 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent flex items-center gap-2">
          <span className="text-2xl sm:text-3xl">📊</span>
          <span className="hidden xs:inline">系统资源监控</span>
          <span className="xs:hidden">资源监控</span>
        </h2>
        
        {chartData.length === 0 ? (
          <div className="text-center py-8 sm:py-12">
            <div className="text-5xl sm:text-6xl mb-3 sm:mb-4 opacity-50">⚠️</div>
            <p className="text-base sm:text-lg text-gray-400 font-medium">系统监控功能未启用</p>
            <p className="text-xs sm:text-sm text-gray-500 mt-2 px-4">后端需要提供 /api/system/stats 接口才能显示实时数据</p>
            <p className="text-xs text-gray-600 mt-1">当前仅显示模型和训练状态信息</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-6">
            {chartData.map((item) => (
              <div key={item.name} className="group">
                <div className="flex justify-between text-xs sm:text-sm text-gray-400 mb-2 group-hover:text-gray-300 transition-colors">
                  <span className="font-medium">{item.name} 使用率</span>
                  <span className="font-mono" style={{ color: item.color }}>{formatValue(item.value)}</span>
                </div>
                <div className="w-full bg-gray-700/50 rounded-full h-2.5 sm:h-3 overflow-hidden">
                  <div
                    className="h-2.5 sm:h-3 rounded-full transition-all duration-500 ease-out relative"
                    style={{ 
                      width: item.value !== null ? `${item.value}%` : '0%',
                      backgroundColor: item.color 
                    }}
                  >
                    <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 模型详细信息展示区域 - 仅在模型已加载时显示 */}
      {status.model_loaded && (
        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-4 sm:p-6 border border-gray-700 shadow-lg">
          <h2 className="text-xl sm:text-2xl font-bold mb-4 sm:mb-6 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent flex items-center gap-2">
            <span className="text-2xl sm:text-3xl">🤖</span>
            <span className="hidden xs:inline">模型详情</span>
            <span className="xs:hidden">模型信息</span>
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4">
            <div className="bg-gray-700/30 rounded-xl p-3 sm:p-4 border border-gray-600/50 hover:border-blue-500/50 transition-all">
              <span className="text-xs text-gray-400 block mb-1 sm:mb-2">参数量</span>
              <p className="text-lg sm:text-2xl font-bold text-blue-400 font-mono">{status.model_info?.parameters?.toLocaleString() || '-'}</p>
            </div>
            <div className="bg-gray-700/30 rounded-xl p-3 sm:p-4 border border-gray-600/50 hover:border-green-500/50 transition-all">
              <span className="text-xs text-gray-400 block mb-1 sm:mb-2">模型大小</span>
              <p className="text-lg sm:text-2xl font-bold text-green-400 font-mono">{status.model_info?.size_mb?.toFixed(2) || '-'} MB</p>
            </div>
            <div className="bg-gray-700/30 rounded-xl p-3 sm:p-4 border border-gray-600/50 hover:border-purple-500/50 transition-all">
              <span className="text-xs text-gray-400 block mb-1 sm:mb-2">词表大小</span>
              <p className="text-lg sm:text-2xl font-bold text-purple-400 font-mono">{status.model_info?.vocab_size || '-'}</p>
            </div>
            <div className="bg-gray-700/30 rounded-xl p-3 sm:p-4 border border-gray-600/50 hover:border-orange-500/50 transition-all">
              <span className="text-xs text-gray-400 block mb-1 sm:mb-2">层数</span>
              <p className="text-lg sm:text-2xl font-bold text-orange-400 font-mono">{status.model_info?.num_layers || '-'}</p>
            </div>
          </div>
        </div>
      )}

      {/* 时间戳显示最后更新时间 */}
      <div className="text-center text-gray-500 text-xs">
        最后更新: {status.timestamp ? new Date(status.timestamp).toLocaleString() : '-'}
      </div>
    </div>
  )
}

export default Dashboard