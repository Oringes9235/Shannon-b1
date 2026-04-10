import React, { useState } from 'react'

/**
 * 布局组件，提供应用的整体结构，包括导航栏、标签页和主内容区域
 * @param {React.ReactNode} children - 主要内容组件
 * @param {string} activeTab - 当前激活的标签页ID
 * @param {Function} setActiveTab - 设置当前激活标签页的回调函数
 * @param {string} apiUrl - API服务地址
 * @param {Object} status - API状态信息对象
 * @returns {JSX.Element} 布局组件的JSX元素
 */
const Layout = ({ children, activeTab, setActiveTab, apiUrl, status }) => {
  // 侧边栏开关状态
  const [sidebarOpen, setSidebarOpen] = useState(false)

  // 定义导航标签页配置数组
  const tabs = [
    { id: 'dashboard', name: '仪表盘', icon: '📊', color: 'from-blue-500 to-cyan-500' },
    { id: 'generate', name: '文本生成', icon: '✍️', color: 'from-purple-500 to-pink-500' },
    { id: 'training', name: '训练监控', icon: '📈', color: 'from-green-500 to-emerald-500' },
    { id: 'models', name: '模型管理', icon: '📦', color: 'from-orange-500 to-red-500' },
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-gray-100">
      {/* 顶部导航栏 */}
      <nav className="bg-gray-900/80 backdrop-blur-xl border-b border-gray-700/50 sticky top-0 z-40 shadow-lg h-14 sm:h-16">
        <div className="max-w-7xl mx-auto px-3 sm:px-6 lg:px-8 h-full">
          <div className="flex items-center justify-between h-full">
            {/* 左侧：汉堡菜单按钮 + Logo */}
            <div className="flex items-center space-x-3 sm:space-x-4">
              {/* 汉堡菜单按钮 */}
              <button
                onClick={() => setSidebarOpen(true)}
                className="p-2 rounded-lg bg-gray-800/50 hover:bg-gray-700/50 transition-colors"
                aria-label="打开菜单"
              >
                <svg className="w-5 h-5 sm:w-6 sm:h-6 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>

              {/* Logo 区域 */}
              <div className="flex items-center space-x-2 sm:space-x-3">
                <div className="relative">
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg blur opacity-75 animate-pulse"></div>
                  <div className="relative w-8 h-8 sm:w-10 sm:h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center text-white font-bold text-base sm:text-lg shadow-lg">
                    S
                  </div>
                </div>
                <div>
                  <h1 className="text-base sm:text-xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                    Shannon-b1
                  </h1>
                  <p className="text-[10px] sm:text-xs text-gray-500 hidden sm:block">AI Language Model Workstation</p>
                </div>
              </div>
            </div>
            
            {/* 右侧操作区 */}
            <div className="flex items-center space-x-2 sm:space-x-4">
              {/* 状态指示器 - 桌面端显示文字 */}
              <div className="hidden md:flex items-center gap-2 px-3 py-1.5 bg-gray-800/50 rounded-full border border-gray-700">
                <div className={`w-2.5 h-2.5 rounded-full ${status?.status === 'running' ? 'bg-green-500 shadow-lg shadow-green-500/50' : 'bg-red-500'} ${status?.status === 'running' ? 'animate-pulse' : ''}`}></div>
                <span className="text-xs font-medium text-gray-300">
                  {status?.status === 'running' ? 'API Online' : 'API Offline'}
                </span>
              </div>
              
              {/* 移动端只显示状态点 */}
              <div className="md:hidden flex items-center">
                <div className={`w-2.5 h-2.5 rounded-full ${status?.status === 'running' ? 'bg-green-500 shadow-lg shadow-green-500/50' : 'bg-red-500'} ${status?.status === 'running' ? 'animate-pulse' : ''}`}></div>
              </div>
              
              {/* 快速加载按钮 */}
              <button
                onClick={async () => {
                  try {
                    const res = await fetch(`${apiUrl}/model/load?model_path=../../checkpoints/shannon_b1.pt`, { method: 'POST' })
                    const data = await res.json()
                    if (res.ok) {
                      const notification = document.createElement('div')
                      notification.className = 'fixed top-20 right-4 bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg z-50 animate-slide-in'
                      notification.textContent = '✅ 模型加载成功'
                      document.body.appendChild(notification)
                      setTimeout(() => notification.remove(), 3000)
                    } else {
                      alert('加载失败: ' + (data.detail || JSON.stringify(data)))
                    }
                  } catch (e) {
                    alert('加载模型时出错: ' + e.message)
                  }
                }}
                className="text-xs sm:text-sm bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white py-1.5 px-3 sm:py-2 sm:px-4 rounded-lg transition-all transform hover:scale-105 shadow-lg whitespace-nowrap"
              >
                ⚡ 快速加载
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* 左侧滑出式侧边栏 */}
      {/* 遮罩层 */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 transition-opacity animate-fade-in"
          onClick={() => setSidebarOpen(false)}
        ></div>
      )}

      {/* 侧边栏内容 */}
      <div className={`fixed top-0 left-0 h-full w-72 sm:w-80 bg-gradient-to-b from-gray-900 via-gray-800 to-gray-900 border-r border-gray-700/50 shadow-2xl z-50 transform transition-transform duration-300 ease-in-out ${
        sidebarOpen ? 'translate-x-0' : '-translate-x-full'
      }`}>
        {/* 侧边栏头部 */}
        <div className="flex items-center justify-between p-4 sm:p-6 border-b border-gray-700/50">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg blur opacity-75"></div>
              <div className="relative w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center text-white font-bold text-lg shadow-lg">
                S
              </div>
            </div>
            <div>
              <h2 className="text-lg font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                Shannon-b1
              </h2>
              <p className="text-xs text-gray-500">Workstation Menu</p>
            </div>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="p-2 rounded-lg hover:bg-gray-700/50 transition-colors"
            aria-label="关闭菜单"
          >
            <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* 侧边栏导航内容 */}
        <div className="flex flex-col h-full overflow-y-auto">
          {/* 导航标签 */}
          <div className="p-4 sm:p-6 space-y-2">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">导航</h3>
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => {
                  setActiveTab(tab.id)
                  setSidebarOpen(false)
                }}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg font-medium transition-all group ${
                  activeTab === tab.id
                    ? 'bg-gray-800 text-white shadow-lg'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
                }`}
              >
                {activeTab === tab.id && (
                  <div className={`absolute left-0 w-1 h-8 bg-gradient-to-b ${tab.color} rounded-r-full`}></div>
                )}
                <span className="text-xl">{tab.icon}</span>
                <span>{tab.name}</span>
                {activeTab === tab.id && (
                  <div className={`ml-auto w-2 h-2 rounded-full bg-gradient-to-r ${tab.color}`}></div>
                )}
              </button>
            ))}
          </div>

          {/* 分隔线 */}
          <div className="border-t border-gray-700/50 my-2"></div>

          {/* 快捷操作 */}
          <div className="p-4 sm:p-6 space-y-3">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">快捷操作</h3>
            
            {/* 快速加载模型 */}
            <button
              onClick={async () => {
                try {
                  const res = await fetch(`${apiUrl}/model/load?model_path=../../checkpoints/shannon_b1.pt`, { method: 'POST' })
                  const data = await res.json()
                  if (res.ok) {
                    const notification = document.createElement('div')
                    notification.className = 'fixed top-20 right-4 bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg z-50 animate-slide-in'
                    notification.textContent = '✅ 模型加载成功'
                    document.body.appendChild(notification)
                    setTimeout(() => notification.remove(), 3000)
                  } else {
                    alert('加载失败: ' + (data.detail || JSON.stringify(data)))
                  }
                } catch (e) {
                  alert('加载模型时出错: ' + e.message)
                }
                setSidebarOpen(false)
              }}
              className="w-full flex items-center gap-3 px-4 py-3 bg-gradient-to-r from-blue-600/20 to-purple-600/20 hover:from-blue-600/30 hover:to-purple-600/30 border border-blue-500/30 rounded-lg text-blue-400 hover:text-blue-300 transition-all"
            >
              <span className="text-xl">⚡</span>
              <span className="font-medium">快速加载模型</span>
            </button>

            {/* 其他快捷操作占位 */}
            <button className="w-full flex items-center gap-3 px-4 py-3 bg-gray-800/30 hover:bg-gray-800/50 border border-gray-700/50 rounded-lg text-gray-400 hover:text-gray-300 transition-all">
              <span className="text-xl">📝</span>
              <span className="font-medium">新建实验</span>
            </button>
            
            <button className="w-full flex items-center gap-3 px-4 py-3 bg-gray-800/30 hover:bg-gray-800/50 border border-gray-700/50 rounded-lg text-gray-400 hover:text-gray-300 transition-all">
              <span className="text-xl">📊</span>
              <span className="font-medium">查看报告</span>
            </button>
          </div>

          {/* 底部状态信息 */}
          <div className="mt-auto p-4 sm:p-6 border-t border-gray-700/50">
            <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-gray-400">API 状态</span>
                <div className={`w-2 h-2 rounded-full ${status?.status === 'running' ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
              </div>
              <p className="text-sm font-medium text-gray-300">
                {status?.status === 'running' ? '在线' : '离线'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* 主内容区域 */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
        <div className="animate-fade-in">
          {children}
        </div>
      </main>

      {/* 全局样式 */}
      <style jsx global>{`
        @keyframes slide-in {
          from {
            transform: translateX(100%);
            opacity: 0;
          }
          to {
            transform: translateX(0);
            opacity: 1;
          }
        }
        
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        .animate-slide-in {
          animation: slide-in 0.3s ease-out;
        }
        
        .animate-fade-in {
          animation: fade-in 0.3s ease-out;
        }
        
        /* 自定义滚动条 */
        ::-webkit-scrollbar {
          width: 8px;
          height: 8px;
        }
        
        ::-webkit-scrollbar-track {
          background: rgba(31, 41, 55, 0.5);
          border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
          background: rgba(75, 85, 99, 0.8);
          border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
          background: rgba(107, 114, 128, 1);
        }
      `}</style>
    </div>
  )
}

export default Layout
