import React from 'react'

const Layout = ({ children, activeTab, setActiveTab, apiUrl, status }) => {
  const tabs = [
    { id: 'generate', name: '文本生成', icon: '✍️' },
    { id: 'training', name: '训练监控', icon: '📈' },
    { id: 'models', name: '模型管理', icon: '📦' },
    { id: 'dashboard', name: '仪表盘', icon: '📊' },
  ]

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      {/* 导航栏 */}
      <nav className="bg-gray-800 border-b border-gray-700 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <span className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                Shannon-b1
              </span>
              <span className="ml-2 text-sm text-gray-400">AI Language Model</span>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-1">
                <div className={`w-2 h-2 rounded-full ${status?.status === 'running' ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
                <span className="text-xs text-gray-400">{status?.status === 'running' ? 'API Online' : 'API Offline'}</span>
              </div>
              <div>
                <button
                  onClick={async () => {
                    try {
                      const res = await fetch(`${apiUrl}/model/load?model_path=../../checkpoints/shannon_b1.pt`, { method: 'POST' })
                      const data = await res.json()
                      if (res.ok) alert('模型加载成功')
                      else alert('加载失败: ' + (data.detail || JSON.stringify(data)))
                    } catch (e) {
                      alert('加载模型时出错: ' + e.message)
                    }
                  }}
                  className="text-sm bg-blue-600 hover:bg-blue-700 text-white py-1 px-3 rounded"
                >
                  快速加载模型
                </button>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* 标签栏 */}
      <div className="border-b border-gray-800 bg-gray-900/50 backdrop-blur sticky top-16 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8 overflow-x-auto">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-3 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-400'
                    : 'border-transparent text-gray-400 hover:text-gray-300'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.name}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* 主内容 */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>
    </div>
  )
}

export default Layout