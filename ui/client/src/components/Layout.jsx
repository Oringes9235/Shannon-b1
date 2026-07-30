import React, { useState, useEffect } from 'react'
import {
  IconDashboard,
  IconEdit,
  IconChart,
  IconPackage,
  IconLightning,
  IconX,
  IconGlobe,
} from './Icons'

const Layout = ({ children, activeTab, setActiveTab, apiUrl, status }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768)

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 768)
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  // Close sidebar when switching to non-mobile
  useEffect(() => {
    if (!isMobile) setSidebarOpen(false)
  }, [isMobile])

  const tabs = [
    { id: 'dashboard', name: 'Dashboard', icon: IconDashboard },
    { id: 'generate', name: 'Text Generator', icon: IconEdit },
    { id: 'training', name: 'Training', icon: IconChart },
    { id: 'models', name: 'Models', icon: IconPackage },
  ]

  const handleQuickLoad = async () => {
    try {
      const res = await fetch(`${apiUrl}/model/load?model_path=../../checkpoints/shannon_b1.pt`, { method: 'POST' })
      const data = await res.json()
      if (res.ok) {
        const n = document.createElement('div')
        n.className = 'fixed top-20 right-4 bg-[#238636] text-white px-6 py-3 rounded-lg shadow-lg z-50 animate-slide-in'
        n.textContent = 'Model loaded successfully'
        document.body.appendChild(n)
        setTimeout(() => n.remove(), 3000)
      } else {
        alert('Load failed: ' + (data.detail || JSON.stringify(data)))
      }
    } catch (e) {
      alert('Error loading model: ' + e.message)
    }
    setSidebarOpen(false)
  }

  const sidebarContent = (
    <div className="flex flex-col h-full">
      {/* Sidebar header */}
      <div className="flex items-center justify-between p-4 border-b border-[#21262d]">
        <div className="flex items-center gap-3">
          <IconGlobe className="w-8 h-8 text-[#58a6ff]" />
          <div>
            <h2 className="text-sm font-semibold text-[#e6edf3]">Shannon-b1</h2>
            <p className="text-xs text-[#8b949e]">Workstation</p>
          </div>
        </div>
        {isMobile && (
          <button
            onClick={() => setSidebarOpen(false)}
            className="p-1.5 rounded-md hover:bg-[#21262d] transition-base text-[#8b949e]"
          >
            <IconX className="w-5 h-5" />
          </button>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-3 px-3 space-y-1 overflow-y-auto">
        <p className="px-3 mb-1 text-xs font-semibold text-[#8b949e] uppercase tracking-wider">Navigation</p>
        {tabs.map((tab) => {
          const TabIcon = tab.icon
          const isActive = activeTab === tab.id
          return (
            <button
              key={tab.id}
              onClick={() => { setActiveTab(tab.id); if (isMobile) setSidebarOpen(false) }}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-base ${
                isActive
                  ? 'bg-[#1f6feb]/15 text-[#e6edf3]'
                  : 'text-[#c9d1d9] hover:bg-[#161b22] hover:text-[#e6edf3]'
              }`}
            >
              <TabIcon className={`w-4 h-4 ${isActive ? 'text-[#58a6ff]' : 'text-[#8b949e]'}`} />
              <span>{tab.name}</span>
            </button>
          )
        })}

      </nav>

      {/* Sidebar footer - status */}
      <div className="p-4 border-t border-[#21262d]">
        <div className="rounded-md bg-[#0d1117] p-3 text-xs">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[#8b949e]">API Status</span>
            <div className={`w-2 h-2 rounded-full ${status?.status === 'running' ? 'bg-[#3fb950]' : 'bg-[#f85149]'}`} />
          </div>
          <p className="text-sm font-medium text-[#c9d1d9]">
            {status?.status === 'running' ? 'Online' : 'Offline'}
          </p>
        </div>
      </div>
    </div>
  )

  return (
    <div className="min-h-screen bg-[#0d1117] text-[#e6edf3] flex">
      {/* PC: persistent sidebar */}
      {!isMobile && (
        <aside className="w-64 flex-shrink-0 border-r border-[#21262d] bg-[#0d1117]">
          {sidebarContent}
        </aside>
      )}

      {/* Mobile overlay */}
      {isMobile && sidebarOpen && (
        <div className="fixed inset-0 bg-black/60 z-40 animate-fade-in" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Mobile sidebar drawer */}
      {isMobile && (
        <aside
          className={`fixed top-0 left-0 h-full w-72 bg-[#0d1117] border-r border-[#21262d] shadow-2xl z-50 transform transition-transform duration-300 ease-in-out ${
            sidebarOpen ? 'translate-x-0' : '-translate-x-full'
          }`}
        >
          {sidebarContent}
        </aside>
      )}

      {/* Main content area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top navbar */}
        <header className="sticky top-0 z-30 bg-[#0d1117] border-b border-[#21262d]">
          <div className="flex items-center justify-between h-14 px-4">
            <div className="flex items-center gap-3">
              {isMobile && (
                <button
                  onClick={() => setSidebarOpen(true)}
                  className="p-2 rounded-md hover:bg-[#161b22] transition-base text-[#8b949e]"
                  aria-label="Open menu"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                  </svg>
                </button>
              )}
              <IconGlobe className="w-7 h-7 text-[#58a6ff]" />
              <div>
                <h1 className="text-sm font-semibold text-[#e6edf3]">Shannon-b1</h1>
                <p className="text-xs text-[#8b949e] hidden sm:block">AI Language Model Workstation</p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              {/* Status indicator */}
              <div className="hidden sm:flex items-center gap-2 px-3 py-1 rounded-full bg-[#161b22] border border-[#30363d] text-xs">
                <div className={`w-2 h-2 rounded-full ${status?.status === 'running' ? 'bg-[#3fb950]' : 'bg-[#f85149]'}`} />
                <span className="text-[#8b949e]">{status?.status === 'running' ? 'API Online' : 'API Offline'}</span>
              </div>

              {/* Quick load button */}
              <button
                onClick={handleQuickLoad}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md bg-[#1f6feb] hover:bg-[#388bfd] text-white transition-base shadow-sm"
              >
                <IconLightning className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Quick Load</span>
              </button>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 p-4 sm:p-6 lg:p-8 overflow-auto">
          <div className="animate-fade-in max-w-5xl mx-auto">{children}</div>
        </main>
      </div>
    </div>
  )
}

export default Layout