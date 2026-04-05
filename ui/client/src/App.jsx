import React, { useState, useEffect } from 'react'
import axios from 'axios'
import Dashboard from './components/Dashboard'
import TrainingMonitor from './components/TrainingMonitor'
import TextGenerator from './components/TextGenerator'
import ModelManager from './components/ModelManager'
import Layout from './components/Layout'

// API基础URL地址
const API_URL = 'http://localhost:8000/api'

/**
 * 主应用程序组件
 * 负责管理应用的整体状态和路由切换
 * @returns {JSX.Element} 应用程序主界面
 */
function App() {
  const [activeTab, setActiveTab] = useState('generate')
  const [status, setStatus] = useState({})
  const [wsConnected, setWsConnected] = useState(false)

  useEffect(() => {
    // 获取服务器状态
    fetchStatus()
    const interval = setInterval(fetchStatus, 5000)
    return () => clearInterval(interval)
  }, [])

  /**
   * 异步获取服务器状态信息
   * 定期向API发送请求以更新服务器运行状态
   * @async
   * @function fetchStatus
   * @returns {void}
   */
  const fetchStatus = async () => {
    try {
      const res = await axios.get(`${API_URL}/status`)
      setStatus(res.data)
    } catch (error) {
      console.error('Failed to fetch status:', error)
    }
  }

  return (
    <Layout apiUrl={API_URL} status={status} activeTab={activeTab} setActiveTab={setActiveTab}>
      {activeTab === 'generate' && <TextGenerator apiUrl={API_URL} status={status} />}
      {activeTab === 'training' && <TrainingMonitor apiUrl={API_URL} />}
      {activeTab === 'models' && <ModelManager apiUrl={API_URL} status={status} />}
      {activeTab === 'dashboard' && <Dashboard apiUrl={API_URL} status={status} />}
    </Layout>
  )
}

export default App