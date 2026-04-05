import React, { useState, useEffect } from 'react'
import axios from 'axios'
import Dashboard from './components/Dashboard'
import TrainingMonitor from './components/TrainingMonitor'
import TextGenerator from './components/TextGenerator'
import ModelManager from './components/ModelManager'
import Layout from './components/Layout'

const API_URL = 'http://localhost:8000/api'

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