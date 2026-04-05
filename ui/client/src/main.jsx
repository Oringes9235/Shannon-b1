import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

/**
 * 应用程序入口点
 * 创建React根节点并渲染应用组件到DOM中
 * 使用严格模式来帮助发现潜在问题
 */
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)