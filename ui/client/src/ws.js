let socket = null
let listeners = []

/**
 * 根据基础URL生成WebSocket连接地址
 * @param {string} base - 基础API URL
 * @returns {string} WebSocket连接地址，如果解析失败则返回默认本地地址
 */
function makeWsUrl(base) {
  try {
    const u = new URL(base)
    return (u.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + u.host + '/ws'
  } catch (e) {
    return 'ws://localhost:8000/ws'
  }
}

/**
 * 初始化WebSocket连接（单例模式）
 * @param {string} apiUrl - API服务器地址
 * @returns {WebSocket} WebSocket实例
 */
export function initSocket(apiUrl) {
  if (socket) return socket
  const url = makeWsUrl(apiUrl)
  socket = new WebSocket(url)
  socket.onopen = () => console.log('Shared WebSocket connected')
  socket.onclose = () => console.log('Shared WebSocket closed')
  
  // 处理接收到的消息并分发给所有订阅者
  socket.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data)
      listeners.forEach((cb) => {
        try { cb(data) } catch (e) { console.error(e) }
      })
    } catch (e) {
      console.error('Invalid WS message', e)
    }
  }
  return socket
}

/**
 * 订阅WebSocket消息
 * @param {Function} cb - 消息处理回调函数
 * @param {string} apiUrl - API服务器地址
 * @returns {Function} 取消订阅函数
 */
export function subscribe(cb, apiUrl) {
  initSocket(apiUrl)
  listeners.push(cb)
  return () => {
    listeners = listeners.filter((f) => f !== cb)
    // keep socket open for session lifetime
  }
}

/**
 * 获取当前WebSocket实例
 * @returns {WebSocket|null} WebSocket实例或null
 */
export function getSocket() {
  return socket
}

export default { initSocket, subscribe, getSocket }