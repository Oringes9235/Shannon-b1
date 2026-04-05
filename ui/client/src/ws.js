let socket = null
let listeners = []

function makeWsUrl(base) {
  try {
    const u = new URL(base)
    return (u.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + u.host + '/ws'
  } catch (e) {
    return 'ws://localhost:8000/ws'
  }
}

export function initSocket(apiUrl) {
  if (socket) return socket
  const url = makeWsUrl(apiUrl)
  socket = new WebSocket(url)
  socket.onopen = () => console.log('Shared WebSocket connected')
  socket.onclose = () => console.log('Shared WebSocket closed')
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

export function subscribe(cb, apiUrl) {
  initSocket(apiUrl)
  listeners.push(cb)
  return () => {
    listeners = listeners.filter((f) => f !== cb)
    // keep socket open for session lifetime
  }
}

export function getSocket() {
  return socket
}

export default { initSocket, subscribe, getSocket }
