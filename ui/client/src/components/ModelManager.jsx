import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  IconDatabase,
  IconFolder,
  IconPackage,
  IconWarning,
  IconCheck,
  IconX,
  IconGear,
  IconTerminal,
  IconClock,
} from './Icons'

const ModelManager = ({ apiUrl, status }) => {
  const savedModelPath = localStorage.getItem('shannon_model_path') || '../../checkpoints/shannon_b1.pt'

  const [checkpoints, setCheckpoints] = useState([])
  const [modelPath, setModelPath] = useState(savedModelPath)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')

  useEffect(() => {
    fetchCheckpoints()
  }, [])

  useEffect(() => {
    localStorage.setItem('shannon_model_path', modelPath)
  }, [modelPath])

  const fetchCheckpoints = async () => {
    try {
      const res = await axios.get(`${apiUrl}/checkpoints`)
      setCheckpoints(res.data)
    } catch (error) {
      console.error('Failed to fetch checkpoints:', error)
    }
  }

  const loadModel = async () => {
    setLoading(true)
    setMessage('')
    try {
      const res = await axios.post(`${apiUrl}/model/load`, null, {
        params: { model_path: modelPath }
      })
      setMessage(`success:${res.data.message}`)
      setTimeout(() => setMessage(''), 4000)
    } catch (error) {
      setMessage(`error:${error.response?.data?.detail || error.message}`)
    } finally {
      setLoading(false)
    }
  }

  const isSuccess = message.startsWith('success:')
  const displayMessage = message.replace(/^(success|error):/, '')

  return (
    <div className="space-y-6">
      {/* Current model info */}
      <div className="bg-[#0d1117] rounded-lg p-6 border border-[#21262d]">
        <h2 className="text-lg font-semibold text-[#e6edf3] mb-5 flex items-center gap-2">
          <IconDatabase className="w-5 h-5 text-[#8b949e]" />
          Current Model
        </h2>

        {status.model_loaded ? (
          <div className="bg-[#238636]/10 border border-[#3fb950]/20 rounded-lg p-5">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-2.5 h-2.5 bg-[#3fb950] rounded-full"></div>
              <span className="text-[#3fb950] text-sm font-semibold">Model Loaded</span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-[#161b22] rounded-lg p-3 border border-[#21262d]">
                <span className="text-[10px] text-[#8b949e] block mb-1">Vocab Size</span>
                <p className="text-lg font-semibold text-[#e6edf3] font-mono">{status.model_info?.vocab_size || '-'}</p>
              </div>
              <div className="bg-[#161b22] rounded-lg p-3 border border-[#21262d]">
                <span className="text-[10px] text-[#8b949e] block mb-1">Model Dim</span>
                <p className="text-lg font-semibold text-[#e6edf3] font-mono">{status.model_info?.d_model || '-'}</p>
              </div>
              <div className="bg-[#161b22] rounded-lg p-3 border border-[#21262d]">
                <span className="text-[10px] text-[#8b949e] block mb-1">Parameters</span>
                <p className="text-lg font-semibold text-[#e6edf3] font-mono">{status.model_info?.parameters?.toLocaleString() || '-'}</p>
              </div>
              <div className="bg-[#161b22] rounded-lg p-3 border border-[#21262d]">
                <span className="text-[10px] text-[#8b949e] block mb-1">Device</span>
                <p className="text-lg font-semibold text-[#e6edf3] font-mono">{status.model_info?.device || '-'}</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-[#9e6a03]/10 border border-[#d29922]/20 rounded-lg p-5">
            <div className="flex items-center gap-3">
              <IconWarning className="w-5 h-5 text-[#d29922] flex-shrink-0" />
              <div>
                <p className="text-[#d29922] text-sm font-semibold">No model loaded</p>
                <p className="text-xs text-[#d29922]/70 mt-1">Select or enter a model path below to load</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Load model section */}
      <div className="bg-[#0d1117] rounded-lg p-6 border border-[#21262d]">
        <h2 className="text-lg font-semibold text-[#e6edf3] mb-5 flex items-center gap-2">
          <IconFolder className="w-5 h-5 text-[#8b949e]" />
          Load Model
        </h2>
        <div className="flex flex-col sm:flex-row gap-3">
          <input
            type="text"
            value={modelPath}
            onChange={(e) => setModelPath(e.target.value)}
            className="flex-1 px-3 py-2 bg-[#161b22] border border-[#30363d] rounded-md focus:ring-2 focus:ring-[#1f6feb] focus:border-transparent text-[#e6edf3] placeholder-[#484f58] text-sm transition-base"
            placeholder="Enter model file path..."
          />
          <button
            onClick={loadModel}
            disabled={loading}
            className="flex items-center justify-center gap-2 bg-[#1f6feb] hover:bg-[#388bfd] disabled:bg-[#21262d] disabled:cursor-not-allowed text-white font-semibold py-2 px-5 rounded-md transition-base text-sm whitespace-nowrap"
          >
            {loading ? (
              <>
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Loading...
              </>
            ) : (
              <>
                <IconPackage className="w-4 h-4" /> Load Model
              </>
            )}
          </button>
        </div>
        {message && (
          <div className={`mt-4 p-3 rounded-md text-xs animate-slide-in ${isSuccess
              ? 'bg-[#238636]/10 border border-[#3fb950]/20 text-[#3fb950]'
              : 'bg-[#da3633]/10 border border-[#f85149]/20 text-[#f85149]'
            }`}>
            <p className="flex items-center gap-1.5">
              {isSuccess ? <IconCheck className="w-3.5 h-3.5 flex-shrink-0" /> : <IconX className="w-3.5 h-3.5 flex-shrink-0" />}
              {displayMessage}
            </p>
          </div>
        )}
      </div>

      {/* Checkpoint list */}
      <div className="bg-[#0d1117] rounded-lg p-6 border border-[#21262d]">
        <div className="flex justify-between items-center mb-5">
          <h2 className="text-lg font-semibold text-[#e6edf3] flex items-center gap-2">
            <IconPackage className="w-5 h-5 text-[#8b949e]" />
            Checkpoints
          </h2>
          <button
            onClick={fetchCheckpoints}
            className="flex items-center gap-1.5 text-xs bg-[#21262d] hover:bg-[#30363d] text-[#c9d1d9] py-1.5 px-3 rounded-md transition-base"
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Refresh
          </button>
        </div>

        <div className="space-y-2 max-h-96 overflow-y-auto pr-1">
          {checkpoints.length === 0 ? (
            <div className="text-center py-12">
              <IconPackage className="w-12 h-12 mx-auto mb-3 opacity-20" />
              <p className="text-sm text-[#8b949e]">No checkpoints found</p>
              <p className="text-[10px] text-[#484f58] mt-1">Checkpoints will appear here after training</p>
            </div>
          ) : (
            checkpoints.map((ckpt, idx) => (
              <div
                key={idx}
                className="flex flex-col sm:flex-row sm:justify-between sm:items-center bg-[#161b22] hover:bg-[#161b22] rounded-md p-4 border border-[#21262d] hover:border-[#30363d] transition-base cursor-pointer"
                onClick={() => setModelPath(ckpt.path)}
              >
                <div className="flex-1 mb-2 sm:mb-0">
                  <p className="font-mono text-xs text-[#e6edf3] font-medium break-all">{ckpt.name}</p>
                  <div className="flex items-center gap-3 mt-1 flex-wrap">
                    <span className="text-[10px] text-[#8b949e] flex items-center gap-1">
                      <IconDatabase className="w-3 h-3" />
                      {ckpt.size_mb} MB
                    </span>
                    <span className="text-[10px] text-[#8b949e] flex items-center gap-1">
                      <IconClock className="w-3 h-3" />
                      {new Date(ckpt.modified).toLocaleDateString()}
                    </span>
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    setModelPath(ckpt.path)
                  }}
                  className="ml-0 sm:ml-4 px-3 py-1.5 bg-[#1f6feb]/10 hover:bg-[#1f6feb]/20 text-[#58a6ff] rounded-md transition-base text-xs font-medium border border-[#1f6feb]/20 self-start sm:self-center"
                >
                  Select
                </button>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

export default ModelManager