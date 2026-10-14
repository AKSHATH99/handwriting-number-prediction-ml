import { useState, useRef, useEffect } from 'react'
import './index.css'

function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [status, setStatus] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [mode, setMode] = useState('draw') // 'draw' | 'upload'
  const canvasRef = useRef(null)
  const [isDrawing, setIsDrawing] = useState(false)

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]
      setFile(selectedFile)
      
      const reader = new FileReader()
      reader.onload = (e) => setPreviewUrl(e.target.result)
      reader.readAsDataURL(selectedFile)
      
      setStatus(null)
      setResult(null)
    }
  }

  const handleUpload = () => {
    if (!file) return
    uploadFile(file)
  }

  const handleClear = () => {
    setFile(null)
    setPreviewUrl(null)
    setStatus(null)
    setResult(null)
    // Clear canvas
    const canvas = canvasRef.current
    if (canvas) {
      const ctx = canvas.getContext('2d')
      ctx.fillStyle = '#ffffff'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
    }
  }

  // Drawing handlers
  const getCanvasCoords = (e) => {
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const clientX = e.touches ? e.touches[0].clientX : e.clientX
    const clientY = e.touches ? e.touches[0].clientY : e.clientY
    return {
      x: (clientX - rect.left) * (canvas.width / rect.width),
      y: (clientY - rect.top) * (canvas.height / rect.height)
    }
  }

  const startDrawing = (e) => {
    e.preventDefault()
    setIsDrawing(true)
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const { x, y } = getCanvasCoords(e)
    ctx.beginPath()
    ctx.moveTo(x, y)
  }

  const draw = (e) => {
    if (!isDrawing) return
    e.preventDefault()
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const { x, y } = getCanvasCoords(e)
    ctx.lineWidth = 32  // Thicker stroke: scales to ~3px at 28x28 (MNIST standard)
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.strokeStyle = '#000000'
    ctx.lineTo(x, y)
    ctx.stroke()
  }

  const stopDrawing = () => {
    setIsDrawing(false)
  }

  const handleCanvasPredict = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
    const hasDrawing = imageData.data.some((v, i) => i % 4 === 3 && v > 0)
    
    if (!hasDrawing) {
      setStatus({ type: 'error', message: 'Draw a digit first!' })
      return
    }

    // Send canvas directly to backend - let backend do proper MNIST preprocessing
    canvas.toBlob((blob) => {
      if (!blob) {
        setStatus({ type: 'error', message: 'Failed to process drawing' })
        return
      }
      const testFile = new File([blob], 'drawing.png', { type: 'image/png' })
      setFile(testFile)
      setPreviewUrl(canvas.toDataURL('image/png'))
      // Pass file directly to avoid React state timing issue
      uploadFile(testFile)
    }, 'image/png')
  }

  const uploadFile = async (fileToUpload) => {
    setLoading(true)
    setStatus(null)
    
    const formData = new FormData()
    formData.append('file', fileToUpload)

    try {
      const res = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      })
      const data = await res.json()
      
      if (res.ok) {
        setResult({ prediction: data.prediction, confidence: data.confidence })
        setStatus({ type: 'success', message: `${data.message} Confidence: ${(data.confidence * 100).toFixed(1)}%` })
      } else {
        setStatus({ type: 'error', message: data.detail || data.error || 'Upload failed' })
      }
    } catch (err) {
      setStatus({ type: 'error', message: 'Failed to connect to backend.' })
    } finally {
      setLoading(false)
    }
  }

  // Initialize canvas with white background
  useEffect(() => {
    const canvas = canvasRef.current
    if (canvas) {
      const ctx = canvas.getContext('2d')
      ctx.fillStyle = '#ffffff'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
    }
  }, [])

  return (
    <div className="container">
      <header>
        <h1>Handwritten Digit Recognition</h1>
        <p>Draw a digit or upload an image</p>
      </header>

      <main>
        {/* Mode Switcher */}
        <div className="mode-switcher">
          <button
            className={mode === 'draw' ? 'mode-btn active' : 'mode-btn'}
            onClick={() => { setMode('draw'); handleClear(); }}
          >
            ✏️ Draw
          </button>
          <button
            className={mode === 'upload' ? 'mode-btn active' : 'mode-btn'}
            onClick={() => { setMode('upload'); handleClear(); }}
          >
            📁 Upload
          </button>
        </div>

        <div className="upload-container">
          {mode === 'draw' ? (
            <div className="canvas-wrapper">
              <canvas
                ref={canvasRef}
                width={280}
                height={280}
                className="drawing-canvas"
                onMouseDown={startDrawing}
                onMouseMove={draw}
                onMouseUp={stopDrawing}
                onMouseLeave={stopDrawing}
                onTouchStart={startDrawing}
                onTouchMove={draw}
                onTouchEnd={stopDrawing}
              />
            </div>
          ) : (
            !previewUrl ? (
              <label className="upload-dropzone">
                <input 
                  type="file" 
                  accept="image/*" 
                  onChange={handleFileChange} 
                  style={{ display: 'none' }} 
                />
                <div className="upload-placeholder">
                  <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="upload-icon">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="17 8 12 3 7 8"></polyline>
                    <line x1="12" y1="3" x2="12" y2="15"></line>
                  </svg>
                  <p>Click to select an image</p>
                </div>
              </label>
            ) : (
              <div className="preview-container">
                <img src={previewUrl} alt="Preview" className="image-preview" />
              </div>
            )
          )}
        </div>

        <div className="controls">
          {mode === 'draw' && (
            <button onClick={handleClear} className="btn-clear" disabled={loading}>
              Clear Canvas
            </button>
          )}
          {mode === 'upload' && (
            <button onClick={handleClear} className="btn-clear" disabled={!previewUrl || loading}>
              Clear
            </button>
          )}
          <button 
            onClick={mode === 'draw' ? handleCanvasPredict : handleUpload} 
            className="btn-predict" 
            disabled={loading || (mode === 'upload' && !file)}
          >
            {loading ? 'Predicting...' : mode === 'draw' ? 'Predict Drawing' : 'Upload Image'}
          </button>
        </div>

        {result && (
          <div className="prediction-result">
            <span className="prediction-digit">{result.prediction}</span>
            <span className="prediction-label">Predicted digit</span>
            <div className="confidence-bar">
              <div
                className="confidence-fill"
                style={{ width: `${(result.confidence * 100).toFixed(1)}%` }}
              />
            </div>
            <span className="confidence-text">
              {(result.confidence * 100).toFixed(1)}% confidence
            </span>
          </div>
        )}

        {status && (
          <div className={`status-message ${status.type}`}>
            {status.message}
          </div>
        )}
      </main>
    </div>
  )
}

export default App