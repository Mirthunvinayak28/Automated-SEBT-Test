import React, { useRef, useState } from 'react';
import { UploadCloud, FileVideo, Trash2, Play, AlertCircle, FileCheck } from 'lucide-react';

export default function VideoUploadCard({
  selectedFile,
  onSelectFile,
  onRemoveFile,
  onStartAnalysis,
  canStartAnalysis,
  isAcknowledged,
  leftLegValid,
  rightLegValid,
  isUploading
}) {
  const fileInputRef = useRef(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [fileError, setFileError] = useState(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (file) => {
    setFileError(null);
    if (!file.name.toLowerCase().endsWith('.mp4')) {
      setFileError('Invalid file format. Please upload an MP4 recording.');
      return;
    }
    const maxSizeBytes = 250 * 1024 * 1024;
    if (file.size > maxSizeBytes) {
      setFileError('File size exceeds the 250MB limit.');
      return;
    }
    onSelectFile(file);
  };

  const videoPreviewUrl = selectedFile ? URL.createObjectURL(selectedFile) : null;

  return (
    <section className="dashboard-card" id="upload-video-section">
      <div className="card-header">
        <div className="card-header-title">
          <span className="card-step-badge">Step 3</span>
          <h2>Upload SEBT Video</h2>
        </div>
      </div>

      {!selectedFile ? (
        <>
          <div
            className={`dropzone ${isDragOver ? 'is-dragover' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="video/mp4"
              onChange={handleFileChange}
              style={{ display: 'none' }}
              id="video-file-picker"
            />
            <div className="dropzone-icon">
              <UploadCloud size={28} />
            </div>
            <h3>Drag & Drop your SEBT MP4 video here</h3>
            <p>or click to browse from your computer (Standard MP4 format, max 250MB)</p>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={(e) => {
                e.stopPropagation();
                fileInputRef.current?.click();
              }}
            >
              <FileVideo size={16} />
              Select MP4 Video
            </button>
          </div>

          {fileError && (
            <div style={{ marginTop: '12px', color: 'var(--danger-text)', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.88rem' }}>
              <AlertCircle size={16} color="var(--danger-dot)" />
              <span>{fileError}</span>
            </div>
          )}
        </>
      ) : (
        <div>
          {/* Selected File Details */}
          <div className="selected-file-card">
            <div className="file-info">
              <div className="file-icon">
                <FileVideo size={22} />
              </div>
              <div>
                <div className="file-name">{selectedFile.name}</div>
                <div className="file-meta">
                  {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • MP4 Video Ready for Analysis
                </div>
              </div>
            </div>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                type="button"
                className="btn btn-sm btn-secondary"
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
              >
                Change Video
              </button>
              <button
                type="button"
                className="btn btn-sm btn-secondary"
                onClick={onRemoveFile}
                disabled={isUploading}
                style={{ color: 'var(--danger-text)' }}
              >
                <Trash2 size={14} />
                Remove
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="video/mp4"
                onChange={handleFileChange}
                style={{ display: 'none' }}
              />
            </div>
          </div>

          {/* Local Video Preview */}
          {videoPreviewUrl && (
            <div style={{ marginTop: '16px', background: '#000000', borderRadius: 'var(--radius-md)', overflow: 'hidden', maxHeight: '360px', display: 'flex', justifyContent: 'center' }}>
              <video
                src={videoPreviewUrl}
                controls
                style={{ maxHeight: '360px', width: 'auto', maxWidth: '100%' }}
              />
            </div>
          )}
        </div>
      )}

      {/* Validation Checklist Requirements before Starting */}
      <div style={{ marginTop: '22px', borderTop: '1px solid var(--border-subtle)', paddingTop: '18px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
        <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: isAcknowledged ? 'var(--success-text)' : 'var(--text-muted)' }}>
            <FileCheck size={14} color={isAcknowledged ? 'var(--success-dot)' : 'var(--text-light)'} />
            Protocol & safety guidelines acknowledged: {isAcknowledged ? 'Yes' : 'Pending Step 1'}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: (leftLegValid && rightLegValid) ? 'var(--success-text)' : 'var(--text-muted)' }}>
            <FileCheck size={14} color={(leftLegValid && rightLegValid) ? 'var(--success-dot)' : 'var(--text-light)'} />
            Valid clinical leg measurements entered: {(leftLegValid && rightLegValid) ? 'Yes' : 'Pending Step 2'}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: selectedFile ? 'var(--success-text)' : 'var(--text-muted)' }}>
            <FileCheck size={14} color={selectedFile ? 'var(--success-dot)' : 'var(--text-light)'} />
            MP4 Video selected: {selectedFile ? 'Yes' : 'Pending Step 3'}
          </div>
        </div>

        <button
          id="start-analysis-btn"
          type="button"
          className="btn btn-primary btn-lg"
          onClick={onStartAnalysis}
          disabled={!canStartAnalysis || isUploading}
        >
          <Play size={18} fill="currentColor" />
          {isUploading ? 'Uploading & Initializing...' : 'Start SEBT Analysis'}
        </button>
      </div>
    </section>
  );
}
