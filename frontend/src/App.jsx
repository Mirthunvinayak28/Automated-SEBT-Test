import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import ProtocolInstructions from './components/ProtocolInstructions';
import LegMeasurementForm from './components/LegMeasurementForm';
import VideoUploadCard from './components/VideoUploadCard';
import ProcessingStatus from './components/ProcessingStatus';
import VideoPlayerSection from './components/VideoPlayerSection';
import MovementsTable from './components/MovementsTable';
import CompositeScoresCard from './components/CompositeScoresCard';
import CalibrationObservations from './components/CalibrationObservations';
import ClinicalLimitations from './components/ClinicalLimitations';
import DownloadSection from './components/DownloadSection';

export default function App() {
  // Step 1: Protocol Acknowledgement
  const [isAcknowledged, setIsAcknowledged] = useState(false);

  // Step 2: Clinical Leg Lengths (cm)
  const [leftLeg, setLeftLeg] = useState('90.0');
  const [rightLeg, setRightLeg] = useState('91.0');

  // Step 3: Video File Selection
  const [selectedFile, setSelectedFile] = useState(null);

  // Job and Processing State
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [resultsData, setResultsData] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);

  // Validation
  const isLeftLegValid = parseFloat(leftLeg) > 0;
  const isRightLegValid = parseFloat(rightLeg) > 0;
  const canStartAnalysis = isAcknowledged && isLeftLegValid && isRightLegValid && !!selectedFile;

  // Poll Job Status when running
  useEffect(() => {
    if (!jobId) return;

    let isMounted = true;
    let pollInterval = null;

    const checkStatus = async () => {
      try {
        const res = await fetch(`/api/sebt/jobs/${jobId}`);
        if (!res.ok) {
          throw new Error(`Status check failed: HTTP ${res.status}`);
        }
        const data = await res.json();
        if (!isMounted) return;

        setJobStatus(data);

        if (data.status === 'completed') {
          clearInterval(pollInterval);
          fetchResults(jobId);
        } else if (data.status === 'failed') {
          clearInterval(pollInterval);
          setError(data.error || 'The analysis pipeline encountered an error.');
        }
      } catch (err) {
        if (!isMounted) return;
        console.error('Polling error:', err);
      }
    };

    // If already completed (e.g. loaded sample), fetch immediately
    if (jobStatus?.status === 'completed') {
      fetchResults(jobId);
      return;
    }

    checkStatus();
    pollInterval = setInterval(checkStatus, 1200);

    return () => {
      isMounted = false;
      if (pollInterval) clearInterval(pollInterval);
    };
  }, [jobId]);

  // Fetch full results
  const fetchResults = async (id) => {
    try {
      const res = await fetch(`/api/sebt/jobs/${id}/results`);
      if (res.ok) {
        const json = await res.json();
        setResultsData(json.data);
      } else {
        const errJson = await res.json().catch(() => ({}));
        setError(errJson.detail || 'Failed to retrieve analysis results.');
      }
    } catch (err) {
      console.error('Results fetch error:', err);
      setError('Network error while retrieving analysis results.');
    }
  };

  // Start analysis upload
  const handleStartAnalysis = async () => {
    if (!canStartAnalysis || isUploading) return;

    setError(null);
    setResultsData(null);
    setJobStatus(null);
    setIsUploading(true);

    const formData = new FormData();
    formData.append('video', selectedFile);
    formData.append('left_leg', leftLeg);
    formData.append('right_leg', rightLeg);

    try {
      const res = await fetch('/api/sebt/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Upload failed with HTTP ${res.status}`);
      }

      const data = await res.json();
      setJobId(data.job_id);
      setJobStatus({
        status: 'queued',
        progress: {
          percent: 5,
          stage: 'Video uploaded successfully. Initializing pipeline...',
          message: 'Dispatching background analysis worker.',
        },
      });

      // Scroll smoothly down to the processing section
      setTimeout(() => {
        const el = document.getElementById('processing-status-section');
        if (el) el.scrollIntoView({ behavior: 'smooth' });
      }, 200);
    } catch (err) {
      console.error('Analysis submission error:', err);
      setError(err.message || 'An error occurred while uploading the video.');
    } finally {
      setIsUploading(false);
    }
  };

  // Reset session
  const handleReset = () => {
    setJobId(null);
    setJobStatus(null);
    setResultsData(null);
    setSelectedFile(null);
    setError(null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const isAnalysisComplete = jobStatus?.status === 'completed' && resultsData;

  return (
    <div className="app-container">
      {/* 1. Header */}
      <Header
        onReset={handleReset}
        currentJobId={jobId}
      />

      {/* 2. Step 1: Protocol Instructions */}
      <ProtocolInstructions
        isAcknowledged={isAcknowledged}
        onToggleAcknowledge={setIsAcknowledged}
      />

      {/* 3. Step 2: Clinical Leg Measurements */}
      <LegMeasurementForm
        leftLeg={leftLeg}
        rightLeg={rightLeg}
        onChangeLeft={setLeftLeg}
        onChangeRight={setRightLeg}
        disabled={jobStatus?.status === 'processing'}
      />

      {/* 4. Step 3: Video Upload Card */}
      <VideoUploadCard
        selectedFile={selectedFile}
        onSelectFile={setSelectedFile}
        onRemoveFile={() => setSelectedFile(null)}
        onStartAnalysis={handleStartAnalysis}
        canStartAnalysis={canStartAnalysis}
        isAcknowledged={isAcknowledged}
        leftLegValid={isLeftLegValid}
        rightLegValid={isRightLegValid}
        isUploading={isUploading}
      />

      {/* 5. Step 4: Real-Time Processing Status */}
      {(jobStatus || error) && (
        <ProcessingStatus statusData={jobStatus} error={error} />
      )}

      {/* Results Sections: Only displayed when analysis completes */}
      {isAnalysisComplete && (
        <>
          {/* 6. Step 5: Processed Video — Depth Visualization */}
          <VideoPlayerSection
            jobId={jobId}
            videoFilename={jobStatus?.video_filename || selectedFile?.name}
          />

          {/* 7. Step 6: Individual Movement Results (Trial-by-Trial) */}
          <MovementsTable movements={resultsData.movements} />

          {/* 8. Step 7: Overall Composite SEBT Results */}
          <CompositeScoresCard compositeScores={resultsData.summary?.composite_scores} />

          {/* 9. Step 8: Calibration & Experimental Observations */}
          <CalibrationObservations
            observations={resultsData.observations}
            summary={resultsData.summary}
          />

          {/* 10. Step 9: Interpretation and Clinical Limitations */}
          <ClinicalLimitations summary={resultsData.summary} />

          {/* 11. Step 10: Download Artifacts & Final Report */}
          <DownloadSection
            jobId={jobId}
            videoFilename={jobStatus?.video_filename || selectedFile?.name}
            summary={resultsData.summary}
            observations={resultsData.observations}
          />
        </>
      )}
    </div>
  );
}
