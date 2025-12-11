import { useState, useEffect, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Play, Square, Eye, EyeOff, Pencil, RefreshCw, Skull, Clock, Pause } from 'lucide-react';
import type { GeneralConfig } from '../types';

interface CameraControlProps {
  config: GeneralConfig;
}

interface CameraArgs {
  use_depthai: boolean;
  virtual_cam: boolean;
  hide_extras: boolean;
  draw_mode: boolean;
}

export function CameraControl({ config }: CameraControlProps) {
  const [isRunning, setIsRunning] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasAutoStarted, setHasAutoStarted] = useState(false);

  // Track toggle states for quick actions (independent of config)
  const [drawActive, setDrawActive] = useState(false);
  const [nerdStatsActive, setNerdStatsActive] = useState(config.show_nerd_stats);
  const [timerActive, setTimerActive] = useState(false);
  const [avoidGesturesActive, setAvoidGesturesActive] = useState(false);

  const startCamera = useCallback(async () => {
    setError(null);
    setIsStarting(true);

    const args: CameraArgs = {
      use_depthai: config.use_depthai_hardware,
      virtual_cam: config.virtual_camera_enabled,
      hide_extras: !config.show_nerd_stats,
      draw_mode: false,
    };

    // Reset toggle states to match initial config
    setDrawActive(false);
    setNerdStatsActive(config.show_nerd_stats);
    setTimerActive(false);
    setAvoidGesturesActive(false);

    try {
      await invoke('start_camera', { args });
      // Status will be updated by the polling
    } catch (err) {
      setError(String(err));
      setIsStarting(false);
    }
  }, [config]);

  const stopCamera = useCallback(async () => {
    setError(null);
    try {
      await invoke('stop_camera');
      setIsRunning(false);
    } catch (err) {
      setError(String(err));
    }
  }, []);

  const sendCommand = useCallback(async (command: string) => {
    try {
      await invoke('send_camera_command', { command });
    } catch (err) {
      console.error('Failed to send command:', err);
    }
  }, []);

  const forceStop = useCallback(async () => {
    setError(null);
    try {
      const result = await invoke<string>('force_stop_camera');
      console.log('Force stop result:', result);
      setIsRunning(false);
      setIsStarting(false);
      setHasAutoStarted(false); // Allow auto-start again after force stop
    } catch (err) {
      setError(String(err));
    }
  }, []);

  // Poll camera status
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const running = await invoke<boolean>('get_camera_status');
        setIsRunning(running);
        if (running) {
          setIsStarting(false);
        }
      } catch (err) {
        console.error('Failed to get camera status:', err);
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  // Auto-start camera on mount
  useEffect(() => {
    if (hasAutoStarted) return;

    const autoStart = async () => {
      // Check if already running first
      try {
        const running = await invoke<boolean>('get_camera_status');
        if (!running) {
          setHasAutoStarted(true);
          // Small delay to ensure config is loaded
          setTimeout(() => {
            startCamera();
          }, 500);
        }
      } catch (err) {
        console.error('Failed to check camera status for auto-start:', err);
      }
    };

    autoStart();
  }, [startCamera, hasAutoStarted]);

  return (
    <div className="section">
      <div className="section-header">
        <div>
          <div className="section-title">Camera Control</div>
          <div className="section-description">
            Start and stop the hand tracking camera
          </div>
        </div>
      </div>

      <div className="card" style={{ padding: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 20 }}>
          <div
            style={{
              width: 12,
              height: 12,
              borderRadius: '50%',
              background: isRunning ? 'var(--success)' : isStarting ? 'var(--warning)' : 'var(--text-tertiary)',
              boxShadow: isRunning ? '0 0 8px var(--success)' : 'none',
            }}
          />
          <div>
            <div style={{ fontWeight: 500, color: 'var(--text-primary)' }}>
              {isRunning ? 'Camera Running' : isStarting ? 'Starting...' : 'Camera Stopped'}
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>
              {config.use_depthai_hardware ? 'DepthAI Hardware Mode' : 'Webcam Mode'}
              {config.virtual_camera_enabled && ' • Virtual Camera'}
            </div>
          </div>
        </div>

        {error && (
          <div
            style={{
              padding: 12,
              marginBottom: 16,
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid var(--error)',
              borderRadius: 'var(--radius-sm)',
              color: 'var(--error)',
              fontSize: 13,
            }}
          >
            {error}
          </div>
        )}

        <div style={{ display: 'flex', gap: 12 }}>
          {!isRunning && !isStarting ? (
            <button className="btn btn-primary" onClick={startCamera} style={{ flex: 1 }}>
              <Play size={16} />
              Start Camera
            </button>
          ) : isStarting ? (
            <button className="btn" disabled style={{ flex: 1 }}>
              <RefreshCw size={16} className="animate-spin" />
              Starting...
            </button>
          ) : (
            <button className="btn btn-danger" onClick={stopCamera} style={{ flex: 1 }}>
              <Square size={16} />
              Stop Camera
            </button>
          )}
          <button
            className="btn"
            onClick={forceStop}
            title="Force kill all demo.py processes"
            style={{ background: 'var(--bg-tertiary)' }}
          >
            <Skull size={16} />
            Force Stop
          </button>
        </div>

        {isRunning && (
          <div style={{ marginTop: 20, paddingTop: 20, borderTop: '1px solid var(--border)' }}>
            <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 12 }}>
              Quick Actions
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <button
                className={`btn btn-sm ${drawActive ? 'btn-active' : ''}`}
                onClick={() => {
                  sendCommand('draw');
                  setDrawActive(!drawActive);
                }}
                title="Toggle draw mode"
              >
                <Pencil size={14} />
                Draw
              </button>
              <button
                className={`btn btn-sm ${nerdStatsActive ? 'btn-active' : ''}`}
                onClick={() => {
                  sendCommand(nerdStatsActive ? 'hide' : 'show');
                  setNerdStatsActive(!nerdStatsActive);
                }}
                title="Toggle nerd stats display"
              >
                {nerdStatsActive ? <Eye size={14} /> : <EyeOff size={14} />}
                Nerd Stats
              </button>
              <button
                className={`btn btn-sm ${timerActive ? 'btn-active' : ''}`}
                onClick={() => {
                  sendCommand('timer');
                  setTimerActive(!timerActive);
                }}
                title="Toggle timer widget"
              >
                <Clock size={14} />
                Timer
              </button>
              <button
                className={`btn btn-sm ${avoidGesturesActive ? 'btn-active' : ''}`}
                onClick={() => {
                  sendCommand('avoid_gestures');
                  setAvoidGesturesActive(!avoidGesturesActive);
                }}
                title="Pause gestures until dual PEACE signs"
              >
                <Pause size={14} />
                Pause Gestures
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="card" style={{ marginTop: 16, padding: 16 }}>
        <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
          <strong style={{ color: 'var(--text-primary)' }}>Tip:</strong> The camera settings from
          the Settings tab will be applied when you start the camera. To change modes (DepthAI vs
          Webcam, Virtual Camera), stop and restart the camera.
        </div>
      </div>
    </div>
  );
}
