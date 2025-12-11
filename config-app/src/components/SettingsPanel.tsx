import type { GeneralConfig } from '../types';

interface SettingsPanelProps {
  config: GeneralConfig;
  onUpdate: (config: GeneralConfig) => void;
}

function Toggle({ active, onChange }: { active: boolean; onChange: (value: boolean) => void }) {
  return (
    <div className={`toggle ${active ? 'active' : ''}`} onClick={() => onChange(!active)}>
      <div className="toggle-knob" />
    </div>
  );
}

export function SettingsPanel({ config, onUpdate }: SettingsPanelProps) {
  const update = (key: keyof GeneralConfig, value: boolean) => {
    onUpdate({ ...config, [key]: value });
  };

  return (
    <div className="section">
      <div className="section-header">
        <div>
          <div className="section-title">General Settings</div>
          <div className="section-description">Configure how Camouflage runs</div>
        </div>
      </div>

      <div className="card">
        <div className="settings-group">
          <div className="settings-group-title">Hardware</div>
          <div className="settings-row">
            <div>
              <div className="settings-label">Use DepthAI Hardware</div>
              <div className="settings-description">
                Use OAK-D camera instead of webcam. Enables edge mode and depth features.
              </div>
            </div>
            <Toggle
              active={config.use_depthai_hardware}
              onChange={(v) => update('use_depthai_hardware', v)}
            />
          </div>
        </div>

        <div className="settings-group">
          <div className="settings-group-title">Output</div>
          <div className="settings-row">
            <div>
              <div className="settings-label">Virtual Camera</div>
              <div className="settings-description">
                Output to virtual camera for use in Zoom, Meet, etc. Requires OBS.
              </div>
            </div>
            <Toggle
              active={config.virtual_camera_enabled}
              onChange={(v) => update('virtual_camera_enabled', v)}
            />
          </div>
          <div className="settings-row">
            <div>
              <div className="settings-label">Mirror Display</div>
              <div className="settings-description">
                Mirror the camera feed like a real mirror (left-right flipped)
              </div>
            </div>
            <Toggle
              active={config.mirror_display}
              onChange={(v) => update('mirror_display', v)}
            />
          </div>
          <div className="settings-row">
            <div>
              <div className="settings-label">Mirror Virtual UI</div>
              <div className="settings-description">
                Flip text and icons so they appear correctly when display is mirrored
              </div>
            </div>
            <Toggle
              active={config.mirror_virtual_ui}
              onChange={(v) => update('mirror_virtual_ui', v)}
            />
          </div>
        </div>

        <div className="settings-group">
          <div className="settings-group-title">Overlay</div>
          <div className="settings-row">
            <div>
              <div className="settings-label">Show Nerd Stats</div>
              <div className="settings-description">Display FPS, hand landmarks, and other debug info on screen</div>
            </div>
            <Toggle active={config.show_nerd_stats} onChange={(v) => update('show_nerd_stats', v)} />
          </div>
        </div>
      </div>
    </div>
  );
}
