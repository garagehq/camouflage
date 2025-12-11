import { useState } from 'react';
import { Hand, LayoutGrid, User, Settings, Loader2, Camera } from 'lucide-react';
import { useConfig } from './hooks/useConfig';
import { PieMenuEditor } from './components/PieMenuEditor';
import { AccountsPanel } from './components/AccountsPanel';
import { SettingsPanel } from './components/SettingsPanel';
import { CameraControl } from './components/CameraControl';
import type { TabId } from './types';
import './index.css';

function App() {
  const [activeTab, setActiveTab] = useState<TabId>('camera');
  const { config, availableWidgets, loading, error, updateConfig } = useConfig();

  const connectedProviders = config.connected_accounts
    .filter((a) => a.connected)
    .map((a) => a.provider);

  if (loading) {
    return (
      <div className="app">
        <div
          style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexDirection: 'column',
            gap: 16,
          }}
        >
          <Loader2 size={32} className="animate-spin" style={{ color: 'var(--accent)' }} />
          <div style={{ color: 'var(--text-secondary)' }}>Loading configuration...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-title">
          <Hand size={24} />
          Camouflage
        </div>
      </header>

      <nav className="nav">
        <button
          className={`nav-item ${activeTab === 'camera' ? 'active' : ''}`}
          onClick={() => setActiveTab('camera')}
        >
          <Camera size={18} />
          Camera
        </button>
        <button
          className={`nav-item ${activeTab === 'widgets' ? 'active' : ''}`}
          onClick={() => setActiveTab('widgets')}
        >
          <LayoutGrid size={18} />
          Widgets
        </button>
        <button
          className={`nav-item ${activeTab === 'accounts' ? 'active' : ''}`}
          onClick={() => setActiveTab('accounts')}
        >
          <User size={18} />
          Accounts
        </button>
        <button
          className={`nav-item ${activeTab === 'settings' ? 'active' : ''}`}
          onClick={() => setActiveTab('settings')}
        >
          <Settings size={18} />
          Settings
        </button>
      </nav>

      <main className="main">
        {error && (
          <div
            style={{
              padding: 12,
              marginBottom: 16,
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid var(--error)',
              borderRadius: 'var(--radius-md)',
              color: 'var(--error)',
              fontSize: 13,
            }}
          >
            {error}
          </div>
        )}

        {activeTab === 'camera' && (
          <CameraControl config={config.general} />
        )}

        {activeTab === 'widgets' && (
          <PieMenuEditor
            config={config.pie_menu}
            availableWidgets={availableWidgets}
            connectedProviders={connectedProviders}
            onUpdate={(pieMenu) => updateConfig({ ...config, pie_menu: pieMenu })}
          />
        )}

        {activeTab === 'accounts' && (
          <AccountsPanel config={config} onUpdate={updateConfig} />
        )}

        {activeTab === 'settings' && (
          <SettingsPanel
            config={config.general}
            onUpdate={(general) => updateConfig({ ...config, general })}
          />
        )}
      </main>
    </div>
  );
}

export default App;
