import { useState } from 'react';
import { open } from '@tauri-apps/plugin-shell';
import { CheckCircle, XCircle, ExternalLink, Loader2 } from 'lucide-react';
import type { AppConfig } from '../types';
import { useTokens } from '../hooks/useConfig';

// Google OAuth configuration - user needs to set up their own credentials
const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID || '';
const GOOGLE_REDIRECT_URI = 'http://localhost:8765/callback';
const GOOGLE_SCOPES = [
  'https://www.googleapis.com/auth/calendar.readonly',
  'https://www.googleapis.com/auth/userinfo.email',
].join(' ');

interface AccountsPanelProps {
  config: AppConfig;
  onUpdate: (config: AppConfig) => void;
}

function GoogleIcon() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24">
      <path
        fill="#4285F4"
        d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
      />
      <path
        fill="#34A853"
        d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
      />
      <path
        fill="#FBBC05"
        d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
      />
      <path
        fill="#EA4335"
        d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
      />
    </svg>
  );
}

export function AccountsPanel({ config, onUpdate }: AccountsPanelProps) {
  const [connecting, setConnecting] = useState<string | null>(null);
  const { deleteToken } = useTokens();

  const googleAccount = config.connected_accounts.find((a) => a.provider === 'google');

  const connectGoogle = async () => {
    if (!GOOGLE_CLIENT_ID) {
      alert(
        'Google OAuth is not configured. Please set VITE_GOOGLE_CLIENT_ID in your environment.'
      );
      return;
    }

    setConnecting('google');

    const authUrl = new URL('https://accounts.google.com/o/oauth2/v2/auth');
    authUrl.searchParams.set('client_id', GOOGLE_CLIENT_ID);
    authUrl.searchParams.set('redirect_uri', GOOGLE_REDIRECT_URI);
    authUrl.searchParams.set('response_type', 'code');
    authUrl.searchParams.set('scope', GOOGLE_SCOPES);
    authUrl.searchParams.set('access_type', 'offline');
    authUrl.searchParams.set('prompt', 'consent');

    try {
      await open(authUrl.toString());
      // In a real implementation, you'd start a local server to receive the callback
      // and exchange the code for tokens. For now, we'll just show a placeholder.
      alert(
        'OAuth flow started. In a full implementation, a local server would handle the callback.'
      );
    } catch (err) {
      console.error('Failed to open browser:', err);
    } finally {
      setConnecting(null);
    }
  };

  const disconnectGoogle = async () => {
    try {
      await deleteToken('google_access_token');
      await deleteToken('google_refresh_token');

      const newAccounts = config.connected_accounts.filter((a) => a.provider !== 'google');
      onUpdate({ ...config, connected_accounts: newAccounts });
    } catch (err) {
      console.error('Failed to disconnect Google:', err);
    }
  };

  const isGoogleConnected = googleAccount?.connected ?? false;

  return (
    <div className="section">
      <div className="section-header">
        <div>
          <div className="section-title">Connected Accounts</div>
          <div className="section-description">
            Connect your accounts to enable widgets that display your data
          </div>
        </div>
      </div>

      <div className="card-grid">
        <div className="account-card">
          <div className="account-icon">
            <GoogleIcon />
          </div>
          <div className="account-info">
            <div className="account-provider">Google</div>
            {isGoogleConnected && googleAccount?.email ? (
              <div className="account-email">{googleAccount.email}</div>
            ) : (
              <div className="account-email">Calendar, email access</div>
            )}
            <div className={`account-status ${isGoogleConnected ? 'connected' : 'disconnected'}`}>
              {isGoogleConnected ? (
                <>
                  <CheckCircle size={12} />
                  Connected
                </>
              ) : (
                <>
                  <XCircle size={12} />
                  Not connected
                </>
              )}
            </div>
          </div>
          {isGoogleConnected ? (
            <button className="btn btn-danger btn-sm" onClick={disconnectGoogle}>
              Disconnect
            </button>
          ) : (
            <button
              className="btn btn-primary btn-sm"
              onClick={connectGoogle}
              disabled={connecting === 'google'}
            >
              {connecting === 'google' ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                <ExternalLink size={14} />
              )}
              Connect
            </button>
          )}
        </div>
      </div>

      <div className="card" style={{ marginTop: 24, padding: 16 }}>
        <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
          <strong style={{ color: 'var(--text-primary)' }}>Privacy Note:</strong> Your account
          credentials are stored securely in your system's keychain (macOS Keychain, Windows
          Credential Manager, or Linux Secret Service). Data is fetched directly from the service
          APIs and is never sent to any third-party servers.
        </div>
      </div>
    </div>
  );
}
