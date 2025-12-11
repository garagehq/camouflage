import { useState, useEffect, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import type { AppConfig, AvailableWidget } from '../types';

const defaultConfig: AppConfig = {
  pie_menu: {
    widgets: [],
    activation_gesture: 'fist',
    activation_delay_ms: 500,
    selection_delay_ms: 500,
  },
  connected_accounts: [],
  general: {
    use_depthai_hardware: true,
    virtual_camera_enabled: false,
    show_nerd_stats: true,
    mirror_display: true,
    mirror_virtual_ui: true,
  },
};

export function useConfig() {
  const [config, setConfig] = useState<AppConfig>(defaultConfig);
  const [availableWidgets, setAvailableWidgets] = useState<AvailableWidget[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        const [loadedConfig, widgets] = await Promise.all([
          invoke<AppConfig>('load_config'),
          invoke<AvailableWidget[]>('get_available_widgets'),
        ]);
        setConfig(loadedConfig);
        setAvailableWidgets(widgets);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load config');
        console.error('Failed to load config:', err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  const saveConfig = useCallback(async (newConfig: AppConfig) => {
    setSaving(true);
    setError(null);
    try {
      await invoke('save_config', { config: newConfig });
      setConfig(newConfig);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save config');
      console.error('Failed to save config:', err);
    } finally {
      setSaving(false);
    }
  }, []);

  const updateConfig = useCallback((updates: Partial<AppConfig>) => {
    const newConfig = { ...config, ...updates };
    setConfig(newConfig);
    saveConfig(newConfig);
  }, [config, saveConfig]);

  return {
    config,
    availableWidgets,
    loading,
    saving,
    error,
    updateConfig,
    saveConfig,
  };
}

export function useTokens() {
  const storeToken = useCallback(async (provider: string, token: string) => {
    await invoke('store_token', { provider, token });
  }, []);

  const getToken = useCallback(async (provider: string): Promise<string | null> => {
    return await invoke<string | null>('get_token', { provider });
  }, []);

  const deleteToken = useCallback(async (provider: string) => {
    await invoke('delete_token', { provider });
  }, []);

  return { storeToken, getToken, deleteToken };
}
