export interface Widget {
  id: string;
  widget_type: string;
  label: string;
  icon: string;
  position: number;
  enabled: boolean;
  config: Record<string, unknown>;
}

export interface AvailableWidget {
  id: string;
  type: string;
  label: string;
  icon: string;
  description: string;
  requires_auth: boolean;
  auth_provider?: string;
}

export interface PieMenuConfig {
  widgets: Widget[];
  activation_gesture: string;
  activation_delay_ms: number;
  selection_delay_ms: number;
}

export interface ConnectedAccount {
  provider: string;
  email: string | null;
  connected: boolean;
}

export interface GeneralConfig {
  use_depthai_hardware: boolean;
  virtual_camera_enabled: boolean;
  show_nerd_stats: boolean;
  mirror_display: boolean;
  mirror_virtual_ui: boolean;
}

export interface AppConfig {
  pie_menu: PieMenuConfig;
  connected_accounts: ConnectedAccount[];
  general: GeneralConfig;
}

export type TabId = 'camera' | 'widgets' | 'accounts' | 'settings';
