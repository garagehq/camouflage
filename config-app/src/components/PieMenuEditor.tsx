import { useState } from 'react';
import {
  DndContext,
  DragOverlay,
  useDraggable,
  useDroppable,
} from '@dnd-kit/core';
import type { DragEndEvent, DragStartEvent } from '@dnd-kit/core';
import {
  Pencil,
  Eye,
  Calendar,
  Cloud,
  Bell,
  Clock,
  Lock,
  Hand,
  Pause,
  X,
} from 'lucide-react';
import type { Widget, AvailableWidget, PieMenuConfig } from '../types';

const iconMap: Record<string, React.ComponentType<{ size?: number }>> = {
  pencil: Pencil,
  eye: Eye,
  calendar: Calendar,
  cloud: Cloud,
  bell: Bell,
  clock: Clock,
  hand: Hand,
  pause: Pause,
};

interface PieMenuEditorProps {
  config: PieMenuConfig;
  availableWidgets: AvailableWidget[];
  connectedProviders: string[];
  onUpdate: (config: PieMenuConfig) => void;
}

function getSlotPosition(index: number, total: number, radius: number) {
  const angleOffset = -Math.PI / 2; // Start from top
  const angle = angleOffset + (2 * Math.PI * index) / total;
  return {
    x: Math.cos(angle) * radius,
    y: Math.sin(angle) * radius,
  };
}

function DraggableWidget({ widget, isInMenu }: { widget: AvailableWidget; isInMenu: boolean }) {
  const { attributes, listeners, setNodeRef, isDragging } = useDraggable({
    id: widget.id,
    data: { widget, source: 'gallery' },
    disabled: isInMenu,
  });

  const Icon = iconMap[widget.icon] || Clock;

  return (
    <div
      ref={setNodeRef}
      {...listeners}
      {...attributes}
      className={`gallery-item ${isInMenu ? 'disabled' : ''} ${isDragging ? 'dragging' : ''}`}
      style={{ opacity: isDragging ? 0.5 : 1 }}
    >
      <div className="gallery-item-icon">
        <Icon size={18} />
      </div>
      <div className="gallery-item-info">
        <div className="gallery-item-label">{widget.label}</div>
        {widget.requires_auth && (
          <div className="gallery-item-auth">
            <Lock size={10} />
            Requires {widget.auth_provider}
          </div>
        )}
      </div>
    </div>
  );
}

function PieSlot({
  index,
  widget,
  position,
  onRemove,
}: {
  index: number;
  widget: Widget | null;
  position: { x: number; y: number };
  onRemove: () => void;
}) {
  const { setNodeRef, isOver } = useDroppable({
    id: `slot-${index}`,
    data: { slotIndex: index },
  });

  const Icon = widget ? iconMap[widget.icon] || Clock : null;

  return (
    <div
      ref={setNodeRef}
      className={`pie-slot ${widget ? 'filled' : ''} ${isOver ? 'drop-target' : ''}`}
      style={{
        left: `calc(50% + ${position.x}px - 28px)`,
        top: `calc(50% + ${position.y}px - 28px)`,
      }}
    >
      {Icon && <Icon size={22} />}
      {widget && (
        <button
          className="btn-icon"
          onClick={onRemove}
          style={{
            position: 'absolute',
            top: -8,
            right: -8,
            width: 20,
            height: 20,
            padding: 0,
            background: 'var(--error)',
            color: 'white',
            borderRadius: '50%',
          }}
        >
          <X size={12} />
        </button>
      )}
    </div>
  );
}

export function PieMenuEditor({
  config,
  availableWidgets,
  connectedProviders,
  onUpdate,
}: PieMenuEditorProps) {
  const [activeWidget, setActiveWidget] = useState<AvailableWidget | null>(null);
  const maxSlots = 6;
  const radius = 100;

  const widgetsInMenu = new Set(config.widgets.map((w) => w.id));

  const handleDragStart = (event: DragStartEvent) => {
    const widget = event.active.data.current?.widget as AvailableWidget;
    setActiveWidget(widget);
  };

  const handleDragEnd = (event: DragEndEvent) => {
    setActiveWidget(null);
    const { active, over } = event;

    if (!over) return;

    const slotIndex = over.data.current?.slotIndex as number | undefined;
    if (slotIndex === undefined) return;

    const widget = active.data.current?.widget as AvailableWidget;
    if (!widget) return;

    // Check if widget requires auth and provider is not connected
    if (widget.requires_auth && widget.auth_provider) {
      if (!connectedProviders.includes(widget.auth_provider)) {
        alert(`Please connect your ${widget.auth_provider} account first.`);
        return;
      }
    }

    // Add widget to slot
    const newWidgets = [...config.widgets.filter((w) => w.position !== slotIndex)];
    newWidgets.push({
      id: widget.id,
      widget_type: widget.type,
      label: widget.label,
      icon: widget.icon,
      position: slotIndex,
      enabled: true,
      config: {},
    });

    onUpdate({ ...config, widgets: newWidgets });
  };

  const removeWidget = (position: number) => {
    const newWidgets = config.widgets.filter((w) => w.position !== position);
    onUpdate({ ...config, widgets: newWidgets });
  };

  const slots = Array.from({ length: maxSlots }, (_, i) => {
    const widget = config.widgets.find((w) => w.position === i) || null;
    const pos = getSlotPosition(i, maxSlots, radius);
    return { index: i, widget, position: pos };
  });

  const Icon = activeWidget ? iconMap[activeWidget.icon] || Clock : null;

  return (
    <DndContext onDragStart={handleDragStart} onDragEnd={handleDragEnd}>
      <div className="section">
        <div className="section-header">
          <div>
            <div className="section-title">Pie Menu Layout</div>
            <div className="section-description">
              Drag widgets from the gallery to the pie menu slots
            </div>
          </div>
        </div>

        <div className="pie-preview-container">
          <div className="pie-preview">
            <div className="pie-center">
              <Hand size={24} />
            </div>
            {slots.map((slot) => (
              <PieSlot
                key={slot.index}
                index={slot.index}
                widget={slot.widget}
                position={slot.position}
                onRemove={() => removeWidget(slot.index)}
              />
            ))}
          </div>

          <div className="widget-gallery">
            <div className="section-title" style={{ marginBottom: 12 }}>
              Available Widgets
            </div>
            <div className="gallery-grid">
              {availableWidgets.map((widget) => (
                <DraggableWidget
                  key={widget.id}
                  widget={widget}
                  isInMenu={widgetsInMenu.has(widget.id)}
                />
              ))}
            </div>
          </div>
        </div>

        <DragOverlay>
          {activeWidget && Icon && (
            <div className="gallery-item" style={{ opacity: 0.8 }}>
              <div className="gallery-item-icon">
                <Icon size={18} />
              </div>
              <div className="gallery-item-info">
                <div className="gallery-item-label">{activeWidget.label}</div>
              </div>
            </div>
          )}
        </DragOverlay>
      </div>

      <div className="section">
        <div className="section-title">Pie Menu Settings</div>
        <div className="card" style={{ marginTop: 12 }}>
          <div className="settings-row">
            <div>
              <div className="settings-label">Activation Delay</div>
              <div className="settings-description">
                How long to hold fist before menu appears
              </div>
            </div>
            <select
              className="select"
              value={config.activation_delay_ms}
              onChange={(e) =>
                onUpdate({ ...config, activation_delay_ms: parseInt(e.target.value) })
              }
            >
              <option value={300}>0.3s</option>
              <option value={500}>0.5s</option>
              <option value={750}>0.75s</option>
              <option value={1000}>1s</option>
            </select>
          </div>
          <div className="settings-row">
            <div>
              <div className="settings-label">Selection Delay</div>
              <div className="settings-description">
                How long to hover over icon to select
              </div>
            </div>
            <select
              className="select"
              value={config.selection_delay_ms}
              onChange={(e) =>
                onUpdate({ ...config, selection_delay_ms: parseInt(e.target.value) })
              }
            >
              <option value={300}>0.3s</option>
              <option value={500}>0.5s</option>
              <option value={750}>0.75s</option>
              <option value={1000}>1s</option>
            </select>
          </div>
        </div>
      </div>
    </DndContext>
  );
}
