package websocket

// SupportsVMCopy reports whether this handler has a VM copy backend wired.
func (h *WebSocketHandler) SupportsVMCopy() bool {
	return h != nil && h.copyService != nil
}

// SupportsVMPortForward reports whether this handler has a VM port-forward backend wired.
func (h *WebSocketHandler) SupportsVMPortForward() bool {
	return h != nil && h.portService != nil
}
