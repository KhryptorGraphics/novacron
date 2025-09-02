export const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8090/ws/events/v1";

export function connectEvents(onWelcome?: (msg: unknown) => void): WebSocket {
  try {
    const ws = new WebSocket(WS_URL);
    let first = true;
    
    ws.addEventListener("open", () => console.log("WS connected"));
    ws.addEventListener("error", (error) => console.warn("WS error:", error));
    ws.addEventListener("close", () => console.log("WS disconnected"));
    ws.addEventListener("message", (ev) => {
      try {
        const payload = JSON.parse(ev.data as string);
        console.log("WS message:", payload);
        if (first) {
          first = false;
          onWelcome?.(payload);
        }
      } catch (e) {
        console.warn("WS message parse error", e);
      }
    });
    
    return ws;
  } catch (error) {
    console.error("Failed to create WebSocket:", error);
    // Return a mock WebSocket that won't cause crashes
    return {
      close: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      send: () => {},
      readyState: WebSocket.CLOSED
    } as WebSocket;
  }
}

